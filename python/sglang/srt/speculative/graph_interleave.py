"""Experimental communication-aware composition of two independent CUDA DAGs.

Each graph retains its capture-time storage and dependency order. Internal edges
default to full completion dependencies, with an opt-in to preserve their types.
A draft segment is
bounded by the target segment before/after its assigned slot. Opposite phases
may execute together; communication/communication and draft compute/communication
cannot. These fences can lengthen target windows, so measure end-to-end latency.
"""

from collections import deque

import torch


def is_communication_kernel(name):
    # NCCL plus the TP hidden-shard all-gather used by the logits processor.
    # The latter lives in distributed/device_communicators/triton_symm_mem_ag.py.
    return "nccl" in name.lower() or "_all_gather_kernel_inner" in name


def place_segments(target_kinds, draft_kinds):
    """Spread ordered draft segments over compatible target slots.

    The desired positions are uniform by segment count, not GPU time. Multiple
    consecutive compute segments can share a communication window. A change of
    draft phase must advance to a later target slot. Reject insufficient slots
    instead of silently allowing a forbidden overlap.
    """
    latest = [0] * len(draft_kinds)
    upper = len(target_kinds) - 1
    for index in range(len(draft_kinds) - 1, -1, -1):
        if (
            index + 1 < len(draft_kinds)
            and draft_kinds[index] != draft_kinds[index + 1]
        ):
            upper -= 1
        while upper >= 0 and target_kinds[upper] == draft_kinds[index]:
            upper -= 1
        latest[index] = upper
    positions = []
    lower = 0
    for index, kind in enumerate(draft_kinds):
        if index and kind != draft_kinds[index - 1]:
            lower = positions[-1] + 1
        upper = latest[index]
        candidates = [
            slot for slot in range(lower, upper + 1) if target_kinds[slot] != kind
        ]
        if not candidates:
            raise ValueError("Not enough opposite-phase target slots for the draft")
        desired = (index + 0.5) * len(target_kinds) / len(draft_kinds) - 0.5
        chosen = min(candidates, key=lambda slot: abs(slot - desired))
        positions.append(chosen)
        lower = chosen
    return positions


class GraphPhases:
    """Topological, homogeneous graph segments; originals must remain alive."""

    def __init__(self, graph, compute_chunks=1):
        if compute_chunks < 1:
            raise ValueError("Compute chunk count must be positive")
        from cuda.bindings import driver

        self.driver = driver
        self.original = graph
        self.graphs = []
        self.kinds = []
        self.node_counts = []
        self.kernel_names = []
        self.pairs = {}
        raw = driver.CUgraph(graph.raw_cuda_graph())
        _, count = self.check(driver.cuGraphGetNodes(raw, 0))
        nodes, _ = self.check(driver.cuGraphGetNodes(raw, count))
        _, _, _, count = self.check(driver.cuGraphGetEdges(raw, 0))
        sources, destinations, _, _ = self.check(driver.cuGraphGetEdges(raw, count))
        by_id = {int(node): node for node in nodes}
        successors = {key: [] for key in by_id}
        degrees = dict.fromkeys(by_id, 0)
        for source, destination in zip(sources, destinations):
            successors[int(source)].append(int(destination))
            degrees[int(destination)] += 1
        ready = deque(key for key in by_id if degrees[key] == 0)
        ordered = []
        while ready:
            key = ready.popleft()
            ordered.append(key)
            for child in successors[key]:
                degrees[child] -= 1
                if degrees[child] == 0:
                    ready.append(child)
        if len(ordered) != len(nodes):
            raise ValueError("Captured graph contains a dependency cycle")
        groups = []
        for key in ordered:
            node = by_id[key]
            (node_type,) = self.check(driver.cuGraphNodeGetType(node))
            # Child graphs and host nodes hide operations; refuse to classify.
            if int(node_type) not in {0, 1, 2, 5, 6, 7}:
                raise ValueError(f"Unsupported phase graph node type: {node_type}")
            kind = (
                groups[-1][0] if groups and int(node_type) in {5, 6, 7} else "compute"
            )
            name = ""
            if int(node_type) == 0:
                (params,) = self.check(driver.cuGraphKernelNodeGetParams(node))
                (name,) = self.check(driver.cuFuncGetName(params.func))
                name = name.decode()
                if is_communication_kernel(name):
                    kind = "communication"
            self.kernel_names.append(name)
            if not groups or kind != groups[-1][0]:
                groups.append((kind, []))
            groups[-1][1].append(key)
        if not any(kind == "communication" for kind, _ in groups):
            raise ValueError("Phase scheduling requires visible NCCL kernel nodes")
        try:
            for kind, keys in groups:
                count = min(compute_chunks, len(keys)) if kind == "compute" else 1
                for index in range(count):
                    keep = set(
                        keys[
                            index * len(keys) // count : (index + 1)
                            * len(keys)
                            // count
                        ]
                    )
                    (clone,) = self.check(driver.cuGraphClone(raw))
                    self.graphs.append(clone)
                    self.kinds.append(kind)
                    self.node_counts.append(len(keep))
                    for key, node in by_id.items():
                        if key not in keep:
                            (copy,) = self.check(
                                driver.cuGraphNodeFindInClone(node, clone)
                            )
                            self.check(driver.cuGraphDestroyNode(copy))
        except BaseException:
            self.close()
            raise

    def check(self, result):
        status, *values = result
        if status != self.driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA graph operation failed: {status}")
        return tuple(values)

    def pair(self, draft, serial=False, preserve_internal_pdl=False):
        key = (draft, serial, preserve_internal_pdl)
        if key not in self.pairs:
            self.pairs[key] = InterleavedGraph(
                self, draft, serial, preserve_internal_pdl
            )
        return self.pairs[key]

    def close(self):
        for pair in self.pairs.values():
            pair.close()
        self.pairs.clear()
        for graph in self.graphs:
            self.check(self.driver.cuGraphDestroy(graph))
        self.graphs.clear()


class InterleavedGraph:
    """A single launch, with fork/join dependencies at each occupied target slot."""

    def __init__(self, target, draft, serial=False, preserve_internal_pdl=False):
        self.target = target
        self.draft = draft
        self.driver = target.driver
        self.check = target.check
        self.executable = None
        self.node_roles = {}
        self.preserve_internal_pdl = preserve_internal_pdl
        self.preserved_programmatic_edges = 0
        self.positions = place_segments(target.kinds, draft.kinds)
        (self.graph,) = self.check(self.driver.cuGraphCreate(0))
        try:
            frontier = []
            for slot, target_graph in enumerate(target.graphs):
                target_frontier = self._append_segment(
                    target_graph, frontier, "target", slot, target.kinds[slot]
                )
                draft_frontier = target_frontier if serial else frontier
                used = False
                for index, position in enumerate(self.positions):
                    if position != slot:
                        continue
                    draft_frontier = self._append_segment(
                        draft.graphs[index],
                        draft_frontier,
                        "draft",
                        index,
                        draft.kinds[index],
                    )
                    used = True
                frontier = target_frontier + (draft_frontier if used else [])
            self._instantiate()
        except BaseException:
            self.close()
            raise

    def _append_segment(self, graph, frontier, role, segment, phase):
        """Copy a segment into the flat graph, preserving all internal edges."""
        driver = self.driver
        _, count = self.check(driver.cuGraphGetNodes(graph, 0))
        nodes, _ = self.check(driver.cuGraphGetNodes(graph, count))
        _, _, _, count = self.check(driver.cuGraphGetEdges(graph, 0))
        sources, destinations, edge_data, _ = self.check(
            driver.cuGraphGetEdges(graph, count)
        )
        parents = {int(node): [] for node in nodes}
        children = {int(node): [] for node in nodes}
        by_id = {int(node): node for node in nodes}
        for source, destination in zip(sources, destinations):
            parents[int(destination)].append(int(source))
            children[int(source)].append(int(destination))
        degrees = {key: len(value) for key, value in parents.items()}
        ready = deque(key for key, degree in degrees.items() if degree == 0)
        copies = {}
        (context,) = self.check(driver.cuCtxGetCurrent())
        while ready:
            key = ready.popleft()
            node = by_id[key]
            dependencies = [copies[parent] for parent in parents[key]] or frontier
            if self.preserve_internal_pdl and parents[key]:
                # Add typed internal dependencies after all nodes exist.
                dependencies = []
            args = (self.graph, dependencies, len(dependencies))
            (kind,) = self.check(driver.cuGraphNodeGetType(node))
            if int(kind) == 0:
                (params,) = self.check(driver.cuGraphKernelNodeGetParams(node))
                result = driver.cuGraphAddKernelNode(*args, params)
            elif int(kind) == 1:
                (params,) = self.check(driver.cuGraphMemcpyNodeGetParams(node))
                result = driver.cuGraphAddMemcpyNode(*args, params, context)
            elif int(kind) == 2:
                (params,) = self.check(driver.cuGraphMemsetNodeGetParams(node))
                result = driver.cuGraphAddMemsetNode(*args, params, context)
            elif int(kind) == 5:
                result = driver.cuGraphAddEmptyNode(*args)
            elif int(kind) == 6:
                (event,) = self.check(driver.cuGraphEventWaitNodeGetEvent(node))
                result = driver.cuGraphAddEventWaitNode(*args, event)
            elif int(kind) == 7:
                (event,) = self.check(driver.cuGraphEventRecordNodeGetEvent(node))
                result = driver.cuGraphAddEventRecordNode(*args, event)
            else:
                raise ValueError(f"Cannot compose graph node type {kind}")
            (copy,) = self.check(result)
            if int(kind) == 0:
                self.check(driver.cuGraphKernelNodeCopyAttributes(copy, node))
            copies[key] = copy
            self.node_roles[int(copy)] = {
                "role": role,
                "segment": segment,
                "phase": phase,
            }
            for child in children[key]:
                degrees[child] -= 1
                if degrees[child] == 0:
                    ready.append(child)
        if self.preserve_internal_pdl:
            if count:
                self.check(
                    driver.cuGraphAddDependencies(
                        self.graph,
                        [copies[int(node)] for node in sources],
                        [copies[int(node)] for node in destinations],
                        edge_data,
                        count,
                    )
                )
                self.preserved_programmatic_edges += sum(
                    int(edge.type) == 1 for edge in edge_data
                )
            # A programmatic dependency can release a successor before its
            # predecessor finishes. Waiting only for leaves would therefore
            # allow unfinished kernels to cross the phase boundary.
            return list(copies.values())
        return [copies[key] for key in copies if not children[key]]

    def _instantiate(self):
        (self.executable,) = self.check(self.driver.cuGraphInstantiate(self.graph, 0))

    def replay(self):
        stream = self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        self.check(self.driver.cuGraphLaunch(self.executable, stream))

    def close(self):
        if self.executable is not None:
            self.check(self.driver.cuGraphExecDestroy(self.executable))
            self.executable = None
        if self.graph is not None:
            self.check(self.driver.cuGraphDestroy(self.graph))
            self.graph = None
