"""Experimental cuts of a captured CUDA DAG, retaining its original storage.

Ordered launches of topological chunks preserve every cross-cut dependency.
They add barriers at the cuts; each chunk still keeps its internal dependencies.
The original torch graph must outlive these graph executables and their buffers.
"""

from collections import deque

import torch


class SpeculativeGraphChunks:
    def __init__(self, graph, num_chunks=8):
        from cuda.bindings import runtime

        self.runtime = runtime
        self.original = graph
        self.executables = []
        self.graphs = []
        raw = runtime.cudaGraph_t(graph.raw_cuda_graph())
        _, count = self._check(runtime.cudaGraphGetNodes(raw))
        nodes, _ = self._check(runtime.cudaGraphGetNodes(raw, count))
        _, _, _, count = self._check(runtime.cudaGraphGetEdges(raw))
        sources, destinations, _, _ = self._check(runtime.cudaGraphGetEdges(raw, count))
        by_id = {int(node): node for node in nodes}
        degrees = dict.fromkeys(by_id, 0)
        successors = {key: [] for key in by_id}
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
        for node in nodes:
            (kind,) = self._check(runtime.cudaGraphNodeGetType(node))
            if int(kind) in {10, 11, 13}:
                raise ValueError(
                    "Graph chunks do not support allocation/free/conditional nodes"
                )
        num_chunks = min(num_chunks, len(ordered))
        self.node_counts = []
        try:
            for index in range(num_chunks):
                keep = set(
                    ordered[
                        index * len(ordered) // num_chunks : (index + 1)
                        * len(ordered)
                        // num_chunks
                    ]
                )
                (clone,) = self._check(runtime.cudaGraphClone(raw))
                self.graphs.append(clone)
                for key, node in by_id.items():
                    if key not in keep:
                        (copy,) = self._check(
                            runtime.cudaGraphNodeFindInClone(node, clone)
                        )
                        self._check(runtime.cudaGraphDestroyNode(copy))
                (executable,) = self._check(runtime.cudaGraphInstantiate(clone, 0))
                self.executables.append(executable)
                self.node_counts.append(len(keep))
        except BaseException:
            self.close()
            raise

    def _check(self, result):
        status, *values = result
        if status != self.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA graph operation failed: {status}")
        return tuple(values)

    def replay(self, index):
        stream = self.runtime.cudaStream_t(torch.cuda.current_stream().cuda_stream)
        self._check(self.runtime.cudaGraphLaunch(self.executables[index], stream))

    def close(self):
        for executable in self.executables:
            self._check(self.runtime.cudaGraphExecDestroy(executable))
        self.executables.clear()
        for graph in self.graphs:
            self._check(self.runtime.cudaGraphDestroy(graph))
        self.graphs.clear()
