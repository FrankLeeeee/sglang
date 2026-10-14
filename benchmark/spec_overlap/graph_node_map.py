"""Benchmark-only CUPTI clone tracking for exact joint-graph kernel attribution.

Subscribe only during graph construction, then unsubscribe before torch.profiler
starts. Clone callbacks map each executable node back to its target/draft source;
no kernel-name heuristic or undocumented graph-ID bit layout is used.
"""

import ctypes
import json
import os
from collections import Counter


class ResourceData(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_void_p) for name in ("context", "resource", "descriptor")
    ]


class GraphData(ctypes.Structure):
    _fields_ = [
        ("graph", ctypes.c_void_p),
        ("original_graph", ctypes.c_void_p),
        ("node", ctypes.c_void_p),
        ("original_node", ctypes.c_void_p),
        ("node_type", ctypes.c_int),
        ("dependency", ctypes.c_void_p),
        ("executable", ctypes.c_void_p),
    ]


def install():
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )
    from sglang.srt.speculative.graph_interleave import GraphPhases, InterleavedGraph

    # Kineto retains its CUPTI subscription after the first profile. Construct
    # every captured shape pair before that profile, while clone callbacks are
    # still available. This instrumentation is confined to the benchmark server.
    captured = {"target": [], "draft": []}
    original_capture = FullCudaGraphBackend.capture_one

    def tracked_capture(self, shape_key, *args, **kwargs):
        result = original_capture(self, shape_key, *args, **kwargs)
        if shape_key in self._speculative_phases:
            role = (
                "draft"
                if self._cuda_graph_runner.model_runner.is_draft_worker
                else "target"
            )
            model = self._cuda_graph_runner.model_runner.model
            if not captured[role] and type(model).__name__ == "GptOssForCausalLM":
                experts = model.model.layers[0].mlp.experts
                method = experts.quant_method
                weight = getattr(experts, "w13_weight", None)
                print(
                    "PHASE_MODEL_BACKEND "
                    + json.dumps(
                        {
                            "pid": os.getpid(),
                            "role": role,
                            "quant_method": type(method).__name__,
                            "expert_weight_dtype": str(getattr(weight, "dtype", None)),
                            "expert_weight_shape": list(getattr(weight, "shape", ())),
                            "moe_runner": str(
                                getattr(
                                    getattr(method, "runner", None),
                                    "runner_backend",
                                    getattr(method, "_fi_kernel", None),
                                )
                            ),
                        }
                    ),
                    flush=True,
                )
            phases = self._speculative_phases[shape_key]
            captured[role].append(phases)
            driver = phases.driver
            raw = driver.CUgraph(phases.original.raw_cuda_graph())
            _, _, _, count = phases.check(driver.cuGraphGetEdges(raw, 0))
            _, _, edge_data, _ = phases.check(driver.cuGraphGetEdges(raw, count))
            print(
                "PHASE_GRAPH_EDGES "
                + json.dumps(
                    {
                        "pid": os.getpid(),
                        "role": role,
                        "shape": str(shape_key),
                        "edge_types": dict(
                            Counter(int(edge.type) for edge in edge_data)
                        ),
                    }
                ),
                flush=True,
            )
        return result

    FullCudaGraphBackend.capture_one = tracked_capture
    original_pair = GraphPhases.pair
    prepared = set()

    def prepared_pair(self, draft, serial=False):
        if serial not in prepared:
            for target_phases in captured["target"]:
                for draft_phases in captured["draft"]:
                    original_pair(target_phases, draft_phases, serial)
            prepared.add(serial)
        return original_pair(self, draft, serial)

    GraphPhases.pair = prepared_pair

    original_instantiate = InterleavedGraph._instantiate

    def tracked_instantiate(self):
        cupti = ctypes.CDLL("libcupti.so.13")
        cupti.cuptiGetGraphNodeId.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        sources = dict(self.node_roles)
        nodes = {}
        errors = []
        callback_type = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p
        )

        @callback_type
        def callback(userdata, domain, callback_id, data):
            if domain != 3 or callback_id != 20:
                return
            resource = ctypes.cast(data, ctypes.POINTER(ResourceData)).contents
            graph = ctypes.cast(resource.descriptor, ctypes.POINTER(GraphData)).contents
            if graph.original_node in sources:
                role = sources[graph.original_node]
                sources[graph.node] = role
                value = ctypes.c_uint64()
                status = cupti.cuptiGetGraphNodeId(graph.node, ctypes.byref(value))
                if status:
                    errors.append(status)
                else:
                    nodes[value.value] = role

        subscriber = ctypes.c_void_p()
        cupti.cuptiSubscribe.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            callback_type,
            ctypes.c_void_p,
        ]
        status = cupti.cuptiSubscribe(ctypes.byref(subscriber), callback, None)
        if status:
            raise RuntimeError(f"CUPTI graph mapping subscription failed: {status}")
        try:
            status = cupti.cuptiEnableCallback(1, subscriber, 3, 20)
            if status:
                raise RuntimeError(f"CUPTI graph clone callback failed: {status}")
            original_instantiate(self)
        finally:
            cupti.cuptiUnsubscribe(subscriber)
        if errors or not nodes:
            raise RuntimeError(f"CUPTI graph mapping incomplete: {errors}")
        print(
            "PHASE_GRAPH_MAP "
            + json.dumps(
                {
                    "pid": os.getpid(),
                    "positions": self.positions,
                    "target_kinds": self.target.kinds,
                    "draft_kinds": self.draft.kinds,
                    "nodes": nodes,
                }
            ),
            flush=True,
        )

    InterleavedGraph._instantiate = tracked_instantiate
