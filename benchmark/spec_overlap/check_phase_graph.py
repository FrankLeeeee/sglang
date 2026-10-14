"""Two-rank NCCL graph composition equivalence (run with torchrun)."""

import argparse
import os

import torch
import torch.distributed as dist

from sglang.srt.speculative.graph_interleave import GraphPhases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preserve-internal-pdl", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    target_group = PyNcclCommunicator(dist.group.WORLD, device=rank)
    draft_group = PyNcclCommunicator(dist.new_group(), device=rank)
    target_group.disabled = False
    draft_group.disabled = False
    target_input = torch.full((1024,), 1.0 + rank, device="cuda")
    draft_input = torch.full((1024,), 3.0 + rank, device="cuda")

    def forward(x, layers, group):
        for _ in range(layers):
            x = x * 0.25
            x = x + 0.5
            group.all_reduce(x)
            x = x * 0.25
            x = x + 0.5
        return x

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward(target_input, 24, target_group)
            forward(draft_input, 2, draft_group)
    torch.cuda.synchronize()
    target = torch.cuda.CUDAGraph(keep_graph=True)
    draft = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(target, stream=stream):
        target_output = forward(target_input, 24, target_group)
    with torch.cuda.graph(draft, stream=stream):
        draft_output = forward(draft_input, 2, draft_group)
    # Nondefault attributes must survive graph composition. In real FA3
    # captures these include cluster dimensions; losing them can trap on GPU.
    from cuda.bindings import driver

    raw = driver.CUgraph(target.raw_cuda_graph())
    status, _, count = driver.cuGraphGetNodes(raw, 0)
    assert status == driver.CUresult.CUDA_SUCCESS
    status, nodes, _ = driver.cuGraphGetNodes(raw, count)
    assert status == driver.CUresult.CUDA_SUCCESS
    attribute = driver.CUkernelNodeAttrValue()
    attribute.priority = -1
    for node in nodes:
        status, kind = driver.cuGraphNodeGetType(node)
        assert status == driver.CUresult.CUDA_SUCCESS
        if int(kind) == 0:
            (status,) = driver.cuGraphKernelNodeSetAttribute(
                node, driver.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_PRIORITY, attribute
            )
            assert status == driver.CUresult.CUDA_SUCCESS
    for chunks in (1, 2, 4, 8):
        target_phases = GraphPhases(target)
        draft_phases = GraphPhases(draft, compute_chunks=chunks)
        try:
            for serial in (False, True):
                pair = target_phases.pair(
                    draft_phases,
                    serial=serial,
                    preserve_internal_pdl=args.preserve_internal_pdl,
                )
                for node, role in pair.node_roles.items():
                    node = driver.CUgraphNode(node)
                    status, kind = driver.cuGraphNodeGetType(node)
                    assert status == driver.CUresult.CUDA_SUCCESS
                    if int(kind) == 0 and role["role"] == "target":
                        status, copied = driver.cuGraphKernelNodeGetAttribute(
                            node, driver.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_PRIORITY
                        )
                        assert status == driver.CUresult.CUDA_SUCCESS
                        assert copied.priority == -1
                for value in (1.0, 3.0, 7.0):
                    target_input.fill_(value + rank)
                    draft_input.fill_(value + 2 + rank)
                    target.replay()
                    draft.replay()
                    expected_target = target_output.clone()
                    expected_draft = draft_output.clone()
                    pair.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        target_output, expected_target, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        draft_output, expected_draft, rtol=0, atol=0
                    )
                print(
                    f"PASS rank={rank} chunks={chunks} serial={serial} "
                    f"preserved_pdl_edges={pair.preserved_programmatic_edges} slots={pair.positions}",
                    flush=True,
                )
        finally:
            target_phases.close()
            draft_phases.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
