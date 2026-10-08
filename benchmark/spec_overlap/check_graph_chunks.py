"""Check split CUDA DAG replay against its unsplit graph, including a fork."""

import torch

from sglang.srt.speculative.graph_chunks import SpeculativeGraphChunks


def main():
    capture = torch.cuda.Stream()
    branch = torch.cuda.Stream()
    values = torch.ones(128, device="cuda")
    capture.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=capture):
        first = values * 2
        branch.wait_stream(capture)
        with torch.cuda.stream(branch):
            left = first.square()
        right = first + 3
        capture.wait_stream(branch)
        result = left + right
    torch.cuda.synchronize()
    for count in (1, 3, 8):
        chunks = SpeculativeGraphChunks(graph, count)
        try:
            for value in (1, 3, 7):
                values.fill_(value)
                graph.replay()
                expected = result.clone()
                for index in range(len(chunks.executables)):
                    chunks.replay(index)
                torch.cuda.synchronize()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
            print(f"PASS chunks={len(chunks.executables)} nodes={chunks.node_counts}")
        finally:
            chunks.close()
    graph.reset()


if __name__ == "__main__":
    main()
