"""Launch SGLang with benchmark-only graph-node attribution in spawned workers."""

from graph_node_map import install

install()

if __name__ == "__main__":
    import os
    import sys

    from sglang.launch_server import load_plugins, prepare_server_args, run_server
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    args = prepare_server_args(sys.argv[1:])
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
