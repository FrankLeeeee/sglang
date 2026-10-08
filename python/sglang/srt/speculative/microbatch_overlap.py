"""Experimental EAGLE3 cross-request draft/verify continuations.

One stream owns all draft work; the caller stream owns all target work. Each
model remains serial with itself. Events express dependencies across models,
and the scheduler observes a single joined result in original request order.
"""

from contextlib import contextmanager
from copy import copy

import torch

from sglang.srt.distributed.parallel_state import (
    init_model_parallel_group,
    patch_tensor_parallel_group,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import (
    attention_backends,
    get_exec,
    get_forward,
    get_lora,
    get_parallel,
    get_resources,
    get_schedule,
    get_spec,
    get_stream,
)
from sglang.srt.speculative.spec_utils import (
    spec_stage_span,
)


def validate_microbatch_config(target_runner):
    if get_spec().speculative_microbatch_mode == "off":
        return
    if target_runner.model.__class__.__name__ not in {
        "LlamaForCausalLM",
        "GptOssForCausalLM",
    }:
        raise ValueError(
            "The microbatch prototype currently supports Llama and GPT-OSS targets only"
        )
    if attention_backends()[1] != "fa3":
        raise ValueError(
            "The microbatch prototype currently requires the fa3 attention backend"
        )
    if not (
        get_spec().speculative_algorithm == "EAGLE3"
        and (
            check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
            or (
                get_spec().speculative_microbatch_mode
                in {"fine_serial", "fine_overlap"}
                and check_cuda_graph_backend(Phase.DECODE, Backend.FULL)
            )
        )
        and check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
        and get_exec().comm.disable_custom_all_reduce
        and not get_exec().comm.enable_flashinfer_allreduce_fusion
        and get_schedule().disable_overlap_schedule
        and get_parallel().pp_size == 1
        and get_parallel().dp_size == 1
        and get_parallel().attn_cp_size == 1
        and get_parallel().moe_ep_size == 1
        and not get_parallel().attn_dp_enabled
        and get_spec().speculative_eagle_topk == 1
        and not get_spec().speculative_adaptive
        and not get_spec().speculative_use_rejection_sampling
        and not get_lora().enable_lora
        and not get_exec().graph.enable_torch_compile
    ):
        raise ValueError(
            "Experimental speculative microbatch execution requires EAGLE3, "
            "topk=1, PP=DP=CP=EP=1, no attention DP, disabled prefill graphs, eager or full decode graphs for fine modes, disabled custom all-reduce, "
            "disabled FlashInfer all-reduce fusion, disabled overlap scheduling, "
            "and no LoRA/adaptive/rejection sampling."
        )


def can_microbatch_overlap(worker, batch):
    """Restrict the prototype to independent greedy requests without side outputs."""
    return (
        get_spec().speculative_microbatch_mode != "off"
        and worker.speculative_algorithm.is_eagle3()
        and worker.device == "cuda"
        and get_parallel().pp_size == 1
        and get_parallel().dp_size == 1
        and get_parallel().attn_cp_size == 1
        and get_parallel().moe_ep_size == 1
        and not get_parallel().attn_dp_enabled
        and worker.topk == 1
        and worker.speculative_num_steps > 0
        and not get_spec().speculative_adaptive
        and not get_spec().speculative_use_rejection_sampling
        and batch.forward_mode.is_decode()
        and batch.spec_info is not None
        and batch.spec_info.is_draft_input()
        and batch.spec_info.future_indices is None
        and len(batch.reqs) >= 2
        and batch.sampling_info.is_all_greedy
        and not batch.sampling_info.has_custom_logit_processor
        and batch.sampling_info.acc_additive_penalties is None
        and batch.sampling_info.acc_scaling_penalties is None
        and batch.sampling_info.logit_bias is None
        and not any(batch.sampling_info.return_sampling_masks or [])
        and not batch.has_grammar
        and not batch.return_logprob
        and not batch.return_hidden_states
        and not any(
            req.return_routed_experts or req.return_indexer_topk for req in batch.reqs
        )
        and batch.beam_tail is None
        and batch.mamba_track_indices is None
        and (
            batch.multimodal_inputs is None
            or all(item is None for item in batch.multimodal_inputs)
        )
        and not get_exec().graph.enable_torch_compile
        and (
            check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
            or (
                get_spec().speculative_microbatch_mode
                in {"fine_serial", "fine_overlap"}
                and check_cuda_graph_backend(Phase.DECODE, Backend.FULL)
            )
        )
        and check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
        and get_exec().comm.disable_custom_all_reduce
    )


def _split(batch):
    midpoint = len(batch.reqs) // 2
    children = []
    for start, end in ((0, midpoint), (midpoint, len(batch.reqs))):
        rows = slice(start, end)
        child = copy(batch)
        child.reqs = batch.reqs[rows]
        for name in (
            "req_pool_indices",
            "req_pool_indices_cpu",
            "seq_lens",
            "orig_seq_lens",
            "seq_lens_cpu",
        ):
            value = getattr(batch, name)
            setattr(child, name, value[rows] if value is not None else None)
        child.seq_lens_sum = None
        child.input_ids = None
        child.out_cache_loc = None
        child.spec_info = copy(batch.spec_info)
        indices = torch.arange(start, end, device=batch.seq_lens.device)
        child.spec_info.filter_batch(indices, list(range(start, end)))
        # The worker receives a forward-only sampling snapshot. Its penalizer
        # is intentionally None; scheduler-side filter_batch cannot be used.
        child.sampling_info = copy(batch.sampling_info)
        for name in (
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seed",
            "rids_int",
            "bootstrap_room_ids_int",
        ):
            value = getattr(batch.sampling_info, name)
            setattr(
                child.sampling_info, name, value[rows] if value is not None else None
            )
        for name in (
            "grammars",
            "return_sampling_masks",
            "return_sampling_support_logprobs",
            "sampling_mask_top_ks",
        ):
            value = getattr(batch.sampling_info, name)
            setattr(
                child.sampling_info, name, value[rows] if value is not None else None
            )
        children.append(child)
    return children


def _draft(worker, batch):
    draft_worker = worker.draft_worker
    with (
        get_forward().scoped(**get_forward().snapshot()),
        patch_tensor_parallel_group(worker.microbatch_draft_group, owns_attention=True),
        forward_context(
            ForwardContext(attn_backend=draft_worker.draft_runner.attn_backend)
        ),
        speculative_moe_backend_context(),
        speculative_moe_a2a_backend_context(),
        spec_stage_span("microbatch.draft"),
    ):
        batch.spec_info = draft_worker.draft(batch)


def _draft_chunks(worker, batch):
    """Use the existing EAGLE proposal computation, yielding between steps."""
    from sglang.srt.speculative.eagle_worker_common import (
        build_eagle_verify_input,
        prepare_for_draft,
    )

    draft_worker = worker.draft_worker
    draft_input = batch.spec_info
    forward_batch, can_run_graph = prepare_for_draft(
        draft_input,
        draft_worker.req_to_token_pool,
        batch,
        draft_worker.cuda_graph_runner,
        draft_worker.draft_runner,
        draft_worker.topk,
        draft_worker.speculative_num_steps,
    )
    assert not can_run_graph
    if draft_worker.speculative_num_steps > 1:
        draft_worker.draft_attn_backend.init_forward_metadata(forward_batch)
        forward_batch.mark_forward_metadata_ready()
    parent_list, scores, tokens, probabilities = yield from (
        draft_worker.draft_forward_chunks(forward_batch)
    )
    batch.spec_info = build_eagle_verify_input(
        batch,
        draft_input,
        parent_list,
        scores,
        tokens,
        probabilities,
        target_worker=draft_worker.target_worker,
        topk=draft_worker.topk,
        num_steps=draft_worker.speculative_num_steps,
        num_draft_tokens=draft_worker.speculative_num_draft_tokens,
        tree_mask_mode=draft_worker.tree_mask_mode,
        device=draft_worker.device,
    )


def _advance_draft_chunk(worker, chunks):
    draft_worker = worker.draft_worker
    with (
        get_forward().scoped(**get_forward().snapshot()),
        patch_tensor_parallel_group(worker.microbatch_draft_group, owns_attention=True),
        forward_context(
            ForwardContext(attn_backend=draft_worker.draft_runner.attn_backend)
        ),
        speculative_moe_backend_context(),
        speculative_moe_a2a_backend_context(),
        spec_stage_span("microbatch.draft_chunk"),
    ):
        try:
            next(chunks)
            return False
        except StopIteration:
            return True


def _extend(worker, batch, result):
    draft_worker = worker.draft_worker
    with (
        get_forward().scoped(**get_forward().snapshot()),
        patch_tensor_parallel_group(worker.microbatch_draft_group, owns_attention=True),
        forward_context(
            ForwardContext(attn_backend=draft_worker.draft_runner.attn_backend)
        ),
        speculative_moe_backend_context(),
        speculative_moe_a2a_backend_context(),
        spec_stage_span("microbatch.draft_extend"),
    ):
        draft_worker._draft_extend_for_decode(batch, result)
        if check_cuda_graph_backend(Phase.DECODE, Backend.FULL):
            for name in (
                "topk_p",
                "topk_index",
                "hidden_states",
                "bonus_tokens",
                "draft_probs",
                "dsa_topk_indices",
            ):
                value = getattr(result.next_draft_input, name)
                if value is not None:
                    setattr(result.next_draft_input, name, value.clone())


def _verify(worker, batch, continuation=None, graph_continuation=None):
    with (
        forward_context(
            ForwardContext(
                attn_backend=worker._target_worker.model_runner.attn_backend,
                layer_continuation=continuation,
                graph_continuation=graph_continuation,
            )
        ),
        spec_stage_span("microbatch.verify"),
    ):
        result = worker.verify(batch)
        if check_cuda_graph_backend(Phase.DECODE, Backend.FULL):
            # The next target replay can overwrite graph-owned aux features.
            # Copy on the target stream before publishing its completion event.
            result.logits_output = copy(result.logits_output)
            result.logits_output.hidden_states = (
                result.logits_output.hidden_states.clone()
            )
        return result


def _merge(results, children):
    first, second = results
    first.next_draft_input.merge_batch(second.next_draft_input)
    # No logprobs, hidden-state returns, grammar, PP, or routed-expert output
    # are admitted by the prototype. Hidden features are only needed by extend.
    return GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=None),
        next_token_ids=torch.cat([r.next_token_ids for r in results]),
        accept_lens=torch.cat([r.accept_lens for r in results]),
        new_seq_lens=torch.cat([r.new_seq_lens for r in results]),
        next_draft_input=first.next_draft_input,
        can_run_cuda_graph=all(result.can_run_cuda_graph for result in results),
        speculative_num_draft_tokens=first.speculative_num_draft_tokens,
        extra_keep_alive_refs=[children, results],
    )


def ensure_microbatch_group(worker):
    if worker.microbatch_draft_group is None:
        target_group = get_parallel().tp_group
        # Concurrent collective streams need independent communicators. The
        # ranks and weights stay on exactly the same GPUs as the target.
        worker.microbatch_draft_group = init_model_parallel_group(
            [target_group.ranks],
            local_rank=target_group.local_rank,
            backend="nccl",
            use_custom_allreduce=False,
            use_mscclpp=False,
            use_torch_symm_mem_allreduce=False,
            group_name="spec_microbatch_draft",
        )
        worker.draft_worker.microbatch_draft_group = worker.microbatch_draft_group


@contextmanager
def private_microbatch_capture(draft_worker):
    from sglang.srt.model_executor.runner_utils.pool import disable_graph_pool_borrow

    disable_graph_pool_borrow(
        "concurrent speculative models require private graph pools"
    )
    resources = get_resources()
    streams = dict(
        resources.streams, cuda_graph_capture=get_stream("spec_microbatch_capture")
    )
    with (
        resources.override(graph_memory_pool=None, streams=streams),
        patch_tensor_parallel_group(
            draft_worker.microbatch_draft_group, owns_attention=True
        ),
    ):
        yield


def run_microbatch_step(worker, batch, on_publish=None):
    ensure_microbatch_group(worker)
    children = _split(batch)
    # draft() rebinds child.spec_info. Retain the caller-stream split tensors
    # separately so the allocator cannot recycle them while draft reads them.
    input_refs = [copy(child.spec_info) for child in children]
    first, second = children
    if get_spec().speculative_microbatch_mode == "serial":
        results = []
        for child in children:
            _draft(worker, child)
            result = _verify(worker, child)
            _extend(worker, child, result)
            results.append(result)
    else:
        target_stream = torch.cuda.current_stream()
        mode = get_spec().speculative_microbatch_mode
        draft_stream = (
            target_stream
            if mode == "fine_serial"
            else get_stream("spec_microbatch_draft")
        )
        draft_stream.wait_stream(target_stream)
        # Keep child inputs alive until both streams rejoin, including tensors
        # created on the target stream and subsequently consumed by draft.
        with torch.cuda.stream(draft_stream):
            _draft(worker, first)
            first_draft_done = draft_stream.record_event()
        target_stream.wait_event(first_draft_done)

        if mode in {"fine_serial", "fine_overlap"}:
            layers = worker._target_worker.model_runner.model.model.layers
            if check_cuda_graph_backend(Phase.DECODE, Backend.FULL):
                done = False
                second_draft_done = None

                def release_graph_draft(chunk_index):
                    nonlocal done, second_draft_done
                    if done or chunk_index != 1:
                        return
                    with torch.cuda.stream(draft_stream):
                        _draft(worker, second)
                        second_draft_done = draft_stream.record_event()
                    done = True

                first_result = _verify(
                    worker,
                    first,
                    continuation=lambda layer: (
                        release_graph_draft(1) if layer == len(layers) // 4 else None
                    ),
                    graph_continuation=release_graph_draft,
                )
                if not done:
                    raise RuntimeError(
                        "The target graph did not release the independent draft"
                    )
            else:
                chunks = _draft_chunks(worker, second)
                done = False
                second_draft_done = None
                num_chunks = max(1, worker.speculative_num_steps - 1)
                interval = max(1, len(layers) // (2 * num_chunks))
                boundaries = {
                    min(len(layers) - 1, interval * (i + 1) - 1)
                    for i in range(num_chunks)
                }

                def release_draft(layer_index):
                    nonlocal done, second_draft_done
                    if done or layer_index not in boundaries:
                        return
                    with torch.cuda.stream(draft_stream):
                        done = _advance_draft_chunk(worker, chunks)
                        if done:
                            second_draft_done = draft_stream.record_event()

                try:
                    first_result = _verify(worker, first, release_draft)
                finally:
                    chunks.close()
                if not done:
                    raise RuntimeError(
                        "The fine-grained draft continuation did not finish"
                    )
            first_verify_done = target_stream.record_event()
        else:
            first_result = _verify(worker, first)
            first_verify_done = target_stream.record_event()
            with torch.cuda.stream(draft_stream):
                _draft(worker, second)
                second_draft_done = draft_stream.record_event()
        with torch.cuda.stream(draft_stream):
            draft_stream.wait_event(first_verify_done)
            _extend(worker, first, first_result)
            first_extend_done = draft_stream.record_event()
        # Eager fine mode validates both overlap pairs. Graph mode keeps the
        # extend/verify fence: graph verification with concurrent eager FA3
        # draft extend has not passed token checks.
        if mode != "fine_overlap" or check_cuda_graph_backend(
            Phase.DECODE, Backend.FULL
        ):
            target_stream.wait_event(first_extend_done)
        target_stream.wait_event(second_draft_done)
        second_result = _verify(worker, second)
        second_verify_done = target_stream.record_event()
        with torch.cuda.stream(draft_stream):
            draft_stream.wait_event(second_verify_done)
            _extend(worker, second, second_result)
            extend_done = draft_stream.record_event()
        target_stream.wait_event(extend_done)
        results = [first_result, second_result]

    result = _merge(results, children)
    result.extra_keep_alive_refs.append(input_refs)
    if on_publish is not None:
        on_publish(result.new_seq_lens)
    return result
