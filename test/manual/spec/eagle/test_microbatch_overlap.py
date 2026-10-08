"""CPU contract checks for the experimental forward-snapshot split and join."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_forward
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.microbatch_overlap import (
    _merge,
    _split,
    can_microbatch_overlap,
    validate_microbatch_config,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase


class TestMicrobatchSnapshot(CustomTestCase):
    def test_concurrent_graph_runners_do_not_alias_vocab_logits(self):
        from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput

        runner = SimpleNamespace(device="cpu", max_shared_logits_buffer_rows=lambda: 32)
        with get_context().override_server_args(
            model_path="dummy",
            speculative_microbatch_mode="fine_overlap",
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="full", bs=[32]),
                prefill=PhaseConfig(backend="disabled"),
            ),
        ):
            target = GraphSharedOutput.create_for_model_runner(runner)
            draft = GraphSharedOutput.create_for_model_runner(runner)
            target_logits = target.get_logits_buffer(16, rows=8)
            draft_logits = draft.get_logits_buffer(16, rows=8)
            target_logits.fill_(1)
            draft_logits.fill_(2)
            torch.testing.assert_close(target_logits, torch.ones(8, 16))
            self.assertNotEqual(target_logits.data_ptr(), draft_logits.data_ptr())

    def test_concurrent_runners_do_not_alias_pooled_positions(self):
        from sglang.srt.model_executor.input_buffers import share_input_buffer

        with get_context().override_server_args(
            model_path="dummy", speculative_microbatch_mode="fine_overlap"
        ):
            target = share_input_buffer("positions", torch.arange(32))
            draft = share_input_buffer("positions", torch.zeros(32, dtype=torch.int64))
            draft[:8].fill_(29)
            torch.testing.assert_close(target, torch.arange(32))
            self.assertNotEqual(target.data_ptr(), draft.data_ptr())

    def test_suspended_forward_flags_restore_nested_updates(self):
        flags = get_forward()
        with flags.scoped(is_extend_in_batch=True, attn_input_scattered=True):
            suspended = flags.snapshot()
            with flags.scoped(**suspended):
                # ForwardBatch construction has legacy sticky flag writers.
                flags.set("is_extend_in_batch", False)
                flags.set("attn_input_scattered", False)
                flags.set("moe_output_buffer", torch.ones(1))
            self.assertEqual(flags.snapshot(), suspended)

    def test_tree_verify_masks_remain_private_until_target_consumes_them(self):
        from sglang.srt.layers.attention.verify_mask import VerifyMask
        from sglang.srt.speculative.eagle_utils import TreeMaskMode
        from sglang.srt.speculative.eagle_worker_common import build_eagle_verify_input

        size, num_draft_tokens = 2, 6
        reusable = torch.zeros(4 * num_draft_tokens**2, dtype=torch.bool)
        backend = SimpleNamespace(
            verify_mask=VerifyMask(
                buffer=reusable, mode=TreeMaskMode.QLEN_ONLY, max_bs=4
            )
        )
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=backend)
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([10, 20]),
            seq_lens_sum=None,
        )

        # Stub only the CUDA tree-builder boundary: emulate its in-place write
        # and returned view, keeping the real publication/allocation logic.
        def build_tree(
            bonus,
            parents,
            scores,
            tokens,
            seq_lens,
            seq_lens_sum,
            topk,
            steps,
            token_count,
            mask_mode,
            mask_buf,
            **kwargs,
        ):
            mask = mask_buf[: size * token_count**2]
            mask.fill_(bool(bonus[0].item()))
            positions = torch.arange(size * token_count)
            retrieval = torch.zeros((size, token_count), dtype=torch.int64)
            return (
                mask,
                positions,
                retrieval,
                retrieval.clone(),
                retrieval.clone(),
                tokens,
            )

        def proposal(marker, topk):
            return build_eagle_verify_input(
                batch,
                EagleDraftInput(bonus_tokens=torch.full((size,), marker)),
                torch.zeros((size, 2), dtype=torch.int64),
                torch.zeros((size, num_draft_tokens - 1), dtype=torch.int64),
                torch.zeros((size, num_draft_tokens), dtype=torch.int64),
                None,
                target_worker=target_worker,
                topk=topk,
                num_steps=5,
                num_draft_tokens=num_draft_tokens,
                tree_mask_mode=TreeMaskMode.QLEN_ONLY,
                device="cpu",
            )

        with patch(
            "sglang.srt.speculative.eagle_worker_common.build_tree_kernel_efficient",
            side_effect=build_tree,
        ):
            for topk in (2, 3, 4):
                with self.subTest(topk=topk):
                    reusable.zero_()
                    with get_context().override_server_args(
                        model_path="dummy", speculative_microbatch_mode="fine_overlap"
                    ):
                        first = proposal(1, topk)
                        second = proposal(0, topk)
                    self.assertNotEqual(
                        first.custom_mask.data_ptr(), second.custom_mask.data_ptr()
                    )
                    self.assertNotEqual(
                        first.custom_mask.data_ptr(), reusable.data_ptr()
                    )
                    self.assertNotEqual(
                        second.custom_mask.data_ptr(), reusable.data_ptr()
                    )
                    second.custom_mask.fill_(True)
                    torch.testing.assert_close(
                        first.custom_mask, torch.ones_like(first.custom_mask)
                    )
                    torch.testing.assert_close(reusable, torch.zeros_like(reusable))
                    second.custom_mask.zero_()
                    torch.testing.assert_close(
                        first.custom_mask, torch.ones_like(first.custom_mask)
                    )

            with get_context().override_server_args(
                model_path="dummy", speculative_microbatch_mode="off"
            ):
                first = proposal(1, 2)
                self.assertEqual(first.custom_mask.data_ptr(), reusable.data_ptr())
                second = proposal(0, 2)
                self.assertEqual(second.custom_mask.data_ptr(), reusable.data_ptr())
                torch.testing.assert_close(
                    first.custom_mask, torch.zeros_like(first.custom_mask)
                )

    def test_tree_rows_survive_asymmetric_split_and_merge(self):
        """Splits select requests, preserving every tree candidate and state row."""
        for size, percent, expected_sizes in (
            (2, 1, [1, 1]),
            (5, 25, [1, 4]),
            (5, 75, [3, 2]),
            (5, 99, [4, 1]),
            (8, 25, [2, 6]),
        ):
            for topk in (2, 3, 4):
                with self.subTest(size=size, percent=percent, topk=topk):
                    rows = torch.arange(size)
                    candidates = torch.arange(size * topk).reshape(size, topk)
                    spec = EagleDraftInput(
                        topk_p=candidates.float() / (size * topk),
                        topk_index=candidates,
                        hidden_states=torch.arange(size * 3).float().reshape(size, 3),
                        bonus_tokens=rows + 100,
                        draft_probs=torch.arange(size * 7).float().reshape(size, 7),
                        dsa_topk_indices=candidates + 200,
                    )
                    sampling = SimpleNamespace(
                        temperatures=rows.float().view(-1, 1),
                        top_ps=rows.float(),
                        top_ks=rows,
                        min_ps=rows.float(),
                        sampling_seed=rows + 20,
                        rids_int=rows + 30,
                        bootstrap_room_ids_int=rows + 40,
                        grammars=None,
                        return_sampling_masks=[False] * size,
                        return_sampling_support_logprobs=None,
                        sampling_mask_top_ks=list(range(size)),
                    )
                    batch = SimpleNamespace(
                        reqs=[SimpleNamespace(rid=str(i)) for i in range(size)],
                        req_pool_indices=rows,
                        req_pool_indices_cpu=rows,
                        seq_lens=rows + 10,
                        orig_seq_lens=rows + 10,
                        seq_lens_cpu=rows + 10,
                        spec_info=spec,
                        sampling_info=sampling,
                    )
                    children = _split(batch, percent)
                    self.assertEqual([len(c.reqs) for c in children], expected_sizes)
                    self.assertEqual(
                        [r.rid for c in children for r in c.reqs],
                        [str(i) for i in range(size)],
                    )
                    fields = (
                        "topk_p",
                        "topk_index",
                        "hidden_states",
                        "bonus_tokens",
                        "draft_probs",
                        "dsa_topk_indices",
                    )
                    for name in fields:
                        torch.testing.assert_close(
                            torch.cat([getattr(c.spec_info, name) for c in children]),
                            getattr(spec, name),
                        )
                    torch.testing.assert_close(
                        torch.cat([c.sampling_info.sampling_seed for c in children]),
                        sampling.sampling_seed,
                    )
                    self.assertEqual(
                        [
                            i
                            for c in children
                            for i in c.sampling_info.sampling_mask_top_ks
                        ],
                        list(range(size)),
                    )
                    children[0].spec_info.merge_batch(children[1].spec_info)
                    for name in fields:
                        torch.testing.assert_close(
                            getattr(children[0].spec_info, name), getattr(spec, name)
                        )
                    # Filtering and merging rebind child fields, never parent state.
                    self.assertEqual(spec.topk_index.shape, (size, topk))

    def test_invalid_schedule_configuration_is_rejected(self):
        invalid = (
            ({"speculative_microbatch_split_percent": 0}, "split percent"),
            ({"speculative_microbatch_split_percent": 100}, "split percent"),
            ({"speculative_microbatch_graph_chunks": 0}, "chunk count"),
            ({"speculative_microbatch_release_chunk": -2}, "release chunk"),
            (
                {
                    "speculative_microbatch_graph_chunks": 2,
                    "speculative_microbatch_release_chunk": 2,
                },
                "release chunk",
            ),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                with get_context().override_server_args(
                    model_path="dummy",
                    speculative_microbatch_mode="fine_overlap",
                    **overrides,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_microbatch_config(SimpleNamespace())

    def test_odd_batch_round_trip_without_scheduler_penalizer(self):
        """A forward-only snapshot must split without filtering a missing penalizer.

        Uneven groups must preserve request/sample/state row alignment, and
        splitting/rebinding a child must leave the parent's next-step state intact.
        """
        size = 5
        rows = torch.arange(size)
        sampling = SamplingBatchInfo(
            temperatures=rows.float().view(-1, 1),
            top_ps=rows.float(),
            top_ks=rows,
            min_ps=rows.float(),
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=16,
            device="cpu",
            penalizer_orchestrator=None,
            return_sampling_masks=[False] * size,
        )
        batch = ScheduleBatch(
            reqs=[
                Req(str(i), "", [1], SamplingParams(temperature=0)) for i in range(size)
            ],
            req_pool_indices=rows,
            req_pool_indices_cpu=rows,
            seq_lens=rows + 10,
            orig_seq_lens=rows + 10,
            seq_lens_cpu=rows + 10,
            sampling_info=sampling,
            device="cpu",
            forward_mode=ForwardMode.DECODE,
            spec_info=EagleDraftInput(
                topk_p=rows.float().view(-1, 1),
                topk_index=rows.view(-1, 1),
                hidden_states=rows.float().view(-1, 1),
                bonus_tokens=rows,
            ),
        )
        # Default all-false capture flags once silently disabled the prototype.
        # Use the real config bags and worker type to guard that eligibility bug.
        worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        worker.speculative_algorithm = SpeculativeAlgorithm.from_string("EAGLE3")
        worker.device = "cuda"
        worker.topk = 1
        worker.speculative_num_steps = 3
        with get_context().override_server_args(
            model_path="dummy",
            speculative_microbatch_mode="overlap",
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="disabled"),
                prefill=PhaseConfig(backend="disabled"),
            ),
            disable_custom_all_reduce=True,
        ):
            for topk in (1, 2, 3, 4):
                with self.subTest(topk=topk):
                    worker.topk = topk
                    self.assertTrue(can_microbatch_overlap(worker, batch))
            sampling.return_sampling_masks[0] = True
            self.assertFalse(can_microbatch_overlap(worker, batch))
            sampling.return_sampling_masks[0] = False
        children = _split(batch)
        self.assertEqual([len(c.reqs) for c in children], [2, 3])
        self.assertEqual(
            [req.rid for c in children for req in c.reqs], [str(i) for i in range(size)]
        )
        torch.testing.assert_close(
            torch.cat([c.sampling_info.top_ks for c in children]), rows
        )
        torch.testing.assert_close(
            torch.cat([c.spec_info.hidden_states for c in children]),
            batch.spec_info.hidden_states,
        )

        results = []
        for child in children:
            child_rows = child.req_pool_indices
            results.append(
                GenerationBatchResult(
                    next_token_ids=torch.stack(
                        [child_rows, child_rows + 10], dim=1
                    ).flatten(),
                    accept_lens=torch.ones_like(child_rows) * 2,
                    new_seq_lens=child.seq_lens + 2,
                    next_draft_input=child.spec_info,
                    speculative_num_draft_tokens=2,
                )
            )
        merged = _merge(results, children)
        torch.testing.assert_close(
            merged.next_token_ids, torch.stack([rows, rows + 10], dim=1).flatten()
        )
        torch.testing.assert_close(merged.next_draft_input.bonus_tokens, rows)
        torch.testing.assert_close(merged.new_seq_lens, batch.seq_lens + 2)
        self.assertEqual(len(batch.spec_info.bonus_tokens), size)
        self.assertEqual(len(batch.sampling_info.top_ks), size)


if __name__ == "__main__":
    unittest.main()
