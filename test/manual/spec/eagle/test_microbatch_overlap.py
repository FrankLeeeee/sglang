"""CPU contract checks for the experimental forward-snapshot split and join."""

import unittest
from types import SimpleNamespace

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
