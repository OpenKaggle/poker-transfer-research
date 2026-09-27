"""Pure synthetic/static tests for the three EVP-DRAFT-04.2 repairs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import poker_attack.evidence_prereg_v4_2 as runner
from poker_attack.evidence_prereg_v4_2 import (
    CONSUMPTION_CLAIM_FILENAME,
    CONSUMPTION_FILENAME,
    EXCEPTION_STOP_FILENAME,
    ExecutionStateV4_2,
    _atomic_json,
    _canonical_result_root_v4_2,
    _consume_authorization_v4_2,
    _write_exception_terminal_v4_2,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_canonical_root(project_root: Path) -> Path:
    work = project_root / "poker/work"
    work.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "poker/work/EVP_v4_2_result_root_binding.json",
        work / "EVP_v4_2_result_root_binding.json",
    )
    return work.resolve(strict=True)


class CanonicalResultRootV4_2Test(unittest.TestCase):
    def test_exact_absolute_root_or_omission_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            work = _install_canonical_root(project_root)
            with patch.object(runner, "_project_root", return_value=project_root):
                self.assertEqual(_canonical_result_root_v4_2(None), work)
                self.assertEqual(_canonical_result_root_v4_2(work), work)

    def test_alternate_relative_and_normalized_equivalent_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            work = _install_canonical_root(project_root)
            alternate = project_root / "alternate"
            alternate.mkdir()
            with patch.object(runner, "_project_root", return_value=project_root):
                for candidate in (alternate, Path("poker/work"), work / "../work"):
                    with self.subTest(candidate=candidate):
                        with self.assertRaisesRegex(PermissionError, "exact canonical"):
                            _canonical_result_root_v4_2(candidate)
            self.assertFalse((alternate / CONSUMPTION_CLAIM_FILENAME).exists())
            self.assertFalse((work / CONSUMPTION_CLAIM_FILENAME).exists())

    def test_symlinked_canonical_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary) / "project"
            (project_root / "poker").mkdir(parents=True)
            external = Path(temporary) / "external"
            external.mkdir()
            shutil.copyfile(
                ROOT / "poker/work/EVP_v4_2_result_root_binding.json",
                external / "EVP_v4_2_result_root_binding.json",
            )
            (project_root / "poker/work").symlink_to(external, target_is_directory=True)
            with patch.object(runner, "_project_root", return_value=project_root):
                with self.assertRaisesRegex(PermissionError, "symlinked"):
                    _canonical_result_root_v4_2(None)


class DurableConsumptionV4_2Test(unittest.TestCase):
    def test_complete_claim_and_receipt_are_append_only_and_single_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            work = _install_canonical_root(project_root)
            authorization = Path(temporary) / "synthetic_authorization.json"
            authorization.write_text("{}\n", encoding="utf-8")
            with patch.object(runner, "_project_root", return_value=project_root):
                receipt = _consume_authorization_v4_2(
                    authorization, work, {"synthetic": True}, {"synthetic": True}
                )
            claim = work / CONSUMPTION_CLAIM_FILENAME
            consumed = work / CONSUMPTION_FILENAME
            self.assertTrue(claim.is_file())
            self.assertTrue(consumed.is_file())
            self.assertEqual(receipt["claim_sha256"], _sha256(claim))
            with patch.object(runner, "_project_root", return_value=project_root):
                with self.assertRaisesRegex(PermissionError, "already consumed"):
                    _consume_authorization_v4_2(
                        authorization, work, {"synthetic": True}, {"synthetic": True}
                    )

    def test_same_authorization_cannot_replay_through_an_alternate_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            work = _install_canonical_root(project_root)
            alternate = project_root / "alternate"
            alternate.mkdir()
            authorization = project_root / "synthetic_authorization.json"
            authorization.write_text("{}\n", encoding="utf-8")
            with patch.object(runner, "_project_root", return_value=project_root):
                _consume_authorization_v4_2(
                    authorization, work, {"synthetic": True}, {"synthetic": True}
                )
                with self.assertRaisesRegex(PermissionError, "exact canonical"):
                    _consume_authorization_v4_2(
                        authorization,
                        alternate,
                        {"synthetic": True},
                        {"synthetic": True},
                    )
            self.assertTrue((work / CONSUMPTION_CLAIM_FILENAME).exists())
            self.assertFalse((alternate / CONSUMPTION_CLAIM_FILENAME).exists())

    def test_partial_claim_write_remains_consumed_and_gets_terminal_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            work = _install_canonical_root(project_root)
            authorization = Path(temporary) / "synthetic_authorization.json"
            authorization.write_text("{}\n", encoding="utf-8")

            def partial_write(descriptor: int, payload: bytes) -> None:
                os.write(descriptor, payload[:7])
                raise OSError("synthetic partial claim write")

            with (
                patch.object(runner, "_project_root", return_value=project_root),
                patch.object(runner, "_write_complete_claim_v4_2", partial_write),
            ):
                with self.assertRaisesRegex(OSError, "partial claim"):
                    _consume_authorization_v4_2(
                        authorization,
                        work,
                        {"synthetic": True},
                        {"synthetic": True},
                    )
            claim = work / CONSUMPTION_CLAIM_FILENAME
            self.assertTrue(claim.is_file())
            self.assertFalse((work / CONSUMPTION_FILENAME).exists())
            _write_exception_terminal_v4_2(
                work,
                ExecutionStateV4_2(),
                OSError("synthetic partial claim write"),
            )
            terminal = json.loads(
                (work / EXCEPTION_STOP_FILENAME).read_text(encoding="utf-8")
            )
            self.assertIs(terminal["binding"]["consumption_claim_present"], True)
            self.assertIs(terminal["binding"]["consumption_receipt_present"], False)
            self.assertEqual(
                terminal["binding"]["consumption_claim_sha256"], _sha256(claim)
            )
            with patch.object(runner, "_project_root", return_value=project_root):
                with self.assertRaisesRegex(PermissionError, "already consumed"):
                    _consume_authorization_v4_2(
                        authorization, work, {"synthetic": True}, {"synthetic": True}
                    )

    def test_malformed_consumption_receipt_does_not_block_terminal_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = _install_canonical_root(Path(temporary))
            (work / CONSUMPTION_CLAIM_FILENAME).write_bytes(b"")
            (work / CONSUMPTION_FILENAME).write_text("{", encoding="utf-8")
            _write_exception_terminal_v4_2(
                work, ExecutionStateV4_2(), ValueError("synthetic malformed receipt")
            )
            terminal = json.loads(
                (work / EXCEPTION_STOP_FILENAME).read_text(encoding="utf-8")
            )
            self.assertIs(terminal["binding"]["consumption_claim_present"], True)
            self.assertIs(terminal["binding"]["consumption_receipt_present"], True)

    def test_existing_result_receipt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text("original", encoding="utf-8")
            with self.assertRaises(PermissionError):
                _atomic_json(path, {"changed": True})
            self.assertEqual(path.read_text(encoding="utf-8"), "original")


class ExactStage2ControlsV4_2Test(unittest.TestCase):
    def test_exact_stage_2_controls_are_explicit_booleans(self):
        controls = ExecutionStateV4_2().terminal_controls()
        for name in (
            "stage_2_development_data_access",
            "stage_2_model_fit",
            "stage_2_aggregate_performance_viewed",
        ):
            self.assertIn(name, controls)
            self.assertIs(type(controls[name]), bool)
            self.assertIs(controls[name], False)

    def test_stage_2_control_transitions_follow_the_method_events(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        stage_2 = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_stage_2_attribution_v4"
        )
        segment = ast.get_source_segment(source, stage_2)
        self.assertLess(
            segment.index("execution.stage_2_development_data_access = True"),
            segment.index("_reload_stage_1_states_v4_2"),
        )
        self.assertLess(
            segment.index("_fit_outer_l0_from_state_v4(state)"),
            segment.index("execution.stage_2_model_fit = True"),
        )
        self.assertLess(
            segment.index("sensitivity = sensitivity_tree_v4"),
            segment.index("execution.stage_2_aggregate_performance_viewed = True"),
        )

    def test_canonical_root_and_runtime_guards_precede_any_data_loader(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_authorized_validation_v4_2"
        )
        segment = ast.get_source_segment(source, run)
        self.assertLess(
            segment.index("_canonical_result_root_v4_2"),
            segment.index("_pre_data_runtime_and_dependency_guard_v4_2"),
        )
        self.assertLess(
            segment.index("_pre_data_runtime_and_dependency_guard_v4_2"),
            segment.index("_validate_result_execution_authorization_v4_2"),
        )
        self.assertLess(
            segment.index("_validate_result_execution_authorization_v4_2"),
            segment.index("_load_full_inputs_v4"),
        )


if __name__ == "__main__":
    unittest.main()
