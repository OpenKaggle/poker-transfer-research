"""Source-only invariants for the frozen EVP-DRAFT-04.2 mechanics."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest

import poker_attack.evidence_prereg_v4_2_core as core


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenCoreV4_2Test(unittest.TestCase):
    def test_delta_and_result_root_marker_hashes_are_exact(self):
        self.assertEqual(
            _sha256(ROOT / "poker/EVIDENCE_ONLY_VALIDATION_PREREG_V4_2_DELTA.md"),
            core.V4_2_DELTA_SHA256,
        )
        self.assertEqual(
            _sha256(ROOT / "poker/work/EVP_v4_2_result_root_binding.json"),
            core.V4_2_RESULT_ROOT_BINDING_SHA256,
        )

    def test_core_imports_no_rejected_executor(self):
        path = Path(core.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("poker_attack.evidence_prereg", imported)
        self.assertNotIn("poker_attack.evidence_prereg_core", imported)

    def test_all_estimator_random_states_are_explicit_in_source(self):
        source = Path(core.__file__).read_text(encoding="utf-8")
        self.assertIn("random_state=INNER_SPLIT_BASE_SEED + outer_fold", source)
        self.assertIn("random_state=seed", source)
        self.assertIn("threadpool_limits(limits=1)", source)

    def test_core_contains_no_external_or_evaluation_io(self):
        source = Path(core.__file__).read_text(encoding="utf-8")
        for token in (
            "evaluation_pairs.csv",
            "sample_submission.csv",
            "submission.csv",
            "requests.",
            "urllib.",
            "subprocess.",
            "kaggle competitions",
        ):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
