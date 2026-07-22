from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from methods.resume_state import make_training_state, restore_training_state


class ResumeTrainingStateTest(unittest.TestCase):
    def test_round_trip_explicit_checkpoint_state(self) -> None:
        best = {"mAP": 7.5, "G-mIoU@3": 35.0, "joint": 1.1}
        checkpoint = {"training_state": make_training_state(best, 17)}
        restored, counter = restore_training_state(
            checkpoint, "unused.jsonl", primary_metric="joint"
        )
        self.assertEqual(restored, best)
        self.assertEqual(counter, 17)

    def test_reconstructs_legacy_best_and_patience_from_log(self) -> None:
        records = [
            {
                "epoch": 1,
                "selection": {"mAP": 1.0, "G-mIoU@3": 2.0, "joint": 1.0},
            },
            {
                "epoch": 2,
                "selection": {"mAP": 2.0, "G-mIoU@3": 1.0, "joint": 0.5},
            },
            {
                "epoch": 3,
                "selection": {"mAP": 1.5, "G-mIoU@3": 3.0, "joint": 0.7},
            },
            # A log can contain later records when resuming an older checkpoint;
            # they must not leak into reconstructed state.
            {
                "epoch": 4,
                "selection": {"mAP": 9.0, "G-mIoU@3": 9.0, "joint": 9.0},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "train_log.jsonl"
            log_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            best, counter = restore_training_state(
                {"epoch": 2}, log_path, primary_metric="joint"
            )
        self.assertEqual(best, {"mAP": 2.0, "G-mIoU@3": 3.0, "joint": 1.0})
        self.assertEqual(counter, 2)

    def test_legacy_resume_fails_closed_without_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jsonl"
            with self.assertRaisesRegex(RuntimeError, "JSONL log is unavailable"):
                restore_training_state(
                    {"epoch": 2}, missing, primary_metric="mAP"
                )


if __name__ == "__main__":
    unittest.main()
