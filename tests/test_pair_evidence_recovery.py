"""Dependency-free tests for non-overwriting pair evidence recovery."""
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pipeline_v3 import recover_pair_evidence as recovery


class RecoveryTests(unittest.TestCase):
    def test_restore_only_allowlisted_missing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'project'
            archive = root / 'reports/experiment_archive/20260914T053818Z/evidence.tar.gz'
            archive.parent.mkdir(parents=True)
            content = {
                'data/specialists/en_ru/exp2/train.jsonl': b'{}\n',
                'reports/experiments/zh_uz_test/existing.json': b'old',
                'configs/review/zh_uz_manual_review_decisions_v1.json': b'{}',
                'reports/diagnostics/fourlang/private.json': b'{}',
            }
            with tarfile.open(archive, 'w:gz') as tf:
                for name, raw in content.items():
                    item = tarfile.TarInfo('evidence/' + name)
                    item.size = len(raw)
                    tf.addfile(item, io.BytesIO(raw))
            existing = root / 'reports/experiments/zh_uz_test/existing.json'
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b'new')
            reconciliation = base / 'reconciliation.json'
            reconciliation.write_text(json.dumps({'archived_files_absent_at_original_path':
                                                  [{'path': p} for p in content]}))
            out = base / 'recovered'
            argv = ['recover', '--root', str(root), '--reconciliation', str(reconciliation), '--output', str(out)]
            with patch.object(sys, 'argv', argv):
                recovery.main()
            self.assertEqual((root / 'data/specialists/en_ru/exp2/train.jsonl').read_bytes(), b'{}\n')
            self.assertEqual(existing.read_bytes(), b'new')
            self.assertFalse((root / 'configs/review/zh_uz_manual_review_decisions_v1.json').exists())
            self.assertFalse((out / 'evidence/reports/diagnostics/fourlang/private.json').exists())
            states = {r['path']: r['status'] for r in json.loads((out / 'recovery_log.json').read_text())}
            self.assertEqual(states['reports/experiments/zh_uz_test/existing.json'], 'existing_preserved')
            with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):
                recovery.main()

    def test_refuse_recovery_output_inside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['recover', '--root', str(root), '--reconciliation', str(root/'unused'), '--output', str(root/'out')]
            with patch.object(sys, 'argv', argv), self.assertRaises(ValueError):
                recovery.main()


if __name__ == '__main__':
    unittest.main()
