import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/runtime-check"


@unittest.skipUnless(sys.platform == "darwin" and BINARY.is_file(), "build runtime-check on macOS first")
class PackageRejectionTests(unittest.TestCase):
    def reject(self, index, data, expected):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "index.json").write_text(json.dumps(index))
            (path / "weights.bin").write_bytes(data)
            run = subprocess.run([str(BINARY), str(path), str(ROOT / "native"), "missing", "1", str(path / "out")],
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 2)
            self.assertIn(expected, run.stderr)

    def index(self):
        return {"format": "qwen-asr-metal-v1", "precision": "int8", "weights_bytes": 65536,
                "weights_sha256": "0"*64, "tensors": {}}

    def test_corrupted_weights_rejected_before_execution(self):
        self.reject(self.index(), b"\0"*65536, "weights checksum mismatch")

    def test_truncated_weights_rejected(self):
        self.reject(self.index(), b"\0"*32, "invalid weights file size/alignment")

    def test_tensor_outside_mapping_rejected(self):
        data = b"\0"*65536
        index = self.index()
        index["weights_sha256"] = hashlib.sha256(data).hexdigest()
        index["tensors"] = {"bad": {"dtype": "f32", "shape": [1000], "offset": 65280, "bytes": 4000}}
        self.reject(index, data, "tensor outside package bounds")


if __name__ == "__main__":
    unittest.main()
