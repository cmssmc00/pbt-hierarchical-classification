import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "tools/check_public_tree.py"
spec = importlib.util.spec_from_file_location("public_tree_check", path)
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


class PublicationTests(unittest.TestCase):
    def test_reject_binary_and_identifiers(self):
        self.assertTrue(scanner.inspect_text("src/test.py", b"binary\x00payload"))
        identifier = "99" + "_" + "99999"
        self.assertTrue(scanner.inspect_text("src/test.py", identifier.encode()))

    def test_reject_credentials_and_machine_paths(self):
        key = "ghp" + "_" + "A" * 36
        self.assertTrue(scanner.inspect_text("configs/test.json", key.encode()))
        private_path = "/" + "Users/" + "example/private"
        self.assertTrue(scanner.inspect_text("docs/test.md", private_path.encode()))

    def test_allow_fictional_template(self):
        template = path.parents[1] / "examples/manifest.example.json"
        self.assertEqual(scanner.inspect_text("examples/manifest.example.json", template.read_bytes()), [])
