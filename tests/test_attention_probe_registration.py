import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AttentionProbeRegistrationTest(unittest.TestCase):
    def test_enhancer_registers_attention_probe_nodes(self):
        tree = ast.parse((ROOT / "__init__.py").read_text())
        registered = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }

        self.assertIn("Qwen21ReferenceAttentionProbe", registered)
        self.assertIn("Qwen21ReferenceAttentionProbing", registered)
        self.assertIn("Qwen21ReferenceAttentionPreview", registered)
        self.assertIn("Qwen21ReferenceMaskPreview", registered)


if __name__ == "__main__":
    unittest.main()
