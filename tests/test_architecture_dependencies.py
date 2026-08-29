import ast
import unittest
from pathlib import Path


class ArchitectureDependencyTest(unittest.TestCase):
    def test_domain_and_application_do_not_import_infrastructure_or_api(self):
        package = Path(__file__).resolve().parents[1] / "src" / "prievo_agent"
        violations = []
        for layer in ("domain", "application"):
            for path in (package / layer).glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith((
                            "prievo_agent.infrastructure", "prievo_agent.api"
                        )):
                            violations.append("{} -> {}".format(path.name, node.module))
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            if name.name.startswith((
                                "prievo_agent.infrastructure", "prievo_agent.api"
                            )):
                                violations.append("{} -> {}".format(path.name, name.name))
        self.assertEqual([], violations)

    def test_core_has_no_runtime_framework_or_adapter_dependencies(self):
        package = Path(__file__).resolve().parents[1] / "src" / "prievo_agent"
        forbidden_prefixes = (
            "prievo_agent.api",
            "prievo_agent.application",
            "prievo_agent.infrastructure",
            "prievo_agent.runtime",
            "fastapi",
            "sqlalchemy",
            "mcp",
            "sse",
        )
        violations = []
        for path in (package / "core").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                elif isinstance(node, ast.Import):
                    names.extend(item.name for item in node.names)
                for name in names:
                    if name.startswith(forbidden_prefixes):
                        violations.append("{} -> {}".format(path.name, name))
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
