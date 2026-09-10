import ast
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "prievo_agent"


def internal_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("prievo_agent"):
                yield node.module
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.name.startswith("prievo_agent"):
                    yield name.name


def dependency_violations(paths, forbidden_prefixes):
    violations = []
    for path in paths:
        for module in internal_imports(path):
            if module.startswith(forbidden_prefixes):
                violations.append(
                    "{} -> {}".format(path.relative_to(PACKAGE), module)
                )
    return violations


class ArchitectureDependencyTest(unittest.TestCase):
    def test_domain_is_dependency_free(self):
        paths = (PACKAGE / "domain").rglob("*.py")
        self.assertEqual([], dependency_violations(paths, ("prievo_agent.",)))

    def test_application_does_not_depend_on_delivery_or_adapters(self):
        paths = (PACKAGE / "application").rglob("*.py")
        forbidden = (
            "prievo_agent.api",
            "prievo_agent.cli",
            "prievo_agent.infrastructure",
        )
        self.assertEqual([], dependency_violations(paths, forbidden))

    def test_feature_logic_does_not_depend_on_delivery_or_adapters(self):
        forbidden = (
            "prievo_agent.api",
            "prievo_agent.cli",
            "prievo_agent.infrastructure",
            "prievo_agent.runtime",
        )
        paths = []
        for package in ("agents", "evaluation", "knowledge", "security"):
            paths.extend((PACKAGE / package).rglob("*.py"))
        self.assertEqual([], dependency_violations(paths, forbidden))

    def test_pure_evolution_modules_stay_framework_independent(self):
        forbidden = (
            "prievo_agent.api",
            "prievo_agent.application",
            "prievo_agent.cli",
            "prievo_agent.infrastructure",
            "prievo_agent.runtime",
        )
        paths = [
            path
            for path in (PACKAGE / "evolution").glob("*.py")
            if path.name != "engine.py"
        ]
        self.assertEqual([], dependency_violations(paths, forbidden))


if __name__ == "__main__":
    unittest.main()
