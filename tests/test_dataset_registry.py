import tempfile
import unittest
from pathlib import Path

from prievo_agent.datasets.registry import DatasetError, DatasetRegistry


class DatasetRegistryTest(unittest.TestCase):
    def test_bundled_datasets_are_scanned_and_loaded(self):
        root = Path(__file__).resolve().parents[1]
        registry = DatasetRegistry(root / "resources" / "datasets")
        datasets = registry.list_datasets()
        self.assertEqual(
            {"brotli", "sqlite", "xgboost-Covtype", "xgboost-PimaIndiansDiabetes"},
            {item.id for item in datasets},
        )
        loaded = registry.load("xgboost-Covtype")
        self.assertEqual("xgboost-Covtype", loaded.info.id)
        self.assertEqual(11, len(loaded.info.independent_columns))
        self.assertGreater(len(loaded.rows), 100)
        self.assertEqual(len(loaded.rows), len(loaded.lookup))

    def test_invalid_csv_is_ignored_and_path_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.csv").write_text("name,value\na,b\n", encoding="utf-8")
            registry = DatasetRegistry(root)
            self.assertEqual([], registry.list_datasets())
            with self.assertRaises(DatasetError):
                registry.get("../resources/datasets/xgboost-Covtype")


if __name__ == "__main__":
    unittest.main()
