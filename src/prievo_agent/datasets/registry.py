import csv
import hashlib
import logging
from pathlib import Path


logger = logging.getLogger("prievo.datasets")


class DatasetError(ValueError):
    pass


class DatasetInfo:
    def __init__(self, dataset_id, filename, display_name, path, columns,
                 independent_columns, dependent_columns, fidelity_columns,
                 row_count, digest):
        self.id = dataset_id
        self.filename = filename
        self.display_name = display_name
        self.path = path
        self.columns = columns
        self.independent_columns = independent_columns
        self.dependent_columns = dependent_columns
        self.fidelity_columns = fidelity_columns
        self.row_count = row_count
        self.digest = digest

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "display_name": self.display_name,
            "row_count": self.row_count,
            "independent_columns": list(self.independent_columns),
            "dependent_columns": list(self.dependent_columns),
            "fidelity_columns": list(self.fidelity_columns),
            "digest": self.digest,
        }


class LoadedDataset:
    def __init__(self, info, rows, configurations, objectives):
        self.info = info
        self.rows = rows
        self.configurations = configurations
        self.objectives = objectives
        self.lookup = {}
        for index, config in enumerate(configurations):
            self.lookup[tuple(config)] = objectives[index]


class DatasetRegistry:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def list_datasets(self):
        datasets = []
        if not self.root.exists():
            logger.warning("Dataset 目录不存在：%s", self.root)
            return datasets
        for path in sorted(self.root.glob("*.csv")):
            try:
                datasets.append(inspect_dataset(path))
            except DatasetError as exc:
                logger.warning("忽略无效 Dataset %s：%s", path.name, exc)
        return datasets

    def get(self, dataset_id):
        # 只接受扫描结果中的 ID，调用者不能把任意文件路径传进来。
        for dataset in self.list_datasets():
            if dataset.id == dataset_id:
                return dataset
        raise DatasetError("Dataset 不存在或格式无效：{}".format(dataset_id))

    def load(self, dataset_id):
        return load_dataset(self.get(dataset_id))


def inspect_dataset(path):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            rows = []
            for row in reader:
                if any(cell.strip() for cell in row):
                    rows.append(row)
    except (OSError, UnicodeError, StopIteration) as exc:
        raise DatasetError("CSV 无法读取") from exc
    if not columns or len(set(columns)) != len(columns):
        raise DatasetError("列名为空或重复")
    fidelity = [name for name in columns if name.startswith("F$")]
    dependent = [name for name in columns if "$<" in name]
    independent = [name for name in columns if "$<" not in name and name not in fidelity]
    if not independent or not dependent:
        raise DatasetError("必须同时包含配置列和 $< 目标列")
    if len(rows) < 10:
        raise DatasetError("有效数据行少于 10")
    column_index = {name: index for index, name in enumerate(columns)}
    for row in rows[:50]:
        if len(row) != len(columns):
            raise DatasetError("数据行列数与表头不一致")
        try:
            for name in dependent:
                float(row[column_index[name]])
        except ValueError as exc:
            raise DatasetError("$< 目标值必须是数值") from exc
    return DatasetInfo(path.stem, path.name, path.stem, path, columns,
                       independent, dependent, fidelity, len(rows), digest)


def load_dataset(info):
    rows = []
    configurations = []
    objectives = []
    with info.path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for name, value in raw.items():
                number = _parse_value(value)
                # 与 PriEvO ReadDataset 保持一致：+$< 目标读入后取负。
                if name.startswith("+$<"):
                    number = -float(number)
                row[name] = number
            rows.append(row)
            configurations.append([row[name] for name in info.independent_columns])
            objectives.append(float(row[info.dependent_columns[-1]]))
    logger.info("已加载目标 Dataset：%s，共 %s 行", info.id, len(rows))
    return LoadedDataset(info, rows, configurations, objectives)


def _parse_value(value):
    try:
        return float(value)
    except ValueError:
        return value
