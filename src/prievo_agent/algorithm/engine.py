class AlgorithmEngine:
    """算法后端的简单替换边界，不引入 Protocol/Generic。"""

    def prepare(self, run_id):
        raise NotImplementedError

    def run(self, run_id):
        raise NotImplementedError

    def resume(self, run_id):
        raise NotImplementedError
