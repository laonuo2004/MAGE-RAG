from typing import Any, Dict


class ContextMessages(list):

    def __init__(self, messages, metadata: Dict[str, Any] | None = None):
        super().__init__(messages)
        self.metadata = metadata or {}


class ContextBuilder:
    name = None

    def __init__(self, cfg=None):
        self.cfg = cfg

    def build(self, benchmark_name, sample, **kwargs):
        if benchmark_name == 'mmlongbench':
            return self.build_mmlongbench(sample, **kwargs)
        if benchmark_name == 'longdocurl':
            return self.build_longdocurl(sample, **kwargs)
        raise ValueError(f'Unsupported benchmark for context builder: {benchmark_name}')

    def build_mmlongbench(self, sample, **kwargs):
        raise NotImplementedError

    def build_longdocurl(self, sample, **kwargs):
        raise NotImplementedError
