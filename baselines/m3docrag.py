from .base import ContextBuilder, ContextMessages

class m3docragContextBuilder(ContextBuilder):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        
    def build_mmlongbench(self, sample, **kwargs):
        raise NotImplementedError('m3docragContextBuilder does not support mmlongbench')
    
    def build_longdocurl(self, sample, **kwargs):
        raise NotImplementedError('m3docragContextBuilder does not support longdocurl')