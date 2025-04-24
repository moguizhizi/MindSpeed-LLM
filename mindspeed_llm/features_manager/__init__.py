from mindspeed.features_manager.tensor_parallel.unaligned_linear_feature import UnalignedLinearFeature

from mindspeed_llm.features_manager.common.training import TrainingDefaultFeature
from mindspeed_llm.features_manager.communication.gloo import DisableGlooFeature

FEATURES_LIST = [
    # MindSpeed Legacy Features
    
    # MindSpeed Mcore Features
    UnalignedLinearFeature(),

    
    # MindSpeed-LLM Mcore Features
    TrainingDefaultFeature(),
    DisableGlooFeature(),

    # MindSpeed-LLM Legacy Features
]