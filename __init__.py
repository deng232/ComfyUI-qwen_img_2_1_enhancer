from comfy_api.latest import ComfyExtension

from .qwen_image21_attention_probe import Qwen21ReferenceAttentionPreview, Qwen21ReferenceAttentionProbe
from .qwen_image21_conditioning_phrase_encoder import QwenImage21ConditioningPhraseEncoder
from .qwen_image21_reference_strength import QwenImage21ReferenceStrength
from .qwen_image21_reference_strength_masked import QwenImage21ReferenceStrengthMasked


class QwenImage21EnhancerExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            Qwen21ReferenceAttentionProbe,
            Qwen21ReferenceAttentionPreview,
            QwenImage21ReferenceStrength,
            QwenImage21ReferenceStrengthMasked,
            QwenImage21ConditioningPhraseEncoder,
        ]


async def comfy_entrypoint():
    return QwenImage21EnhancerExtension()
