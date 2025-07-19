from transformers import LlamaConfig

class LLamaGraftConfig(LlamaConfig):
    model_type = "llama_graft"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)