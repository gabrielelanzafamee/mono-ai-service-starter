from abc import ABC, abstractmethod

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig as PeftLoraConfig

from settings import Config

class LLM(ABC):
    device: str
    dtype: torch.dtype

    model_id: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    processor: AutoProcessor
    peft_config: PeftLoraConfig

    download_model_folder: str
    output_model_folder: str

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model_id = config.model_id
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.peft_config = PeftLoraConfig(
            lora_alpha=self.config.lora_config.lora_alpha,
            lora_dropout=self.config.lora_config.lora_dropout,
            r=self.config.lora_config.r,
            bias=self.config.lora_config.bias,
            task_type=self.config.lora_config.task_type,
            target_modules=self.config.lora_config.target_modules,
        )
        
        self.download_model_folder = "./models"
        self.output_model_folder = "./outputs"

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_lora(self):
        pass

    @abstractmethod
    def unload_lora(self):
        pass

    @abstractmethod
    def get_peft_model(self):
        pass
    
    @abstractmethod
    def generate(self):
        pass
    
    @abstractmethod
    def reset_model(self):
        pass