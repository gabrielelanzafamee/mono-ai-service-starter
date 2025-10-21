from typing import List, Dict, Any

import torch

from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import get_peft_model, PeftModel

from llm import LLM

class Gemma(LLM):
    def load_model(self, quantization=False):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, torch_dtype=self.dtype, cache_dir=self.download_model_folder)
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": self.dtype,
            "cache_dir": self.download_model_folder,
            "attn_implementation": "flash_attention_2",  # Add this for efficiency
            "low_cpu_mem_usage": True,  # Add this for memory efficiency
        }
        
        if quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
        self.model = Gemma3ForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.download_model_folder)

    def load_lora(self, lora_id, from_hub=False):
        if from_hub:
            model_path = lora_id
        else:
            model_path = f"./outputs/{lora_id}"
        
        print(f"Loading LoRA from {model_path}")
        model_to_merge = PeftModel.from_pretrained(self.model, model_path)
        self.model = model_to_merge.merge_and_unload()
    
    def unload_lora(self):
        self.model = self.model.base_model
    
    def get_peft_model(self):
        return get_peft_model(self.model, self.peft_config)
    
    def generate(self, messages: List[Dict[str, Any]]):
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        
        if self.processor is None:
            raise ValueError("Processor not loaded")
        
        # set model to eval mode
        self.model.eval()
        
        # generate the response
        with torch.no_grad():
            # Step 1: Apply chat template and tokenize input messages
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device, dtype=self.model.dtype)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Step 2: Generate response tokens
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    do_sample=False, 
                    max_new_tokens=1024, 
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, 
                    num_return_sequences=1,
                    early_stopping=True
                )
                predicted_tokens = generation[0][input_len:]
            
            generated_text = self.processor.decode(predicted_tokens, skip_special_tokens=True)
            return generated_text
    
    def reset_model(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.peft_config = None
        # clean torch cache
        torch.cuda.empty_cache()
        return self