import torch
import torch.nn as nn
import transformers
import os
import wandb

from trl import SFTTrainer, SFTConfig

from llm.gemma import Gemma
from preprocess import Dataset
from settings import Config

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.llm = Gemma(self.config)
        self.dataset = Dataset(self.config.dataset_id)
        # set the wandb project where this run will be logged
        os.environ["WANDB_PROJECT"]=self.config.project_name
        os.environ["WANDB_LOG_MODEL"]="end"
        os.environ["WANDB_WATCH"]="false"

    def train(self):
        # setup dataset
        self.train_dataset = self.dataset.train
        self.test_dataset = self.dataset.test

        # load model
        self.load_model()
        
        # analyse dataset
        self.train_max_lenght, self.test_max_lenght = self.dataset.analyse_dataset(self.llm)
   
        # load training configs with all configurable hyperparameters
        self.training_arguments = SFTConfig(
            # Run configuration
            run_name=f"{self.config.lora_id}_{self.config.dataset_id}_{self.config.model_id}".replace("/", "-"),
            logging_strategy="steps",
            
            # Evaluation configuration
            eval_strategy="steps",  # Evaluate during training at regular intervals
            eval_steps=self.config.hyperparameters.eval_steps,  # Evaluate every N steps
            
            # Model saving configuration
            save_strategy="steps",  # Save checkpoints at regular intervals
            save_steps=self.config.hyperparameters.save_steps,  # Save every N steps
            
            # Learning rate configuration
            lr_scheduler_type=self.config.hyperparameters.lr_scheduler_type,  # Cosine, linear, etc.

            # Mixed precision configuration
            fp16=self.llm.dtype == torch.float16,  # Use FP16 if available
            bf16=self.llm.dtype == torch.bfloat16,  # Use BF16 if available

            # Sequence length configuration
            max_length=min(self.train_max_lenght, self.config.hyperparameters.max_length_limit),  # Limit max sequence length
            
            # Remove model_init_kwargs since we're passing an already instantiated model
            # model_init_kwargs is only needed when SFTTrainer needs to instantiate the model

            # Training efficiency configuration
            packing=self.config.hyperparameters.packing,  # Enable packing for efficient training
            padding_free=self.config.hyperparameters.padding_free,  # Disable padding-free mode
            activation_offloading=self.config.hyperparameters.activation_offloading,  # Disable activation offloading
            group_by_length=self.config.hyperparameters.group_by_length,  # Group sequences by length for efficiency
            
            # Dataset configuration
            dataset_kwargs={
                "add_special_tokens": True,  # Add special tokens to sequences
                "append_concat_token": True  # Append concatenation token
            },
            
            # Early stopping configuration
            load_best_model_at_end=self.config.hyperparameters.load_best_model_at_end,  # Load best model at end
            metric_for_best_model=self.config.hyperparameters.metric_for_best_model,  # Use eval loss for best model
            greater_is_better=self.config.hyperparameters.greater_is_better,  # Lower loss is better
            
            # Regularization
            weight_decay=self.config.hyperparameters.weight_decay,  # Weight decay for regularization

            # Logging and output configuration
            report_to="wandb",  # Report to Weights & Biases
            output_dir=f"{self.llm.output_model_folder}/{self.config.lora_id}",
            
            # Training parameters
            per_device_train_batch_size=self.config.hyperparameters.batch_size,
            per_device_eval_batch_size=self.config.hyperparameters.batch_size,
            gradient_accumulation_steps=self.config.hyperparameters.gradient_accumulation_steps,
            optim=self.config.hyperparameters.optim,
            max_grad_norm=self.config.hyperparameters.max_grad_norm,
            num_train_epochs=self.config.hyperparameters.num_train_epochs,
            logging_steps=self.config.hyperparameters.logging_steps,
            warmup_ratio=self.config.hyperparameters.warmup_ratio,
            warmup_steps=self.config.hyperparameters.warmup_steps,
            learning_rate=self.config.hyperparameters.learning_rate,
        )
        
        # load SFTTrainer
        # When packing=True, SFTTrainer handles data collation internally
        # so we don't pass a custom data_collator
        trainer_kwargs = {
            "model": self.llm.model,
            "args": self.training_arguments,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.test_dataset,
            # Remove peft_config since model is already PEFT-enabled
        }
        
        # Only add data_collator if packing is disabled
        if not self.training_arguments.packing and not self.training_arguments.padding_free:
            trainer_kwargs["data_collator"] = self.data_collator
            
        self.trainer = SFTTrainer(**trainer_kwargs)

        # train the model
        torch.cuda.empty_cache()
        self.trainer.train()

        # save the model
        try:
            self.save_model()
        except Exception as e:
            print(f"Error saving model: {e}")
        
        # finish wandb
        wandb.finish()

    def save_model(self):
        self.trainer.model.save_pretrained(f"{self.llm.output_model_folder}/{self.config.lora_id}", save_adapter=True, save_config=True)
        self.trainer.model.push_to_hub(f"{self.config.lora_id}", private=True)

    def load_model(self):
        self.llm.load_model(quantization=False)
        
        # Create PEFT model - this should be done before SFTTrainer initialization
        self.llm.model = self.llm.get_peft_model()
        
        self.data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer=self.llm.tokenizer,
            mlm=False
        )

        # move to train mode
        self.llm.model.train()

        # PEFT handles parameter freezing automatically, so we don't need to do it manually
        # Remove the manual parameter freezing loop since PEFT handles this
        
        # Define CastOutputToFloat class outside the loop
        class CastOutputToFloat(nn.Sequential):
            def set_dtype(self, dtype):
                self.dtype = dtype
                return self
            
            def forward(self, x):
                return super().forward(x).to(self.dtype)

        # Configure model settings
        self.llm.model.gradient_checkpointing_enable()
        self.llm.model.enable_input_require_grads()
        
        # For PEFT models, we need to access lm_head through the base model
        try:
            if hasattr(self.llm.model, 'base_model'):
                # PEFT model - access through base_model
                self.llm.model.base_model.lm_head = CastOutputToFloat(self.llm.model.base_model.lm_head).set_dtype(self.llm.dtype)
            else:
                # Regular model
                self.llm.model.lm_head = CastOutputToFloat(self.llm.model.lm_head).set_dtype(self.llm.dtype)
        except AttributeError as e:
            print(f"Warning: Could not configure lm_head casting: {e}")
            # Continue without lm_head casting if it fails
