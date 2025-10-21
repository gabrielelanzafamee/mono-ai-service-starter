from dotenv import load_dotenv
load_dotenv()

import os

print("HF_TOKEN", os.environ["HF_TOKEN"][:5])
print("WANDB_TOKEN", os.environ["WANDB_TOKEN"][:5])
print("OPENAI_API_KEY", os.environ["OPENAI_API_KEY"][:5])

# login to wandb
import wandb
wandb.login(key=os.environ["WANDB_TOKEN"])
# login to huggingface
import huggingface_hub
huggingface_hub.login(token=os.environ["HF_TOKEN"])

import gc
import torch
import asyncio
import argparse
import flash_attn

from trainer import Trainer
from evaluate import Evaluator
from settings import Config, Hyperparameters, LoraConfig

# validate torch and flash attn
print(f"Torch version: {torch.__version__}")
print(f"Flash attn version: {flash_attn.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA not available - running on CPU")

def parse_arguments():
    """
    Parse command line arguments for model training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with model_id, dataset_id, lora_id, and project_name
    """
    parser = argparse.ArgumentParser(description="Train a language model with LoRA fine-tuning")
    
    # Add arguments for the four parameters we want to make configurable
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-3-4b-it",
        help="Model ID to use for training (default: google/gemma-3-4b-it)"
    )
    
    parser.add_argument(
        "--dataset_id", 
        type=str,
        default="DeepMount00/italian_conversations",
        help="Dataset ID to use for training (default: DeepMount00/italian_conversations)"
    )
    
    parser.add_argument(
        "--lora_id",
        type=str,
        default="xjabrvccp/gemma-lora-italian",
        help="LoRA ID for the model (default: xjabrvccp/gemma-lora-italian)"
    )
    
    parser.add_argument(
        "--project_name",
        type=str,
        default="gemma3-lora-italian",
        help="Project name for training (default: gemma3-lora-italian)"
    )
    
    # some hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (default: 2)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps for training (default: 4)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=8e-5,
        help="Learning rate for training (default: 8e-5)"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.15,
        help="Warmup ratio for training (default: 0.15)"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for training (default: 0)"
    )
    
    parser.add_argument(
        "--max_length_limit",
        type=int,
        default=2048,
        help="Maximum sequence length limit (default: 2048)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # load the config with parsed arguments
    config = Config(
        hyperparameters=Hyperparameters(
            # Basic training parameters
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            warmup_steps=args.warmup_steps,
            logging_steps=10,
            optim="paged_adamw_32bit",
            max_grad_norm=1.0,
            
            # Evaluation and saving parameters
            eval_steps=100,  # Evaluate every 100 steps
            save_steps=500,  # Save every 500 steps
            
            # Learning rate scheduler
            lr_scheduler_type="cosine",
            
            # Model and training configuration
            max_length_limit=args.max_length_limit,  # Maximum sequence length limit
            packing=True,  # Enable packing for efficient training
            padding_free=False,  # Disable padding-free mode
            activation_offloading=False,  # Disable activation offloading
            group_by_length=True,  # Group sequences by length for efficiency
            
            # Early stopping configuration
            load_best_model_at_end=True,  # Load best model at end
            metric_for_best_model="eval_loss",  # Use eval loss for best model
            greater_is_better=False,  # Lower loss is better
            
            # Weight decay
            weight_decay=0.01
        ),
        lora_config=LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        ),
        model_id=args.model_id,  # Use parsed argument
        dataset_id=args.dataset_id,  # Use parsed argument
        lora_id=args.lora_id,  # Use parsed argument
        project_name=args.project_name  # Use parsed argument
    )

    # # train the model
    print("Training the model...")
    trainer = Trainer(config=config)
    trainer.train()
    
    # clean VRAM
    gc.collect()
    torch.cuda.empty_cache()
    
    # evaluate the model
    print("Evaluating the model...")
    asyncio.run(Evaluator(config=config).run())