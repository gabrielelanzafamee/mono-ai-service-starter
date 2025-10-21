from dataclasses import dataclass

@dataclass
class Hyperparameters:
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    warmup_ratio: float
    logging_steps: int
    optim: str
    max_grad_norm: float
    warmup_steps: int = 0
    
    # Evaluation and saving parameters
    eval_steps: int = 100
    save_steps: int = 500
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    
    # Model and training configuration
    max_length_limit: int = 1024  # Maximum sequence length limit
    packing: bool = True
    padding_free: bool = False
    activation_offloading: bool = False
    group_by_length: bool = True
    
    # Early stopping configuration
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Weight decay
    weight_decay: float = 0.01

@dataclass
class LoraConfig:
    lora_alpha: int
    lora_dropout: float
    r: int
    bias: str
    task_type: str
    target_modules: list[str]

@dataclass
class Config:
    hyperparameters: Hyperparameters
    lora_config: LoraConfig
    model_id: str
    dataset_id: str
    lora_id: str
    project_name: str