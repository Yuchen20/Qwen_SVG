import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from svg_constraint import SVGConstraints
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

PROMPT_TEMPLATE = """### Instruction
Generate an SVG for the following description. Begin with a short reasoning enclosed in <reasoning>...</reasoning> tags. Follow the reasoning with a constrained SVG as per the specifications.

### Description
{description}

### Generation Process
Always start with a short reasoning section:
1. Identify visual elements from the prompt  
2. Choose a consistent and expressive color palette  
3. Lay out the composition within the 256×256 viewBox  
4. Create base shapes and background  
5. Add details, layering, and visual interest  
6. Use `<g>` and `transform` for grouping or transformations  
7. Review for clarity, balance, completeness, and constraint compliance  

### Constraints
- **Allowed Tags:**  
  `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
- **Allowed Attributes:**  
  `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
- **Output Format:**  
  Always begin the SVG with:
  ```xml
  <svg viewBox="0 0 256 256" width="256" height="256" xmlns="http://www.w3.org/2000/svg">
  ```
  And end with:
  ```xml
  </svg>
  ```
- **No `<text>` elements**, and no tags or attributes outside the allowed lists.  
- **Use HTML-style comments** (`<!-- ... -->`) before each visual block explaining its purpose and color."""

class SVGDataset:
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1792):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
                
        # Add special tokens if not present
        # special_tokens = {"additional_special_tokens": ["<reasoning>", "</reasoning>"]}
        # self.tokenizer.add_special_tokens(special_tokens)
    
    def prepare_dataset(self) -> DatasetDict:
        """Prepare the dataset for training by converting to HF Dataset format and splitting."""
        formatted_data = []
        for item in self.data:
            prompt = PROMPT_TEMPLATE.format(description=item['description'])
            response = f"{item['svg_content']}"
            
            # Format as instruction format
            formatted_data.append({
                "text": f"{prompt}\n\n{response}",
            })
        
        # Convert to HF Dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Split into train/validation/test (90/5/5)
        splits = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        test_valid = splits['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
        
        return DatasetDict({
            'train': splits['train'],
            'validation': test_valid['train'], 
            'test': test_valid['test']
        })

@dataclass 
class SVGMetrics:
    """Calculates various metrics for generated SVG content."""
    svg_constraints: SVGConstraints = SVGConstraints()
    
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        metrics = {}
        
        # Get the predictions and references
        predictions, labels = eval_preds
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        
        # Calculate metrics
        valid_svg_count = 0
        valid_reasoning_count = 0
        
        for pred in decoded_preds:
            # Check for valid SVG content
            svg_match = re.search(r'<svg.*?</svg>', pred, re.DOTALL)
            if svg_match:
                svg_content = svg_match.group(0)
                try:
                    self.svg_constraints.validate_svg(svg_content)
                    valid_svg_count += 1
                except ValueError:
                    pass
            
            # Check for valid reasoning section
            if re.search(r'<reasoning>.*?</reasoning>', pred, re.DOTALL):
                valid_reasoning_count += 1
        
        total = len(decoded_preds)
        metrics['valid_svg_ratio'] = valid_svg_count / total
        metrics['valid_reasoning_ratio'] = valid_reasoning_count / total
        
        return metrics

def create_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    peft_config: Optional[LoraConfig] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Initialize and prepare the model and tokenizer."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"  # Set padding to the left
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Add LoRA if config provided
    if peft_config:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        
    # Resize token embeddings for new special tokens
    # model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def create_peft_config(
    r: int = 64,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=target_modules,
        # lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

class WandbCallback(TrainerCallback):
    """Custom callback for logging to Weights & Biases."""
    def __init__(self, config):
        super().__init__()
        wandb.init(project="svg-generation", config=config)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to W&B."""
        if logs:
            wandb.log(logs)

def train(
    data_path: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_length: int = 1792,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    num_epochs: int = 10,
    warmup_ratio: float = 0.1,
    lora_rank: int = 64,
    fp16: bool = True,
):
    """Train the SVG generation model."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model, tokenizer and LoRA config
    peft_config = create_peft_config(r=lora_rank)
    model, tokenizer = create_model_and_tokenizer(model_name, peft_config)

    # Create dataset
    dataset = SVGDataset(data_path, tokenizer, max_length=max_length)
    dataset_dict = dataset.prepare_dataset()

    # Initialize metrics calculator
    metrics = SVGMetrics()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        logging_first_step=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="valid_svg_ratio",
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        report_to="none",  # We'll use custom W&B logging
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        compute_metrics=metrics.compute_metrics,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if fp16 else None,
        ),
        callbacks=[WandbCallback({
            "model_name": model_name,
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "num_epochs": num_epochs,
            "warmup_ratio": warmup_ratio,
            "fp16": fp16,
        })],
    )

    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    # Run evaluation on test set
    metrics = trainer.evaluate(dataset_dict["test"], metric_key_prefix="test")
    print(f"\nTest set metrics:\n{metrics}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SVG generation model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to svg_results.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-1.5B-Instruct", help="Base model to use")
    parser.add_argument("--max_length", type=int, default=1792, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA adaptation rank")
    parser.add_argument("--no_fp16", action="store_true", help="Disable mixed precision training")
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        lora_rank=args.lora_rank,
        fp16=not args.no_fp16,
    )