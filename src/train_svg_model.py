import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Define our own EvalLoopOutput since it can't be imported
@dataclass
class EvalLoopOutput:
    predictions: Optional[Any] = None
    label_ids: Optional[Any] = None
    metrics: Optional[Dict[str, float]] = None
    num_samples: Optional[int] = None

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
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
                

    
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
        
        # Tokenize the dataset
        def tokenize_function(examples):
            # This is for causal language modeling, so we don't need labels
            return self.tokenizer(examples["text"], padding=False, truncation=True, max_length=self.max_length)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        # Split into train/validation/test (90/5/5)
        splits = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
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
    tokenizer: PreTrainedTokenizer = None
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
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
    def __init__(self, config, tokenizer, dataset_dict=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_dict = dataset_dict
        self.sample_idx = 0  # For rotating through validation samples
        wandb.init(project="svg-generation", config=config)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to W&B."""
        if logs:
            wandb.log(logs)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log sample generations from validation set."""
        if not metrics:
            return
        
        model = kwargs.get('model')
        if model is None and hasattr(state, "model"):
            model = state.model
            
        if model is None:
            return
            
        try:
            # Get a validation sample (rotate through samples over time)
            if self.dataset_dict and 'validation' in self.dataset_dict:
                validation_dataset = self.dataset_dict['validation']
                if len(validation_dataset) > 0:
                    # Rotate through validation samples
                    self.sample_idx = (self.sample_idx + 1) % len(validation_dataset)
                    sample = {k: v.unsqueeze(0).to(args.device) for k, v in validation_dataset[self.sample_idx].items()}
                    
                    # Get original input text for reference
                    input_ids = sample["input_ids"][0].detach().cpu().tolist()
                    input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                    
                    # Find the description in the input for a cleaner log
                    description = "N/A"
                    match = re.search(r'### Description\s*(.*?)(?=###)', input_text, re.DOTALL)
                    if match:
                        description = match.group(1).strip()
                    
                    # Generate output
                    with torch.no_grad():
                        outputs = model.generate(
                            **sample,
                            max_new_tokens=2400,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    
                    # Decode the output
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                    
                    # Find SVG content in the generated text
                    svg_match = re.search(r'<svg.*?</svg>', generated_text, re.DOTALL)
                    svg_content = svg_match.group(0) if svg_match else "No SVG found in generated output"
                    
                    # Log sample info to wandb
                    wandb.log({
                        "validation_description": description,
                        "validation_sample_generation": wandb.Html(f"<pre>{generated_text}</pre>"),
                        "validation_svg": wandb.Html(f"{svg_content}"),
                    })
        except Exception as e:
            print(f"Error logging validation sample: {e}")
            import traceback
            traceback.print_exc()

class MemoryEfficientTrainer(Trainer):
    """Custom trainer with a more memory-efficient evaluation."""
    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Override the evaluation loop to save memory by not storing all predictions.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if not None else args.prediction_loss_only
        
        # Initialize metrics dict and set model to eval mode
        metrics = {}
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        # Disable adding preds to outputs by default to save memory
        compute_metrics = self.compute_metrics
        
        # Main evaluation loop
        batch_size = dataloader.batch_size
        num_samples = len(dataloader.dataset)
        num_batches = len(dataloader)
        
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_samples}")
        logger.info(f"  Batch size = {batch_size}")
        
        # Initialize variables for metrics
        total_loss = 0.0
        total_samples = 0
        
        # For SVG metrics
        all_preds = []
        all_labels = []
        max_samples_for_metrics = min(100, num_samples)  # Limit samples to save memory
        samples_seen = 0
        
        for step, inputs in enumerate(dataloader):
            # Move inputs to appropriate device
            inputs = self._prepare_inputs(inputs)
            
            # Forward pass with no_grad context
            with torch.no_grad():
                # Get outputs and loss
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Update loss stats
                if loss is not None:
                    # Get the actual batch size from the input tensors
                    current_batch_size = inputs["input_ids"].size(0)
                    total_loss += loss.detach().float() * current_batch_size
                    total_samples += current_batch_size
            
            # Collect predictions and labels for a limited number of samples
            if samples_seen < max_samples_for_metrics:
                # Get predictions
                logits = outputs.logits.detach().cpu()
                
                # Get the number of samples we can add without exceeding our limit
                samples_to_add = min(logits.shape[0], max_samples_for_metrics - samples_seen)
                
                # Add samples for metrics computation
                if samples_to_add > 0:
                    all_preds.append(logits[:samples_to_add])
                    
                    # Get labels (masked with -100)
                    labels = inputs.get("labels").detach().cpu()
                    all_labels.append(labels[:samples_to_add])
                    
                    samples_seen += samples_to_add
            
            # Log progress
            if step % args.logging_steps == 0 and step > 0:
                logger.info(f"  Evaluation: {step}/{num_batches} steps complete")
        
        # Compute loss metrics
        if total_samples > 0:
            metrics[f"{metric_key_prefix}_loss"] = total_loss.item() / total_samples
        
        # Compute model-specific metrics if available
        if compute_metrics is not None and len(all_preds) > 0:
            # Concatenate predictions and labels
            eval_preds = torch.cat(all_preds, dim=0).numpy()
            eval_labels = torch.cat(all_labels, dim=0).numpy()
            
            # Compute metrics
            metric_outputs = compute_metrics((eval_preds, eval_labels))
            
            # Update metrics dict
            metrics.update({f"{metric_key_prefix}_{k}": v for k, v in metric_outputs.items()})
        
        # Return metrics
        return EvalLoopOutput(
            predictions=None,  # We don't store full predictions to save memory
            label_ids=None,    # We don't store full labels to save memory
            metrics=metrics,
            num_samples=num_samples,
        )

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
    metrics = SVGMetrics(tokenizer)
    
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
        # Evaluate at fixed intervals during training but only compute loss
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        # Use validation loss as the metric for selecting the best model
        metric_for_best_model="eval_loss",
        # Lower loss is better
        greater_is_better=False,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        report_to="none",  # We'll use custom W&B logging
        remove_unused_columns=False,  # Added to fix potential column issues
        # Do not compute SVG metrics during training evaluation steps
        prediction_loss_only=True,
    )

    # Initialize the Trainer
    trainer = MemoryEfficientTrainer(
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
        }, tokenizer, dataset_dict)],
    )

    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    # Run full evaluation with metrics now that training is complete
    print("\n=== Running final validation set evaluation with full metrics ===")
    # Disable prediction_loss_only to compute all metrics
    trainer.args.prediction_loss_only = False
    val_metrics = trainer.evaluate(dataset_dict["validation"], metric_key_prefix="final_val")
    print(f"\nValidation set metrics:\n{val_metrics}")
    
    # Run evaluation on test set with full metrics
    print("\n=== Running final test set evaluation with full metrics ===")
    test_metrics = trainer.evaluate(dataset_dict["test"], metric_key_prefix="final_test")
    print(f"\nTest set metrics:\n{test_metrics}")
    
    # Log final metrics to wandb
    wandb.log({
        "final_validation_metrics": val_metrics,
        "final_test_metrics": test_metrics
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SVG generation model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to svg_results.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model to use")
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