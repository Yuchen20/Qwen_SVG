import os
import re
import json
import numpy as np
import wandb
import torch
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback

from svg_constraint import SVGConstraints

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Model and data configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/svg_results.jsonl"
OUTPUT_DIR = "models/qwen-svg-lora"
MAX_LENGTH = 1560
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
LORA_RANK = 64
LEFT_PADDING = True  # Use left padding as specified

# Function to create the prompt from description
def create_prompt(description):
    return f"""### Instruction
Generate an SVG for the following description. Begin with a short reasoning enclosed in <reasoning>...</reasoning> tags. Follow the reasoning with a constrained SVG as per the specifications.

### Description
{description}

### 🧠 Generation Process
Always start with a short reasoning section:
1. Identify visual elements from the prompt  
2. Choose a consistent and expressive color palette  
3. Lay out the composition within the 256×256 viewBox  
4. Create base shapes and background  
5. Add details, layering, and visual interest  
6. Use `<g>` and `transform` for grouping or transformations  
7. Review for clarity, balance, completeness, and constraint compliance  

### ⚙️ Constraints
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
- **Use HTML-style comments** (`<!-- ... -->`) before each visual block explaining its purpose and color.
"""

# Custom data collator that handles left padding
@dataclass
class LeftPaddingDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Convert all inputs to torch tensors
        batch = {}
        
        # Handle input_ids with left padding
        input_ids = [feature["input_ids"] for feature in features]
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        attention_mask = []
        
        for ids in input_ids:
            padding_length = max_length - len(ids)
            if LEFT_PADDING:
                padded_ids = [self.tokenizer.pad_token_id] * padding_length + ids
                mask = [0] * padding_length + [1] * len(ids)
            else:
                padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
                mask = [1] * len(ids) + [0] * padding_length
            
            padded_input_ids.append(padded_ids)
            attention_mask.append(mask)
        
        batch["input_ids"] = torch.tensor(padded_input_ids)
        batch["attention_mask"] = torch.tensor(attention_mask)
        
        # Handle labels for language modeling
        if "labels" in features[0]:
            labels = [feature["labels"] for feature in features]
            padded_labels = []
            
            for lab in labels:
                padding_length = max_length - len(lab)
                if LEFT_PADDING:
                    padded_lab = [-100] * padding_length + lab  # -100 is ignored in loss calculation
                else:
                    padded_lab = lab + [-100] * padding_length
                
                padded_labels.append(padded_lab)
            
            batch["labels"] = torch.tensor(padded_labels)
        
        return batch

# Custom Wandb callback to track metrics and log samples
class CustomWandbCallback(WandbCallback):
    def __init__(self, trainer, tokenizer):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.svg_constraints = SVGConstraints()
    
    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        
        if hasattr(self.trainer, 'eval_dataset') and self.trainer.eval_dataset is not None:
            # Get the last 5 samples from eval dataset
            if len(self.trainer.eval_dataset) >= 5:
                samples_to_log = self.trainer.eval_dataset[-5:]
            else:
                samples_to_log = self.trainer.eval_dataset
            
            # Generate outputs for these samples
            for i, sample in enumerate(samples_to_log):
                input_text = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                
                # Skip very long inputs to avoid CUDA OOM
                if len(sample["input_ids"]) > MAX_LENGTH - 200:
                    continue
                
                # Generate with the model
                with torch.no_grad():
                    input_ids = torch.tensor([sample["input_ids"]]).to(self.trainer.model.device)
                    attention_mask = torch.tensor([[1] * len(sample["input_ids"])]).to(self.trainer.model.device)
                    
                    try:
                        outputs = self.trainer.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=1000,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        
                        generated_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
                        
                        # Log to wandb
                        wandb.log({
                            f"sample_{i}/input": wandb.Html(f"<pre>{input_text}</pre>"),
                            f"sample_{i}/generated": wandb.Html(f"<pre>{generated_text}</pre>"),
                        })
                        
                        # Try to extract and render SVG
                        svg_match = re.search(r"<svg.*?</svg>", generated_text, re.DOTALL)
                        if svg_match:
                            svg_content = svg_match.group(0)
                            wandb.log({f"sample_{i}/svg_render": wandb.Html(svg_content)})
                    except Exception as e:
                        print(f"Error generating or logging sample {i}: {e}")

# Custom metrics for evaluating SVG generation
class SVGMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.svg_constraints = SVGConstraints()
    
    def extract_svg(self, text):
        svg_match = re.search(r"<svg.*?</svg>", text, re.DOTALL)
        if svg_match:
            return svg_match.group(0)
        return None
    
    def extract_reasoning(self, text):
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1)
        return None
    
    def compute_metrics(self, eval_preds):
        predictions, labels = eval_preds
        
        # Decode predictions
        decoded_preds = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        
        # Initialize metrics
        metrics = {
            "cross_entropy_loss": float(np.mean(eval_preds[1])),
            "svg_generation_accuracy": 0,
            "reasoning_generation_accuracy": 0,
            "svg_constraint_compliance": 0
        }
        
        # Calculate SVG generation accuracy
        svg_count = 0
        for pred in decoded_preds:
            if self.extract_svg(pred) is not None:
                svg_count += 1
        
        metrics["svg_generation_accuracy"] = svg_count / len(decoded_preds) if len(decoded_preds) > 0 else 0
        
        # Calculate reasoning generation accuracy
        reasoning_count = 0
        for pred in decoded_preds:
            if self.extract_reasoning(pred) is not None:
                reasoning_count += 1
        
        metrics["reasoning_generation_accuracy"] = reasoning_count / len(decoded_preds) if len(decoded_preds) > 0 else 0
        
        # Calculate SVG constraint compliance
        valid_svg_count = 0
        for pred in decoded_preds:
            svg = self.extract_svg(pred)
            if svg is not None:
                try:
                    self.svg_constraints.validate_svg(svg)
                    valid_svg_count += 1
                except ValueError:
                    pass
        
        metrics["svg_constraint_compliance"] = valid_svg_count / svg_count if svg_count > 0 else 0
        
        return metrics

def main():
    # Initialize wandb
    wandb.init(project="qwen-svg-finetuning", name="lora-run")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        # load_in_4bit=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    
    # Load and preprocess the dataset
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    
    # Filter out samples that exceed max length
    filtered_dataset = []
    for sample in dataset:
        # Create input prompt
        prompt = create_prompt(sample["description"])
        target = sample["svg_content"]
        
        # Tokenize input and target
        input_tokens = tokenizer.encode(prompt)
        target_tokens = tokenizer.encode(target)
        
        # Check token length
        if len(input_tokens) + len(target_tokens) <= MAX_LENGTH:
            filtered_dataset.append({
                "id": sample["id"],
                "description": sample["description"],
                "svg_content": sample["svg_content"],
                "prompt": prompt,
                "tokens": len(input_tokens) + len(target_tokens)
            })
    
    print(f"Filtered dataset size: {len(filtered_dataset)} out of {len(dataset)} samples")
    
    # Convert to Dataset object
    filtered_dataset = Dataset.from_list(filtered_dataset)
    
    # Split data into train/valid/test (90/5/5)
    splits = filtered_dataset.train_test_split(test_size=0.1, seed=seed)
    train_dataset = splits["train"]
    temp_eval = splits["test"]
    
    eval_test_splits = temp_eval.train_test_split(test_size=0.5, seed=seed)
    eval_dataset = eval_test_splits["train"]
    test_dataset = eval_test_splits["test"]
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(eval_dataset)}, Test size: {len(test_dataset)}")
    
    # Tokenize the datasets
    def tokenize_function(examples):
        # Format the examples
        prompts = [create_prompt(desc) for desc in examples["description"]]
        targets = examples["svg_content"]
        
        # Tokenize inputs and outputs
        inputs = tokenizer(prompts, padding=False, truncation=False)
        outputs = tokenizer(targets, padding=False, truncation=False)
        
        # Combine input_ids and prepare labels
        examples["input_ids"] = []
        examples["labels"] = []
        
        for i in range(len(prompts)):
            input_ids = inputs["input_ids"][i]
            output_ids = outputs["input_ids"][i]
            
            # Labels: -100 for input tokens (ignored in loss), actual ids for output tokens
            labels = [-100] * len(input_ids) + output_ids
            
            # Combined input_ids
            combined_input_ids = input_ids + output_ids
            
            # Truncate if needed
            if len(combined_input_ids) > MAX_LENGTH:
                combined_input_ids = combined_input_ids[:MAX_LENGTH]
                labels = labels[:MAX_LENGTH]
            
            examples["input_ids"].append(combined_input_ids)
            examples["labels"].append(labels)
        
        return examples
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["id", "description", "svg_content", "prompt", "tokens"])
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=["id", "description", "svg_content", "prompt", "tokens"])
    test_tokenized = test_dataset.map(tokenize_function, batched=True, remove_columns=["id", "description", "svg_content", "prompt", "tokens"])
    
    # Create data collator
    data_collator = LeftPaddingDataCollator(
        tokenizer=tokenizer,
        mlm=False,
        mlm_probability=0
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=WARMUP_RATIO,
        logging_dir="logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"],
        remove_unused_columns=False,
    )
    
    # Create SVG metrics
    svg_metrics = SVGMetrics(tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=svg_metrics.compute_metrics,
    )
    
    # Add custom wandb callback
    custom_wandb_callback = CustomWandbCallback(trainer, tokenizer)
    trainer.add_callback(custom_wandb_callback)
    
    # Start training
    trainer.train()
    
    # Save the PEFT adapter only
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    
    print("Test Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value}")
    
    # Log test results to wandb
    wandb.log({"test": test_results})
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()