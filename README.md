# Qwen_SVG


# evironment
```bash
python venv venv (Qwen_SVG)
source "venv (Qwen_SVG)/bin/activate"
pip install -r requirements.txt
# to kernel
python -m ipykernel install --user --name="venv (Qwen_SVG)" --display-name="venv (Qwen_SVG)"
```






First, look at `Copilot\prime_directive.md` for the prime directive.
---
Write a supervised training script using the Huggingface Trainer.

Use svg_results.jsonl as training data. Each sample is a dict with:

json
Copy
Edit
{
  "id": "...",
  "description": "...",
  "svg_content": "<reasoning>...</reasoning><svg>...</svg>"
}
Split data into 90/5/5 for train/valid/test

Base model: Qwen/Qwen2.5-1.5B-Instruct

Max token length = 1560 (for now). Add a preprocessing script to visualize token length distribution.

Use left-padding

Training Setup
LoRA (from PEFT), with rank = 64

Optimizer: AdamW

Batch size: 4

Learning rate: 5e-5

LR scheduling with warmup

Mixed precision (fp16 if available)

Logging to Weights & Biases (logging_steps=10)

Trainer Arguments
eval_strategy="steps" with eval_steps=50
weight_decay=0.01,
max_grad_norm=1.0,
logging_steps=10,
save_steps=200,
save_total_limit=3,

Only save LoRA adapters (not full model)

Prompt Format
Add a unified instruction template that tells the model to:

Begin with <reasoning>...</reasoning>

Follow the given constraints for allowed SVG tags/attributes

Add comments for each visual block

Metrics
Cross-entropy loss on validation set

For the final evaluation, use the test set and compute:
- Cross-entropy loss
- accuracy of SVG generation (Generating a valid <svg>...</svg> section using regex)
- accuracy of reasoning generation (Generating a valid <reasoning>...</reasoning> section using regex)
- report the last 5 samples of the test set to wandb with extracted reasoning and SVG content
- A custom metric using svg_constraint.py for SVG structural correctness


---
Unified Prompt Template

You can use this as a consistent formatting structure in your data preprocessing:
so use this to all your data samples.

---

### 📌 Prompt Template (with embedded structure and constraints):

```
### Instruction
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
```

---

