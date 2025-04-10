import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

# Set up the tokenizer
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define the prompt template
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

# Load the dataset
data_path = os.path.join("data", "svg_results.jsonl")
dataset = load_dataset("json", data_files=data_path)["train"]

# Calculate token lengths for both input prompts and target outputs
input_token_lengths = []
output_token_lengths = []
total_token_lengths = []

for sample in dataset:
    # Create the input prompt
    input_prompt = create_prompt(sample["description"])
    input_tokens = tokenizer.encode(input_prompt)
    input_token_length = len(input_tokens)
    input_token_lengths.append(input_token_length)
    
    # Target output
    output_tokens = tokenizer.encode(sample["svg_content"])
    output_token_length = len(output_tokens)
    output_token_lengths.append(output_token_length)
    
    # Total token length (input + output)
    total_token_length = input_token_length + output_token_length
    total_token_lengths.append(total_token_length)

# Create a DataFrame for easy analysis
token_data = pd.DataFrame({
    'input_length': input_token_lengths,
    'output_length': output_token_lengths,
    'total_length': total_token_lengths
})

# Calculate statistics
stats = {
    'input': {
        'min': token_data['input_length'].min(),
        'max': token_data['input_length'].max(),
        'mean': token_data['input_length'].mean(),
        'median': token_data['input_length'].median(),
        'p95': token_data['input_length'].quantile(0.95),
    },
    'output': {
        'min': token_data['output_length'].min(),
        'max': token_data['output_length'].max(),
        'mean': token_data['output_length'].mean(),
        'median': token_data['output_length'].median(),
        'p95': token_data['output_length'].quantile(0.95),
    },
    'total': {
        'min': token_data['total_length'].min(),
        'max': token_data['total_length'].max(),
        'mean': token_data['total_length'].mean(),
        'median': token_data['total_length'].median(),
        'p95': token_data['total_length'].quantile(0.95),
    }
}

print("Token Length Statistics:")
print(f"Input Tokens: Min={stats['input']['min']}, Max={stats['input']['max']}, Mean={stats['input']['mean']:.2f}, Median={stats['input']['median']}, 95th Percentile={stats['input']['p95']}")
print(f"Output Tokens: Min={stats['output']['min']}, Max={stats['output']['max']}, Mean={stats['output']['mean']:.2f}, Median={stats['output']['median']}, 95th Percentile={stats['output']['p95']}")
print(f"Total Tokens: Min={stats['total']['min']}, Max={stats['total']['max']}, Mean={stats['total']['mean']:.2f}, Median={stats['total']['median']}, 95th Percentile={stats['total']['p95']}")

# Plot the distributions
plt.figure(figsize=(12, 10))

# Create a 2x2 subplot
plt.subplot(2, 2, 1)
sns.histplot(data=token_data, x='input_length', kde=True)
plt.title('Input Token Length Distribution')
plt.axvline(x=1560, color='red', linestyle='--', label='Max Length (1560)')
plt.legend()

plt.subplot(2, 2, 2)
sns.histplot(data=token_data, x='output_length', kde=True)
plt.title('Output Token Length Distribution')
plt.axvline(x=1560, color='red', linestyle='--', label='Max Length (1560)')
plt.legend()

plt.subplot(2, 2, 3)
sns.histplot(data=token_data, x='total_length', kde=True)
plt.title('Total Token Length Distribution')
plt.axvline(x=1560, color='red', linestyle='--', label='Max Length (1560)')
plt.legend()

plt.subplot(2, 2, 4)
sns.boxplot(data=pd.melt(token_data, value_vars=['input_length', 'output_length', 'total_length'],
                         var_name='token_type', value_name='length'))
plt.title('Token Length Comparison')
plt.axhline(y=1560, color='red', linestyle='--', label='Max Length (1560)')
plt.legend()

plt.tight_layout()
plt.savefig('results/token_length_distribution.png', dpi=300)
plt.close()

# Calculate how many samples exceed the limit
over_limit = (token_data['total_length'] > 1560).sum()
total_samples = len(token_data)
percentage_over = (over_limit / total_samples) * 100

print(f"\nToken Length Analysis:")
print(f"Total samples: {total_samples}")
print(f"Samples exceeding 1560 tokens: {over_limit} ({percentage_over:.2f}%)")

# Create recommendations based on the analysis
print("\nRecommendations:")
if stats['total']['p95'] > 1560:
    print(f"- Consider increasing max token length beyond 1560 (95th percentile is {stats['total']['p95']:.2f})")
if stats['total']['max'] > 1560 * 2:
    print(f"- Some extreme outliers exist (max={stats['total']['max']}). Consider filtering these samples.")
    
print(f"- For the current limit of 1560 tokens, {total_samples - over_limit} samples ({100 - percentage_over:.2f}%) can be used without truncation.")