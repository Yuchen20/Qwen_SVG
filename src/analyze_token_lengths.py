import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from train_svg_model import PROMPT_TEMPLATE



def analyze_token_lengths(data_path: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Analyze and visualize the token length distribution of the dataset."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    special_tokens = {"additional_special_tokens": ["<reasoning>", "</reasoning>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load data and calculate lengths
    lengths = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Calculating token lengths"):
            item = json.loads(line.strip())
            prompt = PROMPT_TEMPLATE.format(description=item['description'])
            full_text = f"{prompt}\n\n{item['svg_content']}"
            length = len(tokenizer.encode(full_text))
            lengths.append(length)

    print(
        f"\nThe Prompt Template itself has {len(tokenizer.encode(PROMPT_TEMPLATE))} tokens.\n"
        
    )
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title('Distribution of Token Lengths')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    # Add statistics
    plt.axvline(np.mean(lengths), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(lengths):.1f}')
    plt.axvline(np.percentile(lengths, 95), color='green', linestyle='dashed', linewidth=2, label='95th percentile')
    plt.legend()
    
    # Save plot
    save_path = Path(__file__).parents[1] / "results" / 'token_length_distribution.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Print statistics
    print(f"\nToken Length Statistics:")
    print(f"Mean: {np.mean(lengths):.1f}")
    print(f"Median: {np.median(lengths):.1f}")
    print(f"95th percentile: {np.percentile(lengths, 95):.1f}")
    print(f"Max: {np.max(lengths):.1f}")
    print(f"Min: {np.min(lengths):.1f}")

if __name__ == "__main__":
    data_path = Path(__file__).parents[1] / "data" / "svg_results.jsonl"
    analyze_token_lengths(str(data_path))