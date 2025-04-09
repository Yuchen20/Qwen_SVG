import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from svg_constraint import SVGConstraints
from train_svg_model import PROMPT_TEMPLATE


def generate_svg(
    prompt: str,
    model_path: str,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.9,
    num_samples: int = 1,
) -> list[str]:
    """Generate SVG(s) from a text description."""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Add special tokens
    # special_tokens = {"additional_special_tokens": ["<reasoning>", "</reasoning>"]}
    # tokenizer.add_special_tokens(special_tokens)
    # model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path)
    
    # Format prompt
    full_prompt = PROMPT_TEMPLATE.format(description=prompt)
    
    # Initialize constraints checker
    constraints = SVGConstraints()
    
    # Generate samples
    results = []
    for _ in range(num_samples):
        # Tokenize input
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode output
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Try to validate SVG
        try:
            # Extract SVG content
            import re
            svg_match = re.search(r'<svg.*?</svg>', generated, re.DOTALL)
            if svg_match:
                svg_content = svg_match.group(0)
                # Validate SVG
                constraints.validate_svg(svg_content)
                results.append(generated)
            else:
                print("Warning: No valid SVG found in generation")
        except ValueError as e:
            print(f"Warning: Generated SVG failed validation: {e}")
    
    return results

def save_results(outputs: list[str], output_dir: Path, prompt: str):
    """Save generation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save prompt
    with open(output_dir / "prompt.txt", "w") as f:
        f.write(prompt)
    
    # Save each generation
    for i, output in enumerate(outputs):
        with open(output_dir / f"generation_{i+1}.txt", "w") as f:
            f.write(output)
        
        # Try to extract and save just the SVG
        import re
        svg_match = re.search(r'<svg.*?</svg>', output, re.DOTALL)
        if svg_match:
            with open(output_dir / f"generation_{i+1}.svg", "w") as f:
                f.write(svg_match.group(0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SVGs using trained model")
    parser.add_argument("--prompt", type=str, required=True, help="Text description to generate SVG from")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained LoRA weights")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen1.5-1.5B-Instruct", help="Base model name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generations")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Generate samples
    outputs = generate_svg(
        prompt=args.prompt,
        model_path=args.model_path,
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
    )
    
    # Save results
    save_results(outputs, Path(args.output_dir), args.prompt)