import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

def print_model_architecture(model, output_file=None):
    """Print and optionally save the model architecture overview"""
    
    # Capture the model structure
    arch_info = []
    arch_info.append("=" * 80)
    arch_info.append("MODEL ARCHITECTURE OVERVIEW")
    arch_info.append("=" * 80)
    arch_info.append(f"Model type: {type(model).__name__}")
    arch_info.append(f"Model config: {model.config}")
    arch_info.append("")
    
    # Print layer structure
    arch_info.append("LAYER STRUCTURE:")
    arch_info.append("-" * 40)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            arch_info.append(f"{name}: {type(module).__name__}")
    
    arch_info.append("")
    arch_info.append("FULL MODEL STRUCTURE:")
    arch_info.append("-" * 40)
    arch_info.append(str(model))
    
    # Print to console
    for line in arch_info:
        print(line)
    
    # Save to file if specified
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(arch_info))
        print(f"\nArchitecture overview saved to: {output_file}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.torch_dtype:
        try:
            dtype = getattr(torch, args.torch_dtype)
        except AttributeError:
            print(f"Warning: torch_dtype '{args.torch_dtype}' not recognized. Defaulting based on availability.")
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")

    print(f"Loading model: {args.base_model_name_or_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=args.trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path,
            trust_remote_code=args.trust_remote_code
        )
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Please ensure the model name/path is correct and you have internet access or the model is cached.")
        return

    print("\nBASE MODEL ARCHITECTURE:")
    print_model_architecture(model)

    if args.peft_model_path:
        print(f"\nApplying PEFT adapter from {args.peft_model_path}")
        model = PeftModel.from_pretrained(
            model,
            args.peft_model_path,
            is_trainable=False 
        )
        print("\nMODEL WITH PEFT ADAPTER:")
        print_model_architecture(model, args.output_file)
        
        if args.merge_before_inference:
            print("\nMerging PEFT adapter into the base model...")
            model = model.merge_and_unload()
            print("PEFT adapter merged.")
            print("\nMERGED MODEL ARCHITECTURE:")
            merged_output = args.output_file.replace('.txt', '_merged.txt') if args.output_file else None
            print_model_architecture(model, merged_output)
    else:
        print("\nNo PEFT adapter specified. Showing base model only.")
        if args.output_file:
            print_model_architecture(model, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print and save model architecture overview with optional PEFT adapter.")

    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="Path or name of the base language model (e.g., meta-llama/Llama-3.2-3B-Instruct).",
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        default=None,
        help="Path to the PEFT adapter model. If None, uses the base model only."
    )
    parser.add_argument(
        "--merge_before_inference",
        action='store_true',
        help="Also show architecture after merging PEFT adapter into the base model."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the model architecture overview (.txt file). If not specified, only prints to console."
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default=None,
        help="PyTorch dtype for loading the model (e.g., 'float16', 'bfloat16', 'float32')."
    )
    parser.add_argument(
        "--trust_remote_code",
        action='store_true',
        help="Allow trusting remote code for model/tokenizer loading (use with caution)."
    )
    
    args = parser.parse_args()
    main(args)
