import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import jsonlines
import os

def main(args):
    print(f"Loading base model: {args.base_model_name_or_path}")
    print(f"Loading adapter from: {args.peft_model_path}")
    print(f"Using device: {args.device}")
    print(f"Using dtype: {args.torch_dtype}")


    # --- Determine torch dtype ---
    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype else None

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=True # Add if needed for specific models
    )
    # Set padding token if necessary (LLaMA models often need this)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Config update might be needed for the model if pad_token_id wasn't initially set
        # config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left" # Important for generation

    print(f"Tokenizer pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    print(f"Tokenizer EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"Tokenizer padding side: {tokenizer.padding_side}")

    # --- Load Base Model ---
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        device_map=args.device if args.device else "auto",
        torch_dtype=args.torch_dtype if args.torch_dtype else None,
        trust_remote_code=True
    )

    # --- Load PEFT Adapter ---
    if args.peft_model_path:
        print(f"Applying PEFT adapter from {args.peft_model_path}")
        model = PeftModel.from_pretrained(
            model,
            args.peft_model_path,
            # torch_dtype=dtype, # PeftModel usually inherits dtype or handles it
            is_trainable=False # Ensure model is in eval mode
        )
        model = model.merge_and_unload() if args.merge_before_inference else model
        if args.merge_before_inference:
            print("Merged PEFT adapter into the base model.")
    else:
        print("Warning: No PEFT adapter path provided. Running inference with the base model only.")

    model.eval() # Set model to evaluation mode

    # --- Load Dataset ---
    print(f"Loading dataset from: {args.test_file}")
    # Load the dataset. It will likely return a DatasetDict.
    # Remove the try-except block as it might obscure the structure.
    # Handle potential loading errors more directly if needed.
    dataset_dict = load_dataset("json", data_files=args.test_file)

    # Check if loading resulted in a dictionary of splits
    if not isinstance(dataset_dict, dict) or not dataset_dict:
        raise TypeError(f"Expected load_dataset to return a DatasetDict, but got {type(dataset_dict)}")

    # Assume the first split is the one we want (usually 'train' by default)
    split_name = list(dataset_dict.keys())[0]
    dataset = dataset_dict[split_name]
    print(f"Using dataset split: '{split_name}'")

    print(f"Dataset loaded with {len(dataset)} examples.") # len() should now report row count

    # Prepare prompts from the 'prefix' column
    if 'prefix' in dataset.column_names:
        prompts = dataset['prefix'] # Access the 'prefix' column directly from the Dataset object
        print(f"Extracted {len(prompts)} prompts using direct column access.")
    else:
        print(f"Error: 'prefix' column not found. Available columns: {dataset.column_names}") # Debugging info
        raise KeyError("Dataset does not contain the expected 'prefix' column.")

    print("\n--- Example Prompt ---")
    print(prompts[0])
    print("----------------------\n")

    # --- Create DataLoader ---
    dataloader = DataLoader(prompts, batch_size=args.batch_size)

    # --- Generation Arguments ---
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": args.do_sample,
    }
    if not args.do_sample: # Beam search settings
        generation_kwargs["num_beams"] = args.num_beams
        generation_kwargs["early_stopping"] = True
    else: # Sampling settings
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_k"] = args.top_k
        generation_kwargs["top_p"] = args.top_p

    generation_config = GenerationConfig(stop_strings="</s>", **generation_kwargs)
    print(f"Generation arguments: {generation_kwargs}")

    # --- Inference Loop ---
    all_predictions = []
    print(f"Starting inference with batch size {args.batch_size}...")
    for batch_prompts in tqdm(dataloader, desc="Generating"):
        # Tokenize batch
        tokenized_inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_input_length # Limit input length if needed
        ).to(model.device)

        prompt_lengths = tokenized_inputs.attention_mask.sum(dim=1)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                tokenizer=tokenizer,
                generation_config=generation_config
            )

        # Decode only the newly generated tokens for the entire batch
        # batch_decode returns a list of strings
        decoded_batch_predictions = tokenizer.batch_decode(
            outputs[:, tokenized_inputs.input_ids.shape[1]:],
            skip_special_tokens=True # Add this to remove tokens like </s> from the output
        )

        # Extend the main list with the list of decoded strings from the batch
        all_predictions.extend(decoded_batch_predictions)

    # --- Save Results ---
    print(f"Saving {len(all_predictions)} predictions to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with jsonlines.open(args.output_file, mode='w') as writer:
        # Now 'pred' will be a complete prediction string for each input prompt
        for pred in all_predictions:
            writer.write({"prediction": pred})

    print("Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a PEFT LoRA model.")

    # Model and Tokenizer Arguments
    parser.add_argument("--base_model_name_or_path", type=str, required=True, help="Path or HuggingFace name of the base model.")
    parser.add_argument("--peft_model_path", type=str, default=None, help="Path to the directory containing the PEFT adapter weights (adapter_model.bin/safetensors and adapter_config.json). If None, runs base model.")
    parser.add_argument("--merge_before_inference", action='store_true', help="Merge PEFT adapter into the base model before running inference.")

    # Data Arguments
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset (JSONL file, expects a 'prefix' key).")
    parser.add_argument("--output_file", type=str, default="predictions.jsonl", help="Path to save the generated predictions.")

    # Inference Arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument("--max_input_length", type=int, default=None, help="Maximum input length for tokenizer (to prevent OOM with long prompts). Default: None (uses model default).") # Adjusted default based on block_size=2048
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on (e.g., 'cuda', 'cuda:0', 'cpu').")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float16", help="PyTorch dtype for loading the model (e.g., 'float16', 'bfloat16', 'float32').")

    # Quantization Arguments (Optional)
    parser.add_argument("--load_in_8bit", action='store_true', help="Load model in 8-bit quantization.")
    parser.add_argument("--load_in_4bit", action='store_true', help="Load model in 4-bit quantization.")

    # Generation Strategy Arguments
    parser.add_argument("--do_sample", action='store_true', help="Use sampling instead of beam search.")
    # Beam search specific
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (only used if --do_sample is False).")
    # Sampling specific
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (only used if --do_sample is True).")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling (only used if --do_sample is True).")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling (only used if --do_sample is True).")


    args = parser.parse_args()

    # --- Argument Validation ---
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot load in both 8-bit and 4-bit. Choose one.")
    if not args.do_sample and args.num_beams <= 1:
        print("Warning: Beam search selected (--do_sample=False) but num_beams <= 1. This is equivalent to greedy decoding.")
    if args.do_sample and args.num_beams > 1:
        print("Warning: Sampling selected (--do_sample=True) but num_beams > 1. num_beams will be ignored.")


    main(args)
