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

    config_kwargs = {
        "only_train_language_modeling": True,
    }
    # --- Load Base Model ---
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        device_map=args.device if args.device else "auto",
        torch_dtype=args.torch_dtype if args.torch_dtype else None,
        trust_remote_code=True,
        **config_kwargs
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
    all_results_metadata = [] # To store metadata like paths
    print(f"Starting inference with batch size {args.batch_size}...")
    print(f"Hidden states will be saved in: {args.hidden_states_dir}")
    os.makedirs(args.hidden_states_dir, exist_ok=True) # Create output dir for tensors

    # Keep track of the absolute index using the dataloader
    prompt_indices = list(range(len(prompts)))
    dataloader = DataLoader(list(zip(prompt_indices, prompts)), batch_size=args.batch_size)

    for batch_data in tqdm(dataloader, desc="Generating"):
        batch_indices, batch_prompts = batch_data
        # Tokenize batch
        tokenized_inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_input_length # Limit input length if needed
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                tokenizer=tokenizer,
                generation_config=generation_config,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        # --- Process Hidden States for INPUT TOKENS (All Layers) ---
        batch_size = tokenized_inputs.input_ids.shape[0]

        if not outputs.hidden_states or not outputs.hidden_states[0]:
            print(f"Warning: No hidden states found for input tokens in batch. Skipping batch starting with index {batch_indices[0].item()}.")
            continue

        # outputs.hidden_states[0] is a tuple of tensors from the input prompt processing.
        # Each tensor in this tuple corresponds to a layer and has shape:
        # (batch_size, input_sequence_length, hidden_size)
        prompt_hidden_states_per_layer = outputs.hidden_states[0]

        num_layers = len(prompt_hidden_states_per_layer)
        if num_layers == 0:
            print(f"Warning: No layers found in prompt hidden states for batch starting with index {batch_indices[0].item()}. Skipping batch.")
            continue
        
        hidden_size = prompt_hidden_states_per_layer[0].shape[-1]
        input_sequence_length = prompt_hidden_states_per_layer[0].shape[1] # Actual sequence length after tokenization

        # Stack the tuple of layer-wise hidden states along a new dimension (dim=1)
        # This creates a single tensor: (batch_size, num_layers, input_sequence_length, hidden_size)
        all_layers_prompt_hs = torch.stack(prompt_hidden_states_per_layer, dim=1)

        # Permute to get (batch_size, input_sequence_length, num_layers, hidden_size)
        # This makes it easier to slice per batch item and matches the desired save format (seq_len, num_layers, hidden_size)
        all_layers_prompt_hs_permuted = all_layers_prompt_hs.permute(0, 2, 1, 3).cpu()

        # --- Save Hidden States and Collect Metadata ---
        for i in range(batch_size):
            original_index = batch_indices[i].item()

            # Get the hidden states for the i-th item in the batch
            # Shape: (input_sequence_length, num_layers, hidden_size)
            input_token_hidden_states = all_layers_prompt_hs_permuted[i]

            # Define path and save the tensor
            # Changed filename to reflect input tokens
            tensor_filename = f"hidden_states_input_tokens_all_layers_{original_index}.pt"
            tensor_path = os.path.join(args.hidden_states_dir, tensor_filename)
            torch.save(input_token_hidden_states, tensor_path)

            # Add metadata to the list
            all_results_metadata.append({
                "id": original_index,
                "prompt": batch_prompts[i], # Optionally store the prompt
                "hidden_state_path": tensor_path,
                "num_input_tokens": input_token_hidden_states.shape[0], # Changed metadata key
                "num_layers": input_token_hidden_states.shape[1],
                "hidden_size": input_token_hidden_states.shape[2]
            })


    # --- Save Results Metadata ---
    print(f"Saving metadata for {len(all_results_metadata)} items to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True) # Ensure metadata output dir exists
    with jsonlines.open(args.output_file, mode='w') as writer:
        for metadata_item in all_results_metadata:
             writer.write(metadata_item)


    print("Inference complete. Hidden states (all layers) saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a PEFT LoRA model and save hidden states.")

    # Model and Tokenizer Arguments
    parser.add_argument("--base_model_name_or_path", type=str, required=True, help="Path or HuggingFace name of the base model.")
    parser.add_argument("--peft_model_path", type=str, default=None, help="Path to the directory containing the PEFT adapter weights (adapter_model.bin/safetensors and adapter_config.json). If None, runs base model.")
    parser.add_argument("--merge_before_inference", action='store_true', help="Merge PEFT adapter into the base model before running inference.")

    # Data Arguments
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset (JSONL file, expects a 'prefix' key).")
    parser.add_argument("--output_file", type=str, default="inference_metadata.jsonl", help="Path to save the metadata (e.g., paths to hidden states).")
    parser.add_argument("--hidden_states_dir", type=str, default="hidden_states_output", help="Directory to save the hidden state tensors.")

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
