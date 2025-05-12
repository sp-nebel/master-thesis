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

        # --- Process Hidden States (All Layers) ---
        # outputs.hidden_states is a tuple (one element per generated token)
        # Each element is another tuple (one element per layer, including embeddings)
        # Each element of the inner tuple is a tensor: (batch_size, sequence_length_at_this_step, hidden_size)

        num_generated_steps = len(outputs.hidden_states)
        batch_size = tokenized_inputs.input_ids.shape[0]
        # Get num_layers and hidden_size from the first step, first layer's state
        # Note: outputs.hidden_states[0] contains states from the input processing step
        # We are interested in the states *during* generation, starting from index 1 if available,
        # but the structure holds. The tuple length gives layer count.
        if num_generated_steps > 0 and len(outputs.hidden_states[0]) > 0:
             num_layers = len(outputs.hidden_states[0]) # Includes embedding layer usually
             hidden_size = outputs.hidden_states[0][0].shape[-1]
        else:
             # Handle case with no generation or unexpected output structure
             print("Warning: Could not determine layer count or hidden size from outputs. Skipping batch.")
             continue


        # List to hold the sequence of hidden states (all layers) for each item in the batch
        # batch_hidden_state_sequences[i] will be a list of tensors, each tensor shape: (num_layers, hidden_size)
        batch_hidden_state_sequences = [[] for _ in range(batch_size)]

        # Iterate through each generation step
        for step in range(num_generated_steps):
            # Get the tuple of hidden states for all layers at this step
            all_layer_states_at_step = outputs.hidden_states[step] # Tuple of tensors [(batch, seq, hidden), ...]

            # List to store the last token's state across all layers for this step
            # Shape will be (batch_size, num_layers, hidden_size)
            last_token_all_layers = []
            for layer_state in all_layer_states_at_step:
                # layer_state shape: (batch_size, sequence_length_at_this_step, hidden_size)
                # Get the state for the last token in the sequence at this step for this layer
                # Shape: (batch_size, hidden_size)
                last_token_state = layer_state[:, -1, :]
                last_token_all_layers.append(last_token_state)

            # Stack the layer states for the last token: Result shape (num_layers, batch_size, hidden_size)
            stacked_last_token_states = torch.stack(last_token_all_layers, dim=0)

            # Permute to get (batch_size, num_layers, hidden_size) and move to CPU
            last_token_states_per_item = stacked_last_token_states.permute(1, 0, 2).cpu()

            # Append the states for this step to each item in the batch
            for i in range(batch_size):
                batch_hidden_state_sequences[i].append(last_token_states_per_item[i]) # Append tensor (num_layers, hidden_size)

        # --- Save Hidden States and Collect Metadata ---
        for i in range(batch_size):
            if not batch_hidden_state_sequences[i]: # Handle cases where no tokens were generated
                print(f"Warning: No hidden states generated for item index {batch_indices[i].item()}. Skipping save.")
                continue

            original_index = batch_indices[i].item() # Get the original index
            # Stack the list of tensors (each shape: num_layers, hidden_size) along a new dimension (dim=0)
            # Final shape: (num_generated_tokens, num_layers, hidden_size)
            hidden_state_sequence_all_layers = torch.stack(batch_hidden_state_sequences[i], dim=0)

            # Define path and save the tensor
            tensor_filename = f"hidden_states_all_layers_{original_index}.pt"
            tensor_path = os.path.join(args.hidden_states_dir, tensor_filename)
            torch.save(hidden_state_sequence_all_layers, tensor_path)

            # Add metadata to the list
            all_results_metadata.append({
                "id": original_index,
                "prompt": batch_prompts[i], # Optionally store the prompt
                "hidden_state_path": tensor_path,
                "num_generated_tokens": hidden_state_sequence_all_layers.shape[0],
                "num_layers": hidden_state_sequence_all_layers.shape[1],
                "hidden_size": hidden_state_sequence_all_layers.shape[2]
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
