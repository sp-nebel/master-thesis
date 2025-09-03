import argparse
from traceback import print_tb
from typing import Optional, Tuple
import concurrent.futures
from regex import F
from safetensors import safe_open
import torch
from torch import device
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import jsonlines
import os
import matplotlib.pyplot as plt
import numpy as np

from transformers.cache_utils import Cache


class LoraHook:
    def __init__(self, lora_A, lora_B, scaling, proc_down, proc_up):
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.scaling = scaling
        self.proc_down = proc_down
        self.proc_up = proc_up

    def __call__(self, module, inputs, output):
        hidden_states = inputs[0]  # Shape: (batch, seq_len, 3072)
        #print(f"Inside LoraHook for {module}, hidden_states shape: {hidden_states.shape}")

        #print("Proc down: \n", self.proc_down)

        hidden_states = hidden_states @ self.proc_down.to(hidden_states.device, hidden_states.dtype)
        #print("Hidden states after proc down: \n", hidden_states)
        hidden_states = hidden_states[..., :2048]
        #print("Hidden states after truncation to 2048: \n", hidden_states)


       # print(f"Hidden states shape before LoRA: {hidden_states.shape}")
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        #print(f"Hidden states flat shape: {hidden_states_flat.shape}")

        lora_A = self.lora_A.to(hidden_states.device, hidden_states.dtype) 
        lora_B = self.lora_B.to(hidden_states.device, hidden_states.dtype)
        
        lora_update_flat = (hidden_states_flat @ lora_A.T @ lora_B.T) * self.scaling
        lora_update = lora_update_flat.view(batch_size, seq_len, -1)
        
        if self.proc_up is not None:
            if lora_update.shape[-1] < self.proc_up.shape[0]:
                #print(f"Padding lora_update from {lora_update.shape[-1]} to {self.proc_up.shape[0]}")
                lora_update = nn.functional.pad(lora_update, (0, self.proc_up.shape[0] - lora_update.shape[-1]))
            lora_update = lora_update @ self.proc_up.to(lora_update.device, lora_update.dtype)
        return output + lora_update
    
class LogitLensHook:
    def __init__(self, activations_list, layer_idx) -> None:
        self.activations_list = activations_list
        self.layer_idx = layer_idx

    def __call__(self, module, inputs, output):
        self.activations_list[self.layer_idx] = output[0].detach()

def load_mapping_pair(layer_idx, directory, device, dtype):
    """
    Loads and processes a single pair of down/up mapping tensors for a given layer.
    This function will be executed by each worker thread.
    """
    down_path = os.path.join(directory, f"3B_layer_{layer_idx}_down.pt")
    up_path = os.path.join(directory, f"3B_layer_{layer_idx}_up.pt")

    down_mapping = torch.load(down_path, map_location=device)
    up_mapping = torch.load(up_path, map_location=device)

    down_mapping = down_mapping.to(dtype)
    up_mapping = up_mapping.to(dtype)
    
    # Return the key and the value to reconstruct the dictionary
    return layer_idx, (down_mapping, up_mapping)

def print_output_attn_tokens(attention_matrix, input_tokens, output_tokens, layer_idx, head_idx, output_dir=None, prompt_idx=0):

    print(f"Printing attention for layer {layer_idx}, head {head_idx} --------------------------------------")
    print(f"attention_matrix shape: {attention_matrix.shape}")
    argmax_attention_per_output = np.argmax(attention_matrix, axis=1)
    filtered_input_tokens = [input_tokens[i] for i in argmax_attention_per_output]
    
    print(f"Argmax input tokens for head {head_idx}: {filtered_input_tokens}")


def visualize_attention(attention_matrix, input_tokens, output_tokens, layer_idx, head_idx, output_dir=None, prompt_idx=0):

    ax_img = plt.imshow(attention_matrix, cmap='hot', interpolation='nearest')
    # Add colorbar
    plt.colorbar(ax_img)

    # Set ticks and labels
    ax_img.axes.set_xticks(np.arange(len(input_tokens)))
    ax_img.axes.set_yticks(np.arange(len(output_tokens)))
    ax_img.axes.set_xticklabels(input_tokens)
    ax_img.axes.set_yticklabels(output_tokens)

    # Use top x-axis
    ax_img.axes.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax_img.axes.get_xticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    ax_img.axes.set_title(f"Attention Heatmap (Layer {layer_idx}, Head {head_idx})")
    ax_img.axes.set_xlabel("Key (Input) Tokens")
    ax_img.axes.set_ylabel("Query (Output) Tokens")
    
    fig = ax_img.get_figure()
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"prompt{prompt_idx}_layer{layer_idx}_head{head_idx}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        print(f"Saved attention visualization to {filepath}")
        plt.close(fig) # Close the figure to free memory
    else:
        plt.show()


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
        output_attentions=True,
        attn_implementation="eager",
        trust_remote_code=True
    )

    layers_to_graft = []

    if args.graft_layers is None:
            layers_to_graft = []
    elif args.graft_layers == [-1]:
        layers_to_graft = list(range(28))
    else:
        layers_to_graft = args.graft_layers

    if args.peft_model_path:
        print(f"Loading PEFT model from {args.peft_model_path}")
        model = PeftModel.from_pretrained(model, args.peft_model_path)
        target_modules_to_modify = list(args.module_map_dict.keys()) if args.module_map_dict else []


        for module in target_modules_to_modify:
            print(f"Disabling original LoRA for {module} in layers: {layers_to_graft}")
            for layer_idx in layers_to_graft:
                try:
                    # Navigate to the parent of the target module
                    parent_module = model.model.model.layers[layer_idx]
                    module_path = module.split('.')
                    module_name = module_path[-1]
                    for part in module_path[:-1]:
                        parent_module = getattr(parent_module, part)

                    target_module = getattr(parent_module, module_name)
                    if hasattr(target_module, 'base_layer'):
                        setattr(parent_module, module_name, target_module.base_layer)
                        print(f"Set {module_name} of {parent_module} in layer {layer_idx} to {target_module.base_layer}.")
                    else:
                        print(f"  - {module_name} in layer {layer_idx} is not a PEFT layer, skipping.")
                except (AttributeError, IndexError) as e:
                    print(f"Warning: Could not access or modify {module} for layer {layer_idx}. Error: {e}")

        if args.merge_before_inference:
            print("Merging PEFT adapter into base model...")
            model = model.merge_and_unload()
            print("Merge complete.")


    lora_state_dict = {}
    # --- Load graft lora ---
    if args.graft_lora_path:

        print(f"Loading Graft LoRA from {args.graft_lora_path}")
        with safe_open(os.path.join(args.graft_lora_path, "adapter_model.safetensors"), framework="pt", device=args.device) as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key) 
                print(f"Loaded {key} with shape {lora_state_dict[key].shape} and dtype {lora_state_dict[key].dtype}")
        print("LoRA weights loaded successfully!")

    mappings = {i: {} for i in layers_to_graft}

    # Load mappings for each module using its specific directory
    for module, mapping_dir in args.module_map_dict.items():
        print(f"Loading mappings for module '{module}' from '{mapping_dir}'...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_layer = {
                executor.submit(load_mapping_pair, i, mapping_dir, args.device, dtype): i
                for i in layers_to_graft
            }
            
            for future in concurrent.futures.as_completed(future_to_layer):
                layer_idx, data = future.result()
                print(f"Loaded mappings for layer {layer_idx} of module '{module}'")
                mappings[layer_idx][module] = data

    # --- Print Model Architecture ---
    print("\n--- Model Architecture ---")
    print(model)
    print("--------------------------\n")


    for i in layers_to_graft:
        for module in args.module_map_dict.keys():
            print(f"Applying graft hook to layer {i} on module {module}...")

            lora_A_ref = lora_state_dict[f'base_model.model.model.layers.0.{module}.lora_A.weight']
            lora_B_ref = lora_state_dict[f'base_model.model.model.layers.0.{module}.lora_B.weight']
            
            for j in range(15):
                a_equal = torch.equal(lora_A_ref, lora_state_dict[f'base_model.model.model.layers.{j}.{module}.lora_A.weight'])
                b_equal = torch.equal(lora_B_ref, lora_state_dict[f'base_model.model.model.layers.{j}.{module}.lora_B.weight'])

                if not a_equal or not b_equal:
                    print(f"Warning: LoRA weights for layer {j}, module {module} do not match the reference layer 0 weights.")

            hook = LoraHook(
                lora_A=lora_state_dict[f'base_model.model.model.layers.0.{module}.lora_A.weight'],
                lora_B=lora_state_dict[f'base_model.model.model.layers.0.{module}.lora_B.weight'],
                scaling=2,
                proc_down=mappings[i][module][0],
                proc_up=mappings[i][module][1]
            )

            # Get the target layer using the index i
            if isinstance(model, PeftModel):
                target_layer = model.model.model.layers[i]
            else:
                target_layer = model.model.layers[i]
            
            # Navigate to the target module and register the hook
            module_to_hook = target_layer
            for part in module.split('.'):
                module_to_hook = getattr(module_to_hook, part)
            module_to_hook.register_forward_hook(hook)
            
            print(f"Successfully registered hook for layer {target_layer} on {module_to_hook}")

    model.eval() # Set model to evaluation mode

    # --- Load Dataset ---
    print(f"Loading dataset from: {args.test_file}")
    # Load the dataset. It will likely return a DatasetDict.
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

    # --- Visualization Mode ---
    if args.visualize_attention_layers:
        print("\n--- ATTENTION VISUALIZATION MODE ---")
        for prompt_idx in args.visualize_prompt_indices:
            if prompt_idx >= len(prompts):
                print(f"Warning: Skipping prompt index {prompt_idx} as it is out of bounds for dataset with {len(prompts)} examples.")
                continue
            
            prompt = prompts[prompt_idx]
            print(f"Visualizing attention for prompt {prompt_idx}:\n{prompt}")

            tokenized_inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            input_ids = tokenized_inputs.input_ids
            
            # Generate with attention output
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    seed=[42, 62],
                    output_attentions=True,
                    return_dict_in_generate=True
                )

            # Extract attentions
            # Shape: (num_generated_tokens, num_layers, batch_size, num_heads, query_len, key_len)
            # We only have batch_size=1 and query_len=1 for each step.
            attentions = outputs.attentions
            print(f"attentions length: {len(attentions)}")

            # Get tokens for labels - they are the same for all layers and heads
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            output_sequence = outputs.sequences[0]
            output_tokens = tokenizer.convert_ids_to_tokens(output_sequence[input_ids.shape[1]:])

            num_layers = len(attentions[0])
            for layer_idx in args.visualize_attention_layers:
                if layer_idx >= num_layers:
                    print(f"Warning: Skipping layer index {layer_idx} as it is out of bounds. Model has {num_layers} layers.")
                    continue

                # For each generation step, get the attention for the specified layer.
                # attentions[step][layer_idx] has shape [batch_size, num_heads, query_len, key_len]
                # We have batch_size=1. For generation steps, query_len=1.
                layer_attentions = [step_att[layer_idx].squeeze(0) for step_att in attentions]

                num_heads = layer_attentions[0].shape[0]
                heads_to_visualize = range(num_heads) if args.visualize_attention_head == -1 else [args.visualize_attention_head]
                
                if args.visualize_attention_head != -1 and args.visualize_attention_head >= num_heads:
                    raise ValueError(f"visualize_attention_head index {args.visualize_attention_head} is out of bounds. Model has {num_heads} heads.")

                for head_idx in heads_to_visualize:
                    # For each step, get the attention for the current head.
                    # This will be shape [query_len, key_len].
                    head_attentions_per_step = [step_head_att[head_idx] for step_head_att in layer_attentions]

                    if args.visualization_print_argmax_only:
                        head_attentions_per_step.pop()
                        attention_to_input_list = [att_step[-1, :input_ids.shape[1]] for att_step in head_attentions_per_step]
                        attention_to_input_list = [t.float() for t in attention_to_input_list]
                        attention_to_input = torch.stack(attention_to_input_list, dim=0).cpu().numpy()

                        print_output_attn_tokens(
                            attention_matrix=attention_to_input,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            output_dir=args.visualization_output_dir,
                            prompt_idx=prompt_idx
                        )
                        continue 
                    
                    else:
                        
                        input_to_input_attentions = head_attentions_per_step[0]

                        visualize_attention(
                            attention_matrix=input_to_input_attentions.cpu().numpy(),
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            output_dir=args.visualization_output_dir,
                            prompt_idx=prompt_idx
                        )

        print("Visualization complete. Exiting.")
        return # Exit after visualization

    # --- Logit lens mode ---
    if args.logit_lens:
        activations = {}

        enum = enumerate(model.model.model.layers) if isinstance(model, PeftModel) else enumerate(model.model.layers)
        
        for i, layer in enum:
            layer.register_forward_hook(LogitLensHook(activations, i))

        prompts = prompts[:5] # Limit to first 5 prompts for testing

        tokenized_inputs = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_input_length # Limit input length if needed
        ).to(model.device)

        fwd_pass = model(tokenized_inputs.input_ids, do_sample=True)

        for layer_idx in range(len(activations)):
            probabilities = torch.nn.functional.softmax(model.lm_head(model.model.norm(activations[layer_idx][0,-1,:])),dim=0)
            max_index = torch.argmax(probabilities)
            top_token = tokenizer.decode(max_index)
            print(f"Layer {layer_idx} prediction: {top_token}")
        
        return # Exit after logit lens

    # --- Create DataLoader ---
    dataloader = DataLoader(prompts, batch_size=args.batch_size)

    # --- Generation Arguments ---
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": args.do_sample
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

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                seed=[42, 62],
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
    parser.add_argument("--graft_lora_path", type=str, default=None, help="Path to the directory containing the Graft LoRA weights.")
    parser.add_argument("--module_mappings", nargs='+', type=str, help="Mapping of target module to its mapping directory, separated by a colon. Example: 'self_attn.q_proj:/path/to/q_maps' 'self_attn.v_proj:/path/to/v_maps'")
    parser.add_argument("--graft_layers", nargs='+', type=int, default=None, help="List of layer indices to apply Graft LoRA to. If None, applies to None, if == -1 applies to all.")
    
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

    # Visualization Arguments
    parser.add_argument("--visualize_attention_layers", nargs='+', type=int, default=None, help="If set, visualizes attention for the specified layer indices and exits. Requires output_attentions=True for the model.")
    parser.add_argument("--visualize_attention_head", type=int, default=-1, help="Head index to visualize for attention map. Default: -1 (all heads).")
    parser.add_argument("--visualize_prompt_indices", nargs='+', type=int, default=[0], help="The indices of the prompts in the test file to use for visualization.")
    parser.add_argument("--visualization_output_dir", type=str, default=None, help="Directory to save attention visualization plots. If None, plots are shown interactively.")
    parser.add_argument("--visualization_print_argmax_only", action='store_true', help="If set, only prints the argmax attention weights.")

    parser.add_argument("--logit_lens", action='store_true', help="If set, performs logit lens analysis and exits.")

    # Generation Strategy Arguments
    parser.add_argument("--do_sample", action='store_true', help="Use sampling instead of beam search.")
    # Beam search specific
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (only used if --do_sample is False).")
    # Sampling specific
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (only used if --do_sample is True).")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling (only used if --do_sample is True).")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling (only used if --do_sample is True).")


    args = parser.parse_args()

    # --- Argument Validation and Processing ---
    if args.module_mappings:
        try:
            args.module_map_dict = {item.split(':', 1)[0]: item.split(':', 1)[1] for item in args.module_mappings}
        except IndexError:
            raise ValueError("Invalid format for --module_mappings. Expected 'module_name:/path/to/directory'.")
    else:
        args.module_map_dict = {}

    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot load in both 8-bit and 4-bit. Choose one.")

    if not args.do_sample and args.num_beams <= 1:
        print("Warning: Beam search selected (--do_sample=False) but num_beams <= 1. This is equivalent to greedy decoding.")
    if args.do_sample and args.num_beams > 1:
        print("Warning: Sampling selected (--do_sample=True) but num_beams > 1. num_beams will be ignored.")

    main(args)
