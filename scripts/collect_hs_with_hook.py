import torch
import argparse

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

class HiddenStateSaverHook:
    def __init__(self, path_to_save, device, dtype, component_name=""):
        self.path_to_save = path_to_save
        self.hidden_states_list = []
        self.attention_masks = []  # Store attention masks for proper processing
        self.device = device
        self.dtype = dtype
        self.component_name = component_name  # For identification (e.g., "pre_rmsnorm", "post_rmsnorm")

    def __call__(self, module, input, output=None):
        """
        This method is the core of the hook. It's called by PyTorch during the forward pass.
        It handles both pre-hooks (which get 'input') and post-hooks (which get 'output').
        """
        tensor_to_save = input[0] if output is None else output

        if isinstance(tensor_to_save, tuple):
            tensor_to_save = tensor_to_save[0]

        self.hidden_states_list.append(tensor_to_save.detach().cpu())

    def set_attention_mask(self, attention_mask):
        """Call this before each forward pass to store the current attention mask"""
        self.attention_masks.append(attention_mask.cpu())

    def save(self):
        if not self.hidden_states_list:
            print(f"No hidden states collected for {self.component_name}. Please check your model and data.")
            return

        print(f"Processing {len(self.hidden_states_list)} batches of hidden states from {self.component_name}...")
        all_relevant_states = []
        
        # Ensure we have matching attention masks
        if len(self.attention_masks) != len(self.hidden_states_list):
            print(f"Warning: Mismatch between hidden states batches ({len(self.hidden_states_list)}) and attention masks ({len(self.attention_masks)})")
            return
        
        for hidden_states_batch, attention_mask in zip(self.hidden_states_list, self.attention_masks):
            # Process each item in the batch
            for j in range(hidden_states_batch.shape[0]):
                sample_hidden_states = hidden_states_batch[j]
                sample_attention_mask = attention_mask[j]
                
                actual_length = int(sample_attention_mask.sum().item())
                # Handle left padding - actual tokens are at the end
                relevant_hs = sample_hidden_states[-actual_length:, :]
                all_relevant_states.append(relevant_hs)

        if all_relevant_states:
            final_tensor = torch.cat(all_relevant_states, dim=0)
            
            output_dir = os.path.dirname(self.path_to_save)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            print(f"Saving {self.component_name} hidden states to {self.path_to_save}")
            torch.save(final_tensor, self.path_to_save)
            print(f"{self.component_name} hidden states saved successfully. Shape: {final_tensor.shape}")
        else:
            print(f"No relevant hidden states found for {self.component_name}")

    def clear(self):
        """Clear stored states to free memory"""
        self.hidden_states_list.clear()
        self.attention_masks.clear()

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
        config = AutoConfig.from_pretrained(
            args.base_model_name_or_path,
            trust_remote_code=args.trust_remote_code
        )

        config.only_train_language_modeling = True
        config.only_train_contrastive = False

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            config=config,
            torch_dtype=dtype,
            device_map=0,
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

    if args.peft_model_path:
        print(f"Applying PEFT adapter from {args.peft_model_path}")
        model = PeftModel.from_pretrained(
            model,
            args.peft_model_path,
            is_trainable=False 
            )
        if args.merge_before_inference:
            print("Merging PEFT adapter into the base model...")
            model = model.merge_and_unload()
            print("PEFT adapter merged.")

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    print(f"Loading dataset from: {args.test_file}")
    try:
        dataset = load_dataset('json', data_files=args.test_file, split='train')
        if args.text_column not in dataset.column_names:
            if args.text_column == 'text' and len(dataset.column_names) > 0:
             text_column_name = dataset.column_names[0]
             print(f"Warning: Default text column '{args.text_column}' not found. Using the first column '{text_column_name}' for input text.")
             dataset = dataset.rename_column(text_column_name, args.text_column)
            else:
                raise ValueError(f"Could not find the specified text column '{args.text_column}' in the dataset. Available columns: {dataset.column_names}")

    except Exception as e:
        print(f"Error loading dataset with default method: {e}")
        print("Please ensure your --test_file is a valid JSON or JSONL file and the --text_column exists.")
        print("The `datasets` library is used for loading, which supports JSONL directly.")
        return

    max_length = args.max_input_length if args.max_input_length else tokenizer.model_max_length
    if max_length is None:
        print("Warning: max_input_length not specified and tokenizer.model_max_length is None. Using 512 as default.")
        max_length = 512
    print(f"Tokenizing with max_length: {max_length}, padding: max_length, truncation: True")

    def tokenize_function(examples):
        return tokenizer(
            examples[args.text_column], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size)

    # Create hooks for different components
    hooks = {}
    hook_handles = []
    
    # Determine which layers to hook
    
    # Adjust layer access based on whether the model is a PeftModel
    try:
        layers = model.base_model.model.model.layers if isinstance(model, PeftModel) else model.model.layers
    except AttributeError:
        print("Error: Could not access model layers. Please check the model architecture and adjust the layer access path.")
        return

    if args.layer_to_collect is None:
        # Hook all layers
        layer_indices = list(range(len(layers)))
        print(f"Collecting hidden states from all {len(layer_indices)} layers")
    else:
        # Hook single layer
        target_layer_idx = args.layer_to_collect if args.layer_to_collect >= 0 else len(layers) + args.layer_to_collect
        layer_indices = [target_layer_idx]
        print(f"Collecting hidden states from layer {target_layer_idx}")
    
    # Hook into specified module of specified layers
    for layer_idx in layer_indices:
        target_layer = layers[layer_idx]
        
        # Check if the specified module exists in the layer
        if hasattr(target_layer, args.hook_module):
            target_module = getattr(target_layer, args.hook_module)

            # Create and register pre-hook
            pre_hook = HiddenStateSaverHook(f"{args.output_file}/{args.output_file}_layer_{layer_idx}_{args.hook_module}_pre.pt", device, dtype, f"pre_{args.hook_module}_layer_{layer_idx}")
            hooks[pre_hook.component_name] = pre_hook
            hook_handles.append(target_module.register_forward_pre_hook(pre_hook))
            
            # Create and register post-hook
            post_hook = HiddenStateSaverHook(f"{args.output_file}/{args.output_file}_layer_{layer_idx}_{args.hook_module}_post.pt", device, dtype, f"post_{args.hook_module}_layer_{layer_idx}")
            hooks[post_hook.component_name] = post_hook
            hook_handles.append(target_module.register_forward_hook(post_hook))
        else:
            print(f"Warning: Module '{args.hook_module}' not found in layer {layer_idx}. Skipping this layer.")
            available_modules = [name for name, _ in target_layer.named_children()]
            print(f"Available modules in layer {layer_idx}: {available_modules}")

    print("Starting inference and hidden state collection with hooks...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Batches"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            for hook in hooks.values():
                hook.set_attention_mask(attention_mask)

            model(input_ids, attention_mask=attention_mask)

    print("\nFinished inference.")

    # Save all collected hidden states
    for hook_name, hook in hooks.items():
        hook.save()
        hook.clear()  # Free memory

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()

    print("All hidden states saved and hooks removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hidden states from a specified layer of a language model for all input tokens.")

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
        help="Merge PEFT adapter into the base model before inference."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test data file (e.g., JSON file containing text examples).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="prefix",
        help="Name of the column in the JSON file that contains the input text."
    )
    parser.add_argument(
        "--layer_to_collect",
        type=int,
        default=None,
        help="Layer from which to collect hidden states. 0 is embeddings, 1 is the first layer, -1 is the last layer. If None, collects from all layers."
    )
    parser.add_argument(
        "--hook_module",
        type=str,
        default="input_layernorm",
        help="Name of the module within each layer to hook into (e.g., 'input_layernorm', 'post_attention_layernorm', 'mlp', 'self_attn')."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the collected last layer hidden states tensor (.pt file).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=None,
        help="Maximum sequence length for tokenization. Pads or truncates to this length. Defaults to model's max length or 512."
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