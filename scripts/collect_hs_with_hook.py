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
        self.batch_counter = 0
        self.accumulated_states_for_saving = []

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

    def process_batch(self):
        """Processes the last collected batch of hidden states and adds them to an accumulation list for saving."""
        if not self.hidden_states_list or not self.attention_masks:
            return

        hidden_states_batch = self.hidden_states_list.pop(0)
        attention_mask = self.attention_masks.pop(0)

        all_relevant_states = []
        for j in range(hidden_states_batch.shape[0]):
            sample_hidden_states = hidden_states_batch[j]
            sample_attention_mask = attention_mask[j]
            
            actual_length = int(sample_attention_mask.sum().item())
            relevant_hs = sample_hidden_states[-actual_length:, :]
            all_relevant_states.append(relevant_hs)
        
        if all_relevant_states:
            self.accumulated_states_for_saving.append(torch.cat(all_relevant_states, dim=0))

    def save_accumulated_states(self, force_save=False):
        """Saves the accumulated hidden states to disk."""
        if not self.accumulated_states_for_saving:
            return

        final_tensor = torch.cat(self.accumulated_states_for_saving, dim=0)
        
        output_dir = os.path.dirname(self.path_to_save)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Append to file if it exists, otherwise create it
        if os.path.exists(self.path_to_save):
            existing_tensor = torch.load(self.path_to_save)
            final_tensor = torch.cat([existing_tensor, final_tensor], dim=0)
        
        torch.save(final_tensor, self.path_to_save)
        self.accumulated_states_for_saving.clear() # Clear after saving

    def save_and_clear_batch(self):
        """Saves the last collected batch of hidden states and clears the lists."""
        if not self.hidden_states_list or not self.attention_masks:
            # This can happen if a batch is processed but no states were collected for this specific hook
            return

        hidden_states_batch = self.hidden_states_list.pop(0)
        attention_mask = self.attention_masks.pop(0)

        all_relevant_states = []
        for j in range(hidden_states_batch.shape[0]):
            sample_hidden_states = hidden_states_batch[j]
            sample_attention_mask = attention_mask[j]
            
            actual_length = int(sample_attention_mask.sum().item())
            relevant_hs = sample_hidden_states[-actual_length:, :]
            all_relevant_states.append(relevant_hs)

        if all_relevant_states:
            final_tensor = torch.cat(all_relevant_states, dim=0)
            
            output_dir = os.path.dirname(self.path_to_save)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Append to file if it exists, otherwise create it
            if os.path.exists(self.path_to_save):
                existing_tensor = torch.load(self.path_to_save)
                final_tensor = torch.cat([existing_tensor, final_tensor], dim=0)
            
            torch.save(final_tensor, self.path_to_save)

    def save(self):
        if self.accumulated_states_for_saving:
            self.save_accumulated_states(force_save=True)

        if os.path.exists(self.path_to_save):
            final_tensor = torch.load(self.path_to_save)
            print(f"Saving {self.component_name} hidden states to {self.path_to_save}")
            print(f"{self.component_name} hidden states saved successfully. Shape: {final_tensor.shape}")
        else:
            print(f"No relevant hidden states found for {self.component_name}")

    def clear(self):
        """Clear stored states to free memory"""
        self.hidden_states_list.clear()
        self.attention_masks.clear()
        self.accumulated_states_for_saving.clear()

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
        
        # Check if the specified module exists in the layer by traversing dot-separated path
        target_module = target_layer
        module_found = True
        try:
            for part in args.hook_module.split('.'):
                target_module = getattr(target_module, part)
        except AttributeError:
            module_found = False

        if module_found:
            if args.hook_type in ['pre', 'both']:
                # Create and register pre-hook
                pre_hook = HiddenStateSaverHook(f"{args.output_file}/{args.output_file}_layer_{layer_idx}_{args.hook_module.replace('.', '_')}_pre.pt", device, dtype, f"pre_{args.hook_module}_layer_{layer_idx}")
                hooks[pre_hook.component_name] = pre_hook
                hook_handles.append(target_module.register_forward_pre_hook(pre_hook))
            
            if args.hook_type in ['post', 'both']:
                # Create and register post-hook
                post_hook = HiddenStateSaverHook(f"{args.output_file}/{args.output_file}_layer_{layer_idx}_{args.hook_module.replace('.', '_')}_post.pt", device, dtype, f"post_{args.hook_module}_layer_{layer_idx}")
                hooks[post_hook.component_name] = post_hook
                hook_handles.append(target_module.register_forward_hook(post_hook))
        else:
            print(f"Warning: Module '{args.hook_module}' not found in layer {layer_idx}. Skipping this layer.")
            available_modules = [name for name, _ in target_layer.named_children()]
            print(f"Available top-level modules in layer {layer_idx}: {available_modules}")

    print("Starting inference and hidden state collection with hooks...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            for hook in hooks.values():
                hook.set_attention_mask(attention_mask)

            model(input_ids, attention_mask=attention_mask)

            # Process the collected states for this batch
            for hook in hooks.values():
                hook.process_batch()

            # Save accumulated states at specified interval
            if (i + 1) % args.save_batch_interval == 0:
                for hook in hooks.values():
                    hook.save_accumulated_states()

    print("\nFinished inference.")

    # Save any remaining collected hidden states
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
        "--hook_type",
        type=str,
        default="both",
        choices=['pre', 'post', 'both'],
        help="Type of hook to register: 'pre' for pre-forward, 'post' for post-forward, or 'both'."
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
        "--save_batch_interval",
        type=int,
        default=10,
        help="How many batches to process before saving hidden states to disk."
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