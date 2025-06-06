import torch
import os
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def main(args):
    print(f"Loading base model: {args.base_model_name_or_path}")
    if args.peft_model_path:
        print(f"Loading adapter from: {args.peft_model_path}")
    print(f"Using device: {args.device}")
    print(f"Using dtype: {args.torch_dtype}")

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "left"

    # --- Load Base Model ---
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        device_map=args.device if args.device else "auto",
        torch_dtype=getattr(torch, args.torch_dtype) if args.torch_dtype else None,
        trust_remote_code=True,
    )

    # --- Load PEFT Adapter ---
    if args.peft_model_path:
        model = PeftModel.from_pretrained(model, args.peft_model_path)
        model = model.merge_and_unload()

    model.eval()

    # --- Load Dataset ---
    dataset_dict = load_dataset("json", data_files=args.test_file)
    split_name = list(dataset_dict.keys())[0]
    dataset = dataset_dict[split_name]
    
    if 'prefix' in dataset.column_names:
        prompts = [item['prefix'] for item in dataset]
    else:
        prompts = [item['input'] for item in dataset]

    print(f"Loaded {len(prompts)} prompts")
    print(f"Example prompt: {prompts[0][:100]}...")

    # --- Create output directory ---
    os.makedirs(args.hidden_states_dir, exist_ok=True)
    
    # --- Generation with hidden state extraction ---
    all_results = []
    
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating with hidden states")):
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        
        prompt_length = input_ids.shape[1]
        
        # Generate tokens one by one to capture hidden states
        generated_ids = input_ids.clone()
        generated_hidden_states = []  # List to store hidden states for each generated token
        
        with torch.no_grad():
            for step in range(args.max_new_tokens):
                # Forward pass with output_hidden_states=True
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                
                # Get next token logits and sample
                next_token_logits = outputs.logits[:, -1, :]
                
                if args.do_sample:
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / args.temperature, dim=-1), 
                        num_samples=1
                    )
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Extract hidden states for the last position (newly generated token)
                # outputs.hidden_states is a tuple of tensors, one for each layer
                # Each tensor has shape [batch_size, seq_len, hidden_size]
                last_position_hidden_states = []
                for layer_idx, layer_hidden_states in enumerate(outputs.hidden_states):
                    # Get hidden state for the last position (the newly generated token)
                    last_pos_hidden = layer_hidden_states[:, -1, :]  # [batch_size, hidden_size]
                    last_position_hidden_states.append(last_pos_hidden)
                
                # Stack all layers: [num_layers, batch_size, hidden_size]
                stacked_hidden_states = torch.stack(last_position_hidden_states, dim=0)
                
                # Remove batch dimension (assuming batch_size=1): [num_layers, hidden_size]
                token_hidden_states = stacked_hidden_states.squeeze(1)
                
                generated_hidden_states.append(token_hidden_states.cpu())
                
                # Append the new token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                ], dim=-1)
                
                # Stop if EOS token is generated
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Save hidden states for all generated tokens
        if generated_hidden_states:
            # Stack all generated token hidden states: [num_generated_tokens, num_layers, hidden_size]
            all_gen_hidden_states = torch.stack(generated_hidden_states, dim=0)
            
            # Save to file
            hidden_states_filename = f"generated_hidden_states_{idx}.pt"
            hidden_states_path = os.path.join(args.hidden_states_dir, hidden_states_filename)
            torch.save(all_gen_hidden_states, hidden_states_path)
            
            # Decode generated text
            generated_text = tokenizer.decode(
                generated_ids[0][prompt_length:], 
                skip_special_tokens=True
            )
            
            # Save metadata
            result = {
                "prompt_index": idx,
                "prompt": prompt,
                "generated_text": generated_text,
                "hidden_states_file": hidden_states_filename,
                "num_generated_tokens": len(generated_hidden_states),
                "num_layers": all_gen_hidden_states.shape[1],
                "hidden_size": all_gen_hidden_states.shape[2]
            }
            all_results.append(result)
            
            print(f"Saved hidden states for {len(generated_hidden_states)} tokens: {hidden_states_filename}")
        else:
            print(f"No tokens generated for prompt {idx}")
    
    # Save metadata
    metadata_file = os.path.join(args.hidden_states_dir, "generation_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved metadata to {metadata_file}")
    print(f"Total examples processed: {len(all_results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_model_path", type=str, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--hidden_states_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)