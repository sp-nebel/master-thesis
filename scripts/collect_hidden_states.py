import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

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

    all_relevant_hidden_states_list = []

    print("Starting inference and hidden state collection...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"Processing batch {i+1}/{len(dataloader)}", end='\r')
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states_batch = outputs.hidden_states[args.layer_to_collect]

            # Process each item in the batch to extract relevant hidden states
            for j in range(hidden_states_batch.shape[0]):
                sample_hidden_states = hidden_states_batch[j]  # Shape: (max_length, hidden_size)
                sample_attention_mask = attention_mask[j]              # Shape: (max_length)
                
                actual_length = int(sample_attention_mask.sum().item())
                
                # Since padding_side is "left", the actual tokens are at the end.
                relevant_hs = sample_hidden_states[-actual_length:, :] # Shape: (actual_length, hidden_size)
                all_relevant_hidden_states_list.append(relevant_hs.cpu())

    print("\nFinished inference.")

    if all_relevant_hidden_states_list:
        print("Concatenating all hidden state tensors...")
        final_tensor = torch.cat(all_relevant_hidden_states_list, dim=0)

        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)
        print(f"Saving concatenated tensor with shape {final_tensor.shape} to: {args.output_file}")
        torch.save(final_tensor, args.output_file)
        print("Done.")
    else:
        print("No hidden states were collected. Please check your data and model.")


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
        default=-1,
        help="Layer from which to collect hidden states. 0 is embeddings, 1 is the first layer, -1 is the last layer."
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