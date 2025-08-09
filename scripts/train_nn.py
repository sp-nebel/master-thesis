import os
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class HiddenStateAlignmentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HiddenStateAlignmentNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x


def main(args):
    # Define your dimensions
    input_dim = args.input_dim
    output_dim = args.output_dim
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    model_output_path = args.output_path
    batch_size = args.batch_size
    validation_split_ratio = args.validation_split

    # Initialize the model
    model = HiddenStateAlignmentNet(input_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5
    )

    input_states = torch.load(args.input_data_path)
    target_states = torch.load(args.target_data_path)

    print(f"Loaded {len(input_states)} sequences from input_data_path.")
    print(f"Loaded {len(target_states)} sequences from target_data_path.")

    input_states_all_tokens = input_states.to(torch.float32)
    target_states_all_tokens = target_states.to(torch.float32)

    print(f"Shape of concatenated input_states: {input_states_all_tokens.shape}")
    print(f"Shape of concatenated target_states: {target_states_all_tokens.shape}")

    if input_states_all_tokens.shape[0] != target_states_all_tokens.shape[0]:
        raise ValueError(
            "The total number of tokens in the input and target datasets must match. "
            f"Got {input_states_all_tokens.shape[0]} input tokens and {target_states_all_tokens.shape[0]} target tokens."
        )

    if input_states_all_tokens.shape[1] != input_dim:
        raise ValueError(
            f"The hidden dimension of the input data ({input_states_all_tokens.shape[1]}) "
            f"does not match the expected input_dim ({input_dim})."
        )
    if target_states_all_tokens.shape[1] != output_dim:
        target_states_all_tokens = nn.functional.pad(target_states_all_tokens, (0, output_dim - target_states_all_tokens.shape[1]))
        print(f"Padded target_states_all_tokens")

    # --- Train/Validation Split ---
    full_dataset = TensorDataset(input_states_all_tokens, target_states_all_tokens)
    num_samples = len(full_dataset)
    val_size = int(validation_split_ratio * num_samples)
    train_size = num_samples - val_size

    if val_size == 0 and num_samples > 0 :
        print(f"Warning: Validation split ratio {validation_split_ratio} resulted in 0 validation samples for {num_samples} total samples. Adjusting to use 1 validation sample.")
        if train_size > 1:
            val_size = 1
            train_size = num_samples - val_size
        else:
            print("Not enough samples for a train/validation split. Using all data for training.")
            train_dataset = full_dataset
            val_dataset = None
    elif val_size > 0 :
        train_dataset_raw, val_dataset_raw = random_split(full_dataset, [train_size, val_size])
    else:
        print("No data to train or validate.")
        return

    # --- Data Normalization (Standardization) ---
    train_inputs_tensor = torch.stack([train_dataset_raw[i][0] for i in range(len(train_dataset_raw))])
    train_targets_tensor = torch.stack([train_dataset_raw[i][1] for i in range(len(train_dataset_raw))])

    input_mean = train_inputs_tensor.mean(dim=0, keepdim=True)
    input_std = train_inputs_tensor.std(dim=0, keepdim=True)
    input_std[input_std == 0] = 1e-6 # Avoid division by zero

    target_mean = train_targets_tensor.mean(dim=0, keepdim=True)
    target_std = train_targets_tensor.std(dim=0, keepdim=True)
    target_std[target_std == 0] = 1e-6 # Avoid division by zero

    normalized_train_inputs = (train_inputs_tensor - input_mean) / input_std
    normalized_train_targets = (train_targets_tensor - target_mean) / target_std

    train_dataset = TensorDataset(normalized_train_inputs, normalized_train_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    if 'val_dataset_raw' in locals() and val_dataset_raw and val_size > 0:
        val_inputs_tensor = torch.stack([val_dataset_raw[i][0] for i in range(len(val_dataset_raw))])
        val_targets_tensor = torch.stack([val_dataset_raw[i][1] for i in range(len(val_dataset_raw))])
        normalized_val_inputs = (val_inputs_tensor - input_mean) / input_std
        normalized_val_targets = (val_targets_tensor - target_mean) / target_std
        val_dataset = TensorDataset(normalized_val_inputs, normalized_val_targets)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        val_dataloader = None

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 25
    avg_epoch_loss = 0.0

    # Pre-train validation
    if val_dataloader:
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_input_val, batch_target_val in val_dataloader:
                batch_input_val = batch_input_val.to(device)
                batch_target_val = batch_target_val.to(device)
                outputs_val = model(batch_input_val)
                val_loss = criterion(outputs_val, batch_target_val)
                epoch_val_loss += val_loss.item()
        avg_epoch_val_loss = epoch_val_loss / len(val_dataloader)
        print(f"Pre-train Validation Loss: {avg_epoch_val_loss:.4f}")
        best_val_loss = avg_epoch_val_loss
    else:
        print("No validation data available. Skipping pre-train validation.")


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_input, batch_target in train_dataloader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        
        # Validation phase
        if val_dataloader:
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_input_val, batch_target_val in val_dataloader:
                    batch_input_val = batch_input_val.to(device)
                    batch_target_val = batch_target_val.to(device)
                    outputs_val = model(batch_input_val)
                    val_loss = criterion(outputs_val, batch_target_val)
                    epoch_val_loss += val_loss.item()
            avg_epoch_val_loss = epoch_val_loss / len(val_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}")
            scheduler.step(avg_epoch_val_loss)
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
                break
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}")
            scheduler.step(avg_epoch_loss)

    output_dir = os.path.dirname(model_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(model.state_dict(), model_output_path)
    print(f"Model saved to {model_output_path}")

    stats_path = os.path.join(output_dir, "normalization_stats.pt")
    normalization_stats = {
        'input_mean': input_mean,
        'input_std': input_std,
        'target_mean': target_mean,
        'target_std': target_std
    }
    torch.save(normalization_stats, stats_path)
    print(f"Normalization stats saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network for hidden state alignment."
    )
    # --- Arguments are unchanged ---
    parser.add_argument("--input_dim",type=int,default=2048)
    parser.add_argument("--output_dim",type=int,default=3072)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--learning_rate",type=float,default=0.001)
    parser.add_argument("--output_path",type=str,default="hidden_state_alignment_model.pth")
    parser.add_argument("--input_data_path",type=str,required=True)
    parser.add_argument("--target_data_path",type=str,required=True)
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--validation_split",type=float,default=0.1)
    parser.add_argument("--num_workers",type=int,default=8)
    args = parser.parse_args()
    main(args)