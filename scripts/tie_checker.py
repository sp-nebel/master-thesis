import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_tied_lora_weights_from_disk(
    base_model_name_or_path: str,
    adapter_path: str,
    lora_target_modules: list # e.g., ["q_proj", "v_proj"]
    ):
    """
    Loads a base model and a LoRA adapter from disk, then verifies
    if the LoRA weights for specified target modules are tied across layers.
    """
    logger.info(f"Loading base model: {base_model_name_or_path}")
    # Load the base model
    # Add any specific model loading arguments if needed (e.g., torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16, # Or your preferred dtype
        device_map="auto" # Or specific device
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    # Load the LoRA model
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        # model = model.merge_and_unload() # Optional: if you want to check merged weights.
                                         # For checking tied *adapter* weights, comment this out
                                         # and ensure the PeftModel structure is traversed correctly below.
                                         # If you keep this, the check below might not be meaningful for *tied LoRA layers*
                                         # as merging creates new weights.
                                         # For verifying tied LoRA weights, it's best to check *before* merging.
                                         # So, for the primary goal, let's assume we don't merge yet.
        # If you *don't* merge_and_unload(), the LoRA layers are still separate.
        # If you *do* merge_and_unload(), the LoRA weights are merged into the base model,
        # and the LoRA-specific layers (like lora_A, lora_B) are gone.
        # The original request implies checking the *adapter* weights themselves.
        # Let's proceed assuming the adapter is loaded but not yet merged.
    except Exception as e:
        logger.error(f"Error loading PeftModel: {e}")
        logger.error("Make sure the adapter_path is correct and contains 'adapter_model.bin' and 'adapter_config.json'.")
        return

    logger.info("Model and adapter loaded successfully.")
    logger.info("Verifying tied LoRA weights...")

    # --- IMPORTANT: Adjust this path according to your model architecture ---
    # This path is common for Llama-like models.
    # For other models, you might need to inspect `model.named_modules()`
    # or `print(model)` to find the correct path to the transformer layers.
    try:
        # If PeftModel wraps the original model, layers might be under model.base_model.model.model.layers
        # or similar, depending on how PeftModel is structured and the original model type.
        # Let's try to access it as it was in your training script:
        llama_layers = model.base_model.model.model.layers
        logger.info(f"Accessing layers via model.base_model.model.model.layers")
    except AttributeError:
        try:
            # Fallback for some architectures or if merge_and_unload was called (though not ideal for this check)
            llama_layers = model.model.layers
            logger.info(f"Accessing layers via model.model.layers")
        except AttributeError:
            logger.error("Could not access model layers. Adjust the path `llama_layers = ...` for your model architecture.")
            logger.info("Try inspecting `model.named_modules()` to find the correct path to the transformer blocks/layers.")
            return

    if not llama_layers:
        logger.warning("No layers found to verify.")
        return

    all_checks_passed = True
    for target_name in lora_target_modules:
        logger.info(f"  Checking target module: {target_name}")

        ref_lora_A_weight_id = None
        ref_lora_B_weight_id = None
        found_in_first_layer = False

        # Get reference ID from the first layer
        if len(llama_layers) > 0:
            for submodule_name, submodule_obj in llama_layers[0].named_modules():
                # Check if the submodule_name ends with the target_name
                # and if it has lora_A and lora_B attributes
                if submodule_name.split('.')[-1] == target_name and \
                   hasattr(submodule_obj, 'lora_A') and hasattr(submodule_obj, 'lora_B') and \
                   isinstance(submodule_obj.lora_A, torch.nn.ModuleDict) and 'default' in submodule_obj.lora_A and \
                   isinstance(submodule_obj.lora_B, torch.nn.ModuleDict) and 'default' in submodule_obj.lora_B:
                    
                    ref_lora_A_weight_id = id(submodule_obj.lora_A['default'].weight.data)
                    ref_lora_B_weight_id = id(submodule_obj.lora_B['default'].weight.data)
                    logger.info(f"    Layer 0 ({submodule_name}): LoRA A ID: {ref_lora_A_weight_id}, LoRA B ID: {ref_lora_B_weight_id}")
                    found_in_first_layer = True
                    break # Found the target module in the first layer
        
        if not found_in_first_layer:
            logger.warning(f"    Target module '{target_name}' with LoRA A/B weights not found in the first layer. Skipping this module.")
            continue

        # Check subsequent layers
        for i, current_llama_layer in enumerate(llama_layers[1:], start=1):
            found_in_current_layer = False
            for submodule_name, submodule_obj in current_llama_layer.named_modules():
                if submodule_name.split('.')[-1] == target_name and \
                   hasattr(submodule_obj, 'lora_A') and hasattr(submodule_obj, 'lora_B') and \
                   isinstance(submodule_obj.lora_A, torch.nn.ModuleDict) and 'default' in submodule_obj.lora_A and \
                   isinstance(submodule_obj.lora_B, torch.nn.ModuleDict) and 'default' in submodule_obj.lora_B:
                    
                    current_lora_A_id = id(submodule_obj.lora_A['default'].weight.data)
                    current_lora_B_id = id(submodule_obj.lora_B['default'].weight.data)
                    
                    a_match = "MATCHES" if current_lora_A_id == ref_lora_A_weight_id else "MISMATCH!"
                    b_match = "MATCHES" if current_lora_B_id == ref_lora_B_weight_id else "MISMATCH!"

                    logger.info(f"    Layer {i} ({submodule_name}): LoRA A ID: {current_lora_A_id} ({a_match}), LoRA B ID: {current_lora_B_id} ({b_match})")
                    if a_match == "MISMATCH!" or b_match == "MISMATCH!":
                        logger.error(f"      Tieing FAILED for '{target_name}' in layer {i} compared to layer 0.")
                        all_checks_passed = False
                    found_in_current_layer = True
                    break # Found the target module in this layer
            if not found_in_current_layer:
                logger.warning(f"    Target module '{target_name}' with LoRA A/B weights not found in layer {i}.")
    
    if all_checks_passed:
        logger.info("Verification of tied LoRA weights complete. All checked modules appear to be tied.")
    else:
        logger.error("Verification of tied LoRA weights complete. Some mismatches were found.")

if __name__ == "__main__":
    # --- Configuration ---
    BASE_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct" # Or your specific base model
    # Example: "/home/ka/ka_stud/ka_usxcp/master-thesis/outputs/llama2-7b-lora-test/checkpoint-500/adapter_model"
    # Ensure this path points to the directory containing adapter_config.json and adapter_model.bin
    ADAPTER_DIRECTORY_PATH = "train_lora_1B/checkpoint-1800" 
    # Or if you saved it at the end of training directly in output_dir:
    # ADAPTER_DIRECTORY_PATH = "/home/ka/ka_stud/ka_usxcp/master-thesis/outputs/your_experiment_run/adapter_model" 
    
    # These should match the `lora_target_modules` from your training LoraConfig
    # Example from your training script: lora_hyper['lora_target_modules']
    # You might need to load your lora_config.json from training to get this list dynamically
    # For now, let's assume you know them:
    TARGET_MODULES = ["q_proj", "v_proj"] # Replace with your actual target modules

    # --- Run Verification ---
    logger.info(f"Using ADAPTER_DIRECTORY_PATH: {ADAPTER_DIRECTORY_PATH}")

    
    verify_tied_lora_weights_from_disk(
        base_model_name_or_path=BASE_MODEL_PATH,
        adapter_path=ADAPTER_DIRECTORY_PATH,
        lora_target_modules=TARGET_MODULES
    )

