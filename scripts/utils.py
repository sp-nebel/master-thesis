import json
import re
import torch

from transformers import AutoTokenizer

massive_lang_map = {"en": "en-US",
                    "af": "af-ZA",
                    "am": "am-ET",
                    "ar": "ar-SA",
                    "az": "az-AZ",
                    "bn": "bn-BD",
                    "cy": "cy-GB",
                    "da": "da-DK",
                    "de": "de-DE",
                    "el": "el-GR",
                    "es": "es-ES",
                    "fa": "fa-IR",
                    "fi": "fi-FI",
                    "fr": "fr-FR",
                    "id": "id-ID",
                    "he": "he-IL",
                    "hi": "hi-IN",
                    "hu": "hu-HU",
                    "hy": "hy-AM",
                    "id": "id-ID",
                    "is": "is-IS",
                    "it": "it-IT",
                    "ja": "ja-JP",
                    "jv": "jv-ID",
                    "ka": "ka-GE",
                    "km": "km-KH",
                    "kn": "kn-IN",
                    "ko": "ko-KR",
                    "lv": "lv-LV",
                    "ml": "ml-IN",
                    "mn": "mn-MN",
                    "pl": "pl-PL",
                    "pt": "pt-PT",
                    "ru": "ru-RU",
                    "sw": "sw-KE",
                    "th": "th-TH",
                    "tl": "tl-PH",
                    "tr": "tr-TR",
                    "ur": "ur-PK",
                    "zh": "zh-CN",
                    }

idx2label_ner = {1: "(person)", 2: "(organization)", 3: "(location)", 4: "(miscellaneous)"}

translation_lang_map = {
    "am": "Amharic",
    "ar": "modern standard Arabic",
    "bn": "Bengali",
    "cs": "Czech", "ces": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German", "deu": "German",
    "en": "English", "eng": "English",
    "es": "Spanish", "spa": "Spanish",
    "el": "Greek",
    "fa": "Farsi (Persian)",
    "fr": "French", "fra": "French",
    "hr": "Croatian",
    "is": "Icelandic",
    "it": "Italian", "ita": "Italian",
    "he": "Hebrew",
    "hi": "Hindi",
    "ja": "Japanese",
    "jv": "Javanese",
    "ko": "Korean",
    "mn": "Mongolian",
    "nl": "Dutch", "nld": "Dutch",
    "pt": "Portuguese", "por": "Portuguese",
    "ro": "Romanian", "ron": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sr": "Serbian",
    "sv": "Swedish",
    "swh": "Swahili",
    "te": "Telugu",
    "tl": "Tagalog",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "zh": "mandarin Chinese", "cmn": "mandarin Chinese",
}


uner_input_delimiter = {"en": "what is the output result?",
                        "da": "I betragtning af sætningen nedenfor, hvad er resultatet?",
                        "hr": "S obzirom na niže navedenu ulaznu rečenicu, koji je konačni izlaz?",
                        "pt": "Considerando a frase de entrada abaixo, qual é o resultado produzido?",
                        "sk": "Aký bude výstup vzhľadom na vstup uvedený nižšie?",
                        "sr": "S obzirom na niže navedenu ulaznu rečenicu, koji je konačni izlaz?",
                        "sv": "Med tanke på meningen nedan, vad är resultatet?",
                        "zh": "考慮到下麵的輸入句子，輸出結果是什麼？"}


def convert_row_massive_slot_filling(example):
    matches = re.findall('\[(.+?)\]', example["annot_utt"])
    if matches:
        example["annot_utt"] = "; ".join(matches).replace(" :", ":")
    else:
        example["annot_utt"] = "None"
    return example


def convert_row_translation(example, sl, tl):
    translation = example["translation"]
    # sl, tl = translation.keys()
    example["src"] = translation[sl]
    example["tgt"] = translation[tl]
    return example


def parse_json_entities(input_string, entity_in):
    # Regex pattern to match the location entries
    pattern = fr'"TypeName": "{entity_in}",\s*"Text": "([^"]+)"'
    # Find all matches
    entities = re.findall(pattern, input_string)
    # Format the locations
    return '; '.join(f"{entity_in}: {e}" for e in entities)


def parse_3standard_ner_from_json(input_string):
    input_string = input_string.strip().split("\"Results\": [")[1].strip("]")
    if not input_string:
        return "None"

    parsed_output = []
    for entity in ["LOC", "PER", "ORG"]:
        parsed_res = parse_json_entities(input_string, entity)
        if parsed_res:
            parsed_output.append(parsed_res)
    return "; ".join(parsed_output)

def tie_lora_weights(model, lora_config):
    
    llama_layers = model.base_model.model.model.layers

    for target_name_in_lora_config in lora_config.target_modules:
        
        ref_lora_A_weight = None
        ref_lora_B_weight = None

        first_llama_layer = llama_layers[0]
        for submodule_path, submodule_obj in first_llama_layer.named_modules():
            if (submodule_path.split('.')[-1] != target_name_in_lora_config or
                    not (hasattr(submodule_obj, 'lora_A') and
                         hasattr(submodule_obj, 'lora_B') and
                         isinstance(submodule_obj.lora_A, torch.nn.ModuleDict) and
                         'default' in submodule_obj.lora_A and
                         isinstance(submodule_obj.lora_B, torch.nn.ModuleDict) and
                         'default' in submodule_obj.lora_B)):
                continue
                
            ref_lora_A_weight = submodule_obj.lora_A['default'].weight
            ref_lora_B_weight = submodule_obj.lora_B['default'].weight
            break 

        for current_llama_layer in llama_layers:
            for submodule_path, submodule_obj in current_llama_layer.named_modules():
                if (submodule_path.split('.')[-1] != target_name_in_lora_config or
                    not (hasattr(submodule_obj, 'lora_A') and
                         hasattr(submodule_obj, 'lora_B') and
                         isinstance(submodule_obj.lora_A, torch.nn.ModuleDict) and
                         'default' in submodule_obj.lora_A and
                         isinstance(submodule_obj.lora_B, torch.nn.ModuleDict) and
                         'default' in submodule_obj.lora_B)):
                    continue
                       
                submodule_obj.lora_A['default'].weight = ref_lora_A_weight
                submodule_obj.lora_B['default'].weight = ref_lora_B_weight
                # logger.debug(f"    Tied LoRA weights for '{target_name_in_lora_config}' in layer {i} to layer 0's weights.")
                break
