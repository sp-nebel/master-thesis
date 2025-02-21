import json
import re

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