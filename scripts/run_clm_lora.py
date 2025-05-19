#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset

from scripts.utils import tie_lora_weights
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
# xxx: 2023-04-11
import copy
import json
from transformers.utils import add_start_docstrings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl, EarlyStoppingCallback
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModel
)
import time



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# xxx: 2023-03-21
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    contrastive_data_mode: int = field(
        default=0,
        metadata={"help": "How to add semantically parallel data as input."},
    )
    additional_loss_layer: int = field(
        default=16,
        metadata={"help": "TBA"},
    )
    contrastive_loss_temperature: float = field(
        default=1.0,
        metadata={"help": "TBA"},
    )
    distance_function: str = field(
        default="cosine",
        metadata={"help": "TBA"},
    )
    contrastive_loss_weight: float = field(
        default=1.0,
        metadata={"help": "TBA"},
    )
    only_train_contrastive: bool = field(
        default=False,
        metadata={"help": "Deactivate language modeling objective"},
    )
    only_train_language_modeling: bool = field(
        default=False,
        metadata={"help": "Deactivate contrastive learning objective"},
    )
    alternate_training: bool = field(
        default=False,
        metadata={"help": "Alternate training between main and auxilliary task"},
    )
    multitask_training: bool = field(
        default=False,
        metadata={"help": "joint MTL"},
    )
    unidirectional_contrastive_loss: bool = field(
        default=False,
        metadata={"help": "If true, detach one input of the contrastive loss to push the other representation to this"},
    )
    contrastive_pooling_type: str = field(
        default="mean",
        metadata={"help": "How to pool contrastive instances to fixed length for comparision"},
    )
    inject_Ws: bool = field(
        default=False,
        metadata={"help": "If true, load W matrices (trained offline) and hold them unchanged in training"},
    )
    alignment_matrices_path: str = field(
        default=None,
        metadata={"help": "Path to alignment matrices"},
    )
    apply_inverse: bool = field(
        default=False,
        metadata={"help": "If true, apply inverse of W after lora"},
    )
    # use_encoder_decoder: bool = field(
    #     default=False,
    #     metadata={"help": "If true, use encoder-decoder model instead of decoder-only model"},
    # )
    use_dora: bool = field(
        default=False,
        metadata={"help": "If true, use dora"},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    patience: Optional[int] = field(
        default=5,
        metadata={
            "help": ""
        },
    )
    # max_train_length: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "Skip all training samples longer than this"
    #         )
    #     },
    # )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


# xxx: 2023-04-11, customized args
@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class LoRATrainingArguments(TrainingArguments):
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA."}
    )
    use_int8_training: bool = field(
        default=False,
        metadata={"help": "Whether to use int8 training."}
    )
    lora_config: Optional[str] = field(
        default=None,
        metadata={"help": "LoRA config file."},
    )
    load_lora_from: Optional[str] = field(
        default=None,
        metadata={"help": "Load LoRA model from a path as initialization."},
    )


# xxx: save peft adapters at steps/epoch end
class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # if os.path.exists(pytorch_model_path):
        #     os.remove(pytorch_model_path)
        return control


# xxx: save peft at train end
class SavePeftModelAtEndCallback(TrainerCallback):
    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        peft_model_path = os.path.join(args.output_dir, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        # pytorch_model_path = os.path.join(state.best_model_checkpoint, "pytorch_model.bin")
        # if os.path.exists(pytorch_model_path):
        #     os.remove(pytorch_model_path)
        return control


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # xxx: 2023-04-12
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoRATrainingArguments))
    #parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            if "," in data_args.train_file: # DL: a list of files provided
                data_files["train"] = data_args.train_file.split(",")
            else:
                data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            if "," in data_args.validation_file:  # DL: a list of files provided
                data_files["validation"] = data_args.validation_file.split(",")
            else:
                data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "additional_loss_layer": model_args.additional_loss_layer,
        "contrastive_loss_temperature": model_args.contrastive_loss_temperature,
        "distance_function": model_args.distance_function,
        "contrastive_loss_weight": model_args.contrastive_loss_weight,
        "only_train_contrastive": model_args.only_train_contrastive,
        "only_train_language_modeling": model_args.only_train_language_modeling,
        "unidirectional_contrastive_loss": model_args.unidirectional_contrastive_loss,
        "contrastive_pooling_type": model_args.contrastive_pooling_type,
        "inject_Ws": model_args.inject_Ws,
        "alignment_matrices_path": model_args.alignment_matrices_path,
        "apply_inverse": model_args.apply_inverse,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if "llama-2" in model_args.model_name_or_path:
        if model_args.use_fast_tokenizer:
            # tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer",
            #                                    padding_side="right") this breaks </s>
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf", from_slow=True)
        else:
            tokenizer_class = LlamaTokenizer #if "llama" in model_args.model_name_or_path else AutoTokenizer
            tokenizer = tokenizer_class.from_pretrained(
                model_args.model_name_or_path,
                padding_side="right",
            )
    logger.info(f"Tokenizer is fast: {tokenizer.is_fast}")

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        # xxx: 2023-04-11, LoRA
        # xxx: int8 is not compatible with DeepSpeed (require not to pass device_map)
        # xxx: 8bit models should not be converted to DDP
        if training_args.use_int8_training:
            #world_size = int(os.environ.get("WORLD_SIZE", 1))
            #device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size != 1 else "auto"
            device_map = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                load_in_8bit=True,      # xxx: int8 load in
                device_map=device_map,  # xxx: int8 requires passing device_map
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
            )
        else:
            # if not model_args.use_encoder_decoder:
                # if "llama-3" not in model_args.model_name_or_path.lower():
                #     model = AutoModelForCausalLM.from_pretrained(
                #         model_args.model_name_or_path,
                #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
                #         config=config,
                #         cache_dir=model_args.cache_dir,
                #         revision=model_args.model_revision,
                #         use_auth_token=True if model_args.use_auth_token else None,
                #         torch_dtype=torch_dtype,
                #         attn_implementation="flash_attention_2",
                #     )
                # else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
            )
            # else:
            #     model = AutoModelForSeq2SeqLM.from_pretrained(
            #         model_args.model_name_or_path,
            #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
            #         config=config,
            #         cache_dir=model_args.cache_dir,
            #         revision=model_args.model_revision,
            #         use_auth_token=True if model_args.use_auth_token else None,
            #         torch_dtype=torch_dtype,
            #     )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # xxx: 2023-03-21, add padding
    # if tokenizer.pad_token is None:
    #     if "Llama-3" in model_args.model_name_or_path:
    #         tokenizer.pad_token = "<|reserved_special_token_0|>"
    #     else:
    #         tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    # tokenizer.padding_side = "right"

    # xxx: 2023-03-21, add special tokens
    if "llama-2" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    elif "llama-3" in model_args.model_name_or_path.lower():
        print("adding special tokens...")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.unk_token = "<|reserved_special_token_0|>"
    elif "mistral" in model_args.model_name_or_path.lower():
        print("adding special tokens...")
        tokenizer.pad_token = tokenizer.eos_token

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    logger.info("================ pad, eos, bos, unk, padding ================ ")
    logger.info(f"{tokenizer.pad_token}, {tokenizer.pad_token_id}")
    logger.info(f"{tokenizer.eos_token}, {tokenizer.eos_token_id}")
    logger.info(f"{tokenizer.bos_token}, {tokenizer.bos_token_id}")
    logger.info(f"{tokenizer.unk_token}, {tokenizer.unk_token_id}")
    logger.info(f"{tokenizer.padding_side}")

    # xxx: 2023-04-11, setup LoRA
    if training_args.use_lora:
        if training_args.use_int8_training:
            model = prepare_model_for_kbit_training(model)
        lora_hyper = json.load(open(training_args.lora_config))
        for key, value in lora_hyper.items():
            logger.info("{} : {}".format(key, value))
        lora_config = LoraConfig(
            r=lora_hyper['lora_r'],
            lora_alpha=lora_hyper['lora_alpha'],
            target_modules=lora_hyper['lora_target_modules'],
            lora_dropout=lora_hyper['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=model_args.use_dora,
        )
        logger.info(f"LoRA configs: {lora_config}")
        # xxx: To avoid "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        # xxx: Seems due to gradient_checkpointing, to check later
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if training_args.load_lora_from is None:
            model = get_peft_model(model, lora_config)
        else:
            peft_model_id = training_args.load_lora_from
            # config = PeftConfig.from_pretrained(peft_model_id)
            model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=True)
        print(dir(model.base_model.model))
        tie_lora_weights(model, lora_config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        print(model)
    else:
        logger.info("Running full finetuning, incl. embeddings and LM head")
        # logger.info("Running full finetuning, freezing embeddings and LM head")
        # for name, param in model.named_parameters():
        #     if "embed" in name or "lm_head" in name:
        #         param.requires_grad = False
        logger.info(f"*** trainable params: {model.num_parameters(only_trainable=True)}")  # Be more transparent about the % of trainable params.
        print(model)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    logger.info(f"block size: {block_size}")

    # xxx: 2023-03-14
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # xxx: 2023-05-17, leave padding to DataCollatorForSeq2Seq for batch padding and avoid unnecessary paddings
    def preprocess_function(examples):
        with CaptureLogger(tok_logger) as cl:
            # xxx: 2023-04-07; text: target, prefix: source
            padding = "max_length"  # or False
            text = examples[text_column_name]  # may have multiple strings
            if "prefix" in column_names:
                # if not model_args.use_encoder_decoder:
                prefix = examples["prefix"]  # may have multiple strings
                text = [s + t for s, t in zip(prefix, text)]
                prefix_tokenized = tokenizer(prefix, truncation=True, max_length=block_size, padding=False)
                text_tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=False)
                labels = copy.deepcopy(text_tokenized["input_ids"])
                prefix_lengths = [len(p) for p in prefix_tokenized["input_ids"]]
                for label, prefix_len in zip(labels, prefix_lengths):  # Do not compute loss for prompt inputs
                    label[:prefix_len] = [IGNORE_INDEX] * prefix_len  # [IGNORE_INDEX for i in range(prefix_len)]
                # else:
                #     text_tokenized = tokenizer(examples["prefix"],
                #                                text_target=text, truncation=True, max_length=block_size, padding=False)
                #     labels = text_tokenized["labels"]
                    # labels = tokenizer(truncation=True, max_length=block_size, padding=False)
            else:
                text_tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=False)
                labels = copy.deepcopy(text_tokenized["input_ids"])
            text_tokenized["labels"] = labels
            # DL: add extra fields
            # text_tokenized["prefix_length"] = prefix_lengths
            # text_tokenized["language_id"] = examples["language"]
            # text_tokenized["task_id"] = examples["task"]
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return text_tokenized

    def preprocess_function_plus_contrastive(examples):
        # --------------> TODO(DL): remove duplication
        with CaptureLogger(tok_logger) as cl:
            # xxx: 2023-04-07; text: target, prefix: source
            padding = "max_length"  # or False
            text = examples[text_column_name]  # may have multiple strings
            if "prefix" in column_names:
                prefix = examples["prefix"]  # may have multiple strings
                text = [s + t for s, t in zip(prefix, text)]
                prefix_tokenized = tokenizer(prefix, truncation=True, max_length=block_size, padding=False)
                text_tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=False)
                labels = copy.deepcopy(text_tokenized["input_ids"])
                prefix_lengths = [len(p) for p in prefix_tokenized["input_ids"]]
                for label, prefix_len in zip(labels, prefix_lengths):  # Do not compute loss for prompt inputs
                    label[:prefix_len] = [IGNORE_INDEX] * prefix_len  # [IGNORE_INDEX for i in range(prefix_len)]
            else:
                text_tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=False)
                labels = copy.deepcopy(text_tokenized["input_ids"])
            text_tokenized["labels"] = labels
            # DL: add extra fields
            text_tokenized["prefix_length"] = prefix_lengths
            # text_tokenized["language_id"] = examples["language"]
            # text_tokenized["task_id"] = examples["task"]
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        # <-------------- TODO(DL): remove duplication
        input_parallel_1, input_parallel_2 = examples["text_parallel_0"], examples["text_parallel_1"]
                                        #["I'm English"] * len(examples["prefix"]), \
        #                                 ["Ich bin Deutsch Ich bin Deutsch Ich bin Deutsch"] * len(examples["prefix"])
        #examples["text"], examples["text_parallel"]

        # contains input_ids, attention_mask, labels
        # leaving padding to collator
        for i, additional_input in enumerate([input_parallel_1, input_parallel_2]):
            additional_input_tokenized = tokenizer(additional_input, truncation=True, max_length=block_size, padding=False)
            # we do not need labels for contrastive examples
            # additional_input_tokenized["labels"] = copy.deepcopy(additional_input_tokenized["input_ids"])

            for k in additional_input_tokenized:  # merge input_ids, attention_mask, labels to existing dictionary
                text_tokenized[f"{k}_parallel_{i}"] = additional_input_tokenized[k]
        return text_tokenized

    def preprocess_function_contrastive(examples):
        # input, input_parallel = ["=========== I'm English"] * len(examples), \
        #                         ["Ich bin Deutsch ============="] * len(examples)
        input, input_parallel = examples["text"], examples["text_parallel"]

        # contains input_ids, attention_mask, labels
        # leaving padding to collator
        input_tokenized = tokenizer(input, truncation=True, max_length=block_size, padding=False)
        input_tokenized["labels"] = copy.deepcopy(input_tokenized["input_ids"])

        input_parallel_tokenized = tokenizer(input_parallel, truncation=True, max_length=block_size, padding=False)
        input_parallel_tokenized["labels"] = copy.deepcopy(input_parallel_tokenized["input_ids"])

        for k in input_parallel_tokenized:  # merge input_ids, attention_mask, labels to existing dictionary
            input_tokenized[f"{k}_parallel"] = input_parallel_tokenized[k]
        return input_tokenized
    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # xxx: 2023-03-17
    with training_args.main_process_first(desc="example per line with padding"):
        if model_args.contrastive_data_mode == 0:
            preprocess_func = preprocess_function
        elif model_args.contrastive_data_mode == 1:     # only contrastive
            preprocess_func = preprocess_function_contrastive
        elif model_args.contrastive_data_mode == 2:     # only contrastive   # task and contrastive
            preprocess_func = preprocess_function_plus_contrastive
        else:
            raise NotImplementedError

        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                preprocess_func,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Tokenize with padding",
            )
        else:
            lm_datasets = raw_datasets.map(
                preprocess_func,
                batched=True,
                remove_columns=column_names,
            )


    if training_args.do_train:
        #if "train" not in tokenized_datasets:
        # xxx: 2023-03-14
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")

        train_dataset = lm_datasets["train"]

        # if data_args.max_train_length is not None:
        #     logger.info(f"Filtering training set by max length: {data_args.max_train_length}")
        #     logger.info(f"Before: {len(train_dataset)} samples")
        #     train_dataset = train_dataset.filter(lambda x: len(x["labels"]) <= data_args.max_train_length)#,
        #                                          #num_proc=data_args.preprocessing_num_workers,)
        #     logger.info(f"After: {len(train_dataset)} samples")

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # xxx: print samples
        logger.info("xxx: Showcase the tokenized training samples.")
        for i in range(3):
            print(next(iter(train_dataset)))

    if training_args.do_eval:
        #if "validation" not in tokenized_datasets:
        # xxx: 2023-03-14
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if len(logits) == 0:
                return None
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", experiment_id=str(time.time()))

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # xxx: 2023-04-13, load pretrained adapter weights
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # xxx: checkpoint is a folder
        if checkpoint:
            peft_model_name = os.path.join(checkpoint, "adapter_model/adapter_model.bin")
            if os.path.exists(peft_model_name):
                logger.info(f"xxx: Load pretrained adapter weights from {peft_model_name}")
                adapters_weights = torch.load(peft_model_name)
                set_peft_model_state_dict(model, adapters_weights)
                logger.info(f"xxx: Double check the trainable parameters...")
                model.print_trainable_parameters()

    callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.patience)]
    peft_callbacks = [SavePeftModelCallback, SavePeftModelAtEndCallback] if training_args.use_lora else []
    callbacks.extend(peft_callbacks)

    # Initialize our Trainer
    # DL: updated to multitask trainer for logging different losses
    if model_args.alternate_training:
        from transformers import AlternateTrainer
        trainer = AlternateTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            #data_collator=default_data_collator,
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                              padding=True, label_pad_token_id=IGNORE_INDEX),
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
            callbacks=callbacks,     # xxx: 2023-04-12, callbacks for
        )
    # elif model_args.multitask_training:
    #     from hf_mtask_trainer import HfMultiTaskTrainer
    #     trainer = HfMultiTaskTrainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset if training_args.do_train else None,
    #         eval_dataset=eval_dataset if training_args.do_eval else None,
    #         tokenizer=tokenizer,
    #         # Data collator will default to DataCollatorWithPadding, so we change it.
    #         # data_collator=default_data_collator,
    #         data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
    #                                                           padding=True, label_pad_token_id=IGNORE_INDEX),
    #         compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
    #         preprocess_logits_for_metrics=preprocess_logits_for_metrics
    #         if training_args.do_eval and not is_torch_tpu_available()
    #         else None,
    #         callbacks=callbacks,  # xxx: 2023-04-12, callbacks for
    #     )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            # data_collator=default_data_collator,
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                              padding=True, label_pad_token_id=IGNORE_INDEX),
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
            callbacks=callbacks,  # xxx: 2023-04-12, callbacks for
        )
    # xxx: 2023-04-11, LoRA
    if training_args.use_lora:
        model.config.use_cache = False

    # DL: not compatible with grad checkpointing
    model.config.use_cache = False

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
