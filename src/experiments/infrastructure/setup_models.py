import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, assert_never

from huggingface_hub import hf_hub_download, login

from src.core.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID, is_falcon
from src.core.types import MODEL_ARCH, MODEL_ARCH_AND_SIZE, TDevice, TModel, TModelSize, TTokenizer


def get_tokenizer_path(model_arch_and_size: MODEL_ARCH_AND_SIZE) -> str:
    model_arch = model_arch_and_size.arch
    model_size = model_arch_and_size.size

    if model_arch == MODEL_ARCH.MAMBA2:
        return "EleutherAI/gpt-neox-20b"
    return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch][model_size]


def get_tokenizer_config_from_hub(model_arch_and_size: MODEL_ARCH_AND_SIZE) -> dict:
    tokenizer_path = get_tokenizer_path(model_arch_and_size)

    filename = "tokenizer_config.json"

    # Download just the tokenizer config
    path = Path(hf_hub_download(repo_id=tokenizer_path, filename=filename))

    # Load the JSON content
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def get_tokenizer(model_arch: MODEL_ARCH, model_size: TModelSize) -> TTokenizer:
    from transformers import AutoTokenizer

    tokenizer_path = get_tokenizer_path(MODEL_ARCH_AND_SIZE(model_arch, model_size))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_tokenizer_and_model(
    model_arch: MODEL_ARCH, model_size: TModelSize, device: Optional[TDevice] = None
) -> tuple[TTokenizer, TModel]:
    if os.getenv("HUGGINGFACE_TOKEN") is not None:
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
    else:
        print("No HuggingFace token found")

    minimal_kwargs = {
        "device": device,
        "device_map": "auto" if device is None else None,
    }

    model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch][model_size]
    tokenizer = get_tokenizer(model_arch, model_size)

    match model_arch:
        case MODEL_ARCH.MAMBA2:
            import src.experiments.knockout.mamba.mamba2.minimal_mamba2 as minimal_mamba2

            model = minimal_mamba2.Mamba2LMHeadModel.from_pretrained(model_id, **minimal_kwargs)  # type: ignore
        case MODEL_ARCH.MAMBA1:
            if is_falcon(model_size):
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
            else:
                from transformers import MambaForCausalLM

                if device:
                    model = MambaForCausalLM.from_pretrained(model_id)
                    model.to(device)  # type: ignore
                else:
                    model = MambaForCausalLM.from_pretrained(model_id, device_map="auto")
        case MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2 | MODEL_ARCH.LLAMA3:
            from transformers import LlamaForCausalLM

            if device:
                model = LlamaForCausalLM.from_pretrained(model_id)
                model.to(device)  # type: ignore
            else:
                model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto")
        case MODEL_ARCH.MISTRAL0_1 | MODEL_ARCH.MISTRAL0_3:
            from transformers import MistralForCausalLM

            if device:
                model = MistralForCausalLM.from_pretrained(model_id)
                model.to(device)  # type: ignore
            else:
                model = MistralForCausalLM.from_pretrained(model_id, device_map="auto")
        case MODEL_ARCH.QWEN2_5 | MODEL_ARCH.QWEN2:
            from transformers import Qwen2ForCausalLM

            if device:
                model = Qwen2ForCausalLM.from_pretrained(model_id)
                model.to(device)  # type: ignore
            else:
                model = Qwen2ForCausalLM.from_pretrained(model_id, device_map="auto")
        case MODEL_ARCH.GPT2:
            from transformers import GPT2LMHeadModel

            model = GPT2LMHeadModel.from_pretrained(model_id, device_map="auto")
        case _:
            assert_never(model_arch)

    return tokenizer, model
