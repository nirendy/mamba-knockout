import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from src.data_ingestion.helpers.logits_utils import find_token_range

B_D = 1  # Batch dimension
INTRM_D = 8  # Intermediate dimension
T = 8

TQ = torch.Tensor
TK = torch.Tensor
TV = torch.Tensor


@pytest.fixture
# @tensor_type_check
def tokenizer() -> PreTrainedTokenizerFast | PreTrainedTokenizer:
    """Fixture providing a tokenizer for the test."""
    return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


@pytest.mark.parametrize(
    "model_id,string,subject",
    [
        ("meta-llama/Llama-2-7b-hf", "Beats music is owned by", "Beats music"),
        ("meta-llama/Llama-2-7b-hf", "The capital of France is", "France"),
        ("meta-llama/Llama-2-7b-hf", "The holy roman empire was founded by", "holy roman empire"),
        ("meta-llama/Llama-2-7b-hf", "The intel x86 processor was invented in ", "intel x86 processor"),
        ("Qwen/Qwen2-0.5B", "Beats music is owned by", "Beats music"),
        ("Qwen/Qwen2-0.5B", "The capital of France is", "France"),
        ("Qwen/Qwen2-0.5B", "The holy roman empire was founded by", "holy roman empire"),
        ("Qwen/Qwen2-0.5B", "The intel x86 processor was invented in ", "intel x86 processor"),
        ("Qwen/Qwen2-1.5B", "Beats music is owned by", "Beats music"),
        ("Qwen/Qwen2-1.5B", "The capital of France is", "France"),
        ("Qwen/Qwen2-1.5B", "The holy roman empire was founded by", "holy roman empire"),
        ("Qwen/Qwen2-1.5B", "The intel x86 processor was invented in ", "intel x86 processor"),
        ("Qwen/Qwen2-7B", "Beats music is owned by", "Beats music"),
        ("Qwen/Qwen2-7B", "The capital of France is", "France"),
        ("Qwen/Qwen2-7B", "The holy roman empire was founded by", "holy roman empire"),
        ("Qwen/Qwen2-7B", "The intel x86 processor was invented in ", "intel x86 processor"),
        ("Qwen/Qwen2.5-0.5B", "Beats music is owned by", "Beats music"),
        ("Qwen/Qwen2.5-0.5B", "The capital of France is", "France"),
        ("Qwen/Qwen2.5-0.5B", "The holy roman empire was founded by", "holy roman empire"),
        ("Qwen/Qwen2.5-0.5B", "The intel x86 processor was invented in ", "intel x86 processor"),
        ("Qwen/Qwen2.5-1.5B", "Beats music is owned by", "Beats music"),
        ("Qwen/Qwen2.5-1.5B", "The capital of France is", "France"),
        ("Qwen/Qwen2.5-1.5B", "The holy roman empire was founded by", "holy roman empire"),
        ("Qwen/Qwen2.5-1.5B", "The intel x86 processor was invented in ", "intel x86 processor"),
        ("Qwen/Qwen2.5-3B", "Beats music is owned by", "Beats music"),
        ("Qwen/Qwen2.5-3B", "The capital of France is", "France"),
        ("Qwen/Qwen2.5-3B", "The holy roman empire was founded by", "holy roman empire"),
        ("Qwen/Qwen2.5-3B", "The intel x86 processor was invented in ", "intel x86 processor"),
        ("Qwen/Qwen2.5-7B", "Beats music is owned by", "Beats music"),
        ("Qwen/Qwen2.5-7B", "The capital of France is", "France"),
        ("Qwen/Qwen2.5-7B", "The holy roman empire was founded by", "holy roman empire"),
        ("Qwen/Qwen2.5-7B", "The intel x86 processor was invented in ", "intel x86 processor"),
    ],
)
def test_knockout(model_id, string, subject):
    """Test knockout scan behavior with different knockout configurations."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    string_toks = tokenizer(string)
    subj_range = find_token_range(
        tokenizer,
        string_toks["input_ids"],
        subject,
    )
    subj_toks = tokenizer(string)["input_ids"]
    subj_toks = subj_toks[subj_range[0] : subj_range[1]]
    decoded_subj = tokenizer.decode(subj_toks)
    assert subject in decoded_subj, f"Decoded subject '{decoded_subj}' does not match expected subject '{subject}'"
    decoded_subj = tokenizer.decode(subj_toks[1:])
    assert subject not in decoded_subj, (
        f"Decoded subject '{decoded_subj}' should not match expected subject '{subject}'"
    )
    decoded_subj = tokenizer.decode(subj_toks[:-1])
    assert subject not in decoded_subj, (
        f"Decoded subject '{decoded_subj}' should not match expected subject '{subject}'"
    )
