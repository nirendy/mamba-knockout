from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, assert_never, cast

import pandas as pd
import torch

from src.core.names import COLS
from src.core.types import (
    MODEL_ARCH,
    TDevice,
    TNum2Mask,
    TokenType,
    TPromptData,
    TPromptOriginalIndex,
    TTokenIndex,
    TTokenizer,
    TWindow,
)


def get_last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :]


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def get_top_k_outputs_and_probs(logits: torch.Tensor, tokenizer, top_k: int):
    next_probs = logits_to_probs(get_last_token_logits(logits))
    top_probs, top_indices = torch.topk(next_probs, top_k)
    top_outputs = [
        (idx.item(), str(tokenizer.decode([idx])), prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])
    ]
    return top_outputs


def get_top_outputs(probs, tokenizer, top_k):
    # Get the top 5 outputs and their probs
    top_probs, top_indices = map(torch.Tensor.tolist, torch.topk(torch.Tensor(probs), top_k))
    top_tokens = list(map(tokenizer.batch_decode, top_indices))
    return list(
        map(
            list,
            map(
                lambda x: zip(*x),
                zip(
                    top_indices,
                    top_tokens,
                    top_probs,
                ),
            ),
        )
    )


# Taken from https://github.com/google-research/google-research/blob/master/dissecting_factual_predictions/utils.py
def decode_tokens(tokenizer: TTokenizer, token_array: torch.Tensor) -> Union[list[str], list[list[str]]]:
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [cast(list[str], decode_tokens(tokenizer, row)) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(
    tokenizer: TTokenizer,
    token_array: torch.Tensor,
    substring: str,
) -> tuple[TTokenIndex, TTokenIndex]:
    """Find the tokens corresponding to the given substring in token_array."""
    toks = decode_tokens(tokenizer, token_array)

    # whole_string = "".join(toks)  # type: ignore
    # if ' ' not in whole_string:
    #     whole_string = " ".join(toks)  # type: ignore
    whole_string = tokenizer.decode(token_array)
    # print(f"whole_string: {whole_string}")
    # print(f"substring: {substring}")
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        loc = len(tokenizer.decode(token_array[:i]))
        if tok_start is None and loc > char_loc:
            tok_start = i - 1
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i
            break
    # print(loc, char_loc, whole_string, substring, tok_start, tok_end)
    # print(tokenizer.decode(token_array[tok_start:tok_end]))
    assert tok_start is not None and tok_end is not None, "Token range not found"
    return (tok_start, tok_end)


@dataclass
class Prompt:
    # prompt_row: TPromptData
    prompt_row: pd.DataFrame | pd.Series

    @property
    def original_idx(self) -> TPromptOriginalIndex:
        return cast(TPromptOriginalIndex, self.prompt_row.name)

    @property
    def prompt(self) -> str:
        return str(self.prompt_row[COLS.COUNTER_FACT.PROMPT])

    @property
    def subject(self) -> str:
        return str(self.prompt_row[COLS.COUNTER_FACT.SUBJECT])

    @property
    def true_word(self) -> str:
        return str(self.prompt_row[COLS.COUNTER_FACT.TARGET_TRUE])

    @property
    def base_prob(self) -> float:
        return cast(float, self.prompt_row[COLS.EVALUATE_MODEL.TARGET_PROBS])

    @property
    def target_rank(self) -> int:
        return cast(int, self.prompt_row[COLS.EVALUATE_MODEL.TARGET_RANK])

    @property
    def relation(self) -> str:
        return cast(str, self.prompt_row[COLS.COUNTER_FACT.RELATION])

    def get_value(self, column: COLS.COUNTER_FACT) -> Any:
        return cast(Any, self.prompt_row[column])

    def true_id(self, tokenizer, device: TDevice) -> torch.Tensor:
        toks = tokenizer(self.true_word, add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(
            device=device
        )
        # if toks.shape[1] == 1:
        return toks
        # else:
        # return toks[:, 1:]

    def true_id_v2(self, tokenizer: TTokenizer, device: TDevice) -> torch.Tensor:
        # compute by the difference between the prompt and the prompt + true_word
        prompt_ids = self.input_ids(tokenizer, device)
        prompt_plus_true_ids = tokenizer(self.prompt + self.true_word, return_tensors="pt", padding=True).input_ids.to(
            device=device
        )
        assert (prompt_ids == prompt_plus_true_ids[:, : len(prompt_ids[0])]).all()
        return prompt_plus_true_ids[:, len(prompt_ids[0]) :]

    def input_ids(self, tokenizer: TTokenizer, device: TDevice) -> torch.Tensor:
        return tokenizer(self.prompt, return_tensors="pt", padding=True).input_ids.to(device=device)

    def get_column(self, column: COLS.COUNTER_FACT) -> Any:
        return self.prompt_row[column]

    def last_index(self, tokenizer: TTokenizer, device: TDevice) -> TTokenIndex:
        input_ids = self.input_ids(tokenizer, device)
        return input_ids.shape[1] - 1

    def is_relation_last_token(self) -> bool:
        relation_suffix = self.get_value(COLS.COUNTER_FACT.RELATION_SUFFIX)
        return relation_suffix is not None and relation_suffix != ""

    def get_knockout_idx(self, tokenizer: TTokenizer, knockout: TokenType, device: TDevice) -> list[TTokenIndex]:
        input_ids = self.input_ids(tokenizer, device)
        last_idx = input_ids.shape[1] - 1
        subject_token_range = find_token_range(tokenizer, input_ids[0], self.subject)
        subject_tokens = list(range(*subject_token_range))

        if knockout == TokenType.first:
            return [0]
        elif knockout == TokenType.last:
            return [last_idx]
        elif knockout == TokenType.subject:
            return subject_tokens
        elif knockout == TokenType.relation:
            return [i for i in range(last_idx + 1) if i not in subject_tokens]
        elif knockout == TokenType.context:
            return [i for i in range(subject_tokens[0])]
        elif knockout == TokenType.all:
            return list(range(last_idx + 1))
        elif knockout == TokenType.relation_minus_last:
            return [i for i in range(last_idx) if i not in subject_tokens]
        else:
            assert_never(knockout)


def get_num_to_masks(
    prompt: Prompt,
    tokenizer: TTokenizer,
    window: TWindow,
    knockout_source: TokenType,
    knockout_target: TokenType,
    device: TDevice,
) -> tuple[TNum2Mask, bool]:
    input_ids = prompt.input_ids(tokenizer, device)
    num_to_masks = TNum2Mask(defaultdict(list))
    first_token = False

    tok_start, tok_end = find_token_range(tokenizer, input_ids[0], prompt.subject)
    subject_tokens = list(range(tok_start, tok_end))
    if 0 in subject_tokens:
        first_token = True

    src_idx = prompt.get_knockout_idx(tokenizer, knockout_source, device)
    target_idx = prompt.get_knockout_idx(tokenizer, knockout_target, device)

    for layer in window:
        for src in src_idx:
            for target in target_idx:
                num_to_masks[layer].append((target, src))
    # print(knockout_source, knockout_target)
    # print('num_to_masks init', num_to_masks)

    return num_to_masks, first_token


def get_prompt_row_index(data: TPromptData, prompt_idx: TPromptOriginalIndex) -> Prompt:
    return Prompt(prompt_row=data.loc[prompt_idx])  # type: ignore


def get_subj_idx(
    input: str,
    subj: str,
    tokenizer: TTokenizer,
    last: bool = True,
) -> TTokenIndex:
    prefix = input.split(subj)[0]
    sent2subj = prefix
    if last:
        sent2subj = prefix + subj

    sent2subj_tokens = tokenizer(sent2subj)["input_ids"]
    return len(sent2subj_tokens) - 1  # type: ignore


def _get_logits(out, model_arch: MODEL_ARCH):
    match model_arch:
        case MODEL_ARCH.MAMBA2:
            logits, _ = out
        # TO ADD AN ARCH
        case (
            MODEL_ARCH.MAMBA1
            | MODEL_ARCH.LLAMA2
            | MODEL_ARCH.LLAMA3_2
            | MODEL_ARCH.GPT2
            | MODEL_ARCH.LLAMA3
            | MODEL_ARCH.MISTRAL0_1
            | MODEL_ARCH.MISTRAL0_3
            | MODEL_ARCH.QWEN2
            | MODEL_ARCH.QWEN2_5
        ):
            logits = out.logits
        case _:
            assert_never(model_arch)

    return logits


def generate_next_tokens(
    model: Any,
    input_ids: torch.Tensor,
    num_tokens_to_generate: int,
    model_arch: MODEL_ARCH,
):
    """
    Generate the next `num_tokens_to_generate` tokens and collect their logits for each input in the batch.

    Args:
        model: The language model (e.g., LLaMA) used for token generation.
        input_ids: Tokenized input IDs (torch.Tensor) with shape [batch_size, sequence_length].
        num_tokens_to_generate: Number of tokens to generate for each input in the batch.
        model_arch: The architecture of the model.
    Returns:
        all_logits: with shape [batch_size, num_tokens_to_generate, vocab_size].
        first_next_logits: with shape [batch_size, vocab_size].
        new_input_ids: with shape [batch_size, num_tokens_to_generate].
    """
    first_logits = None
    logits = None
    for _ in range(num_tokens_to_generate):
        with torch.no_grad():
            out = model(input_ids)
        logits = _get_logits(out, model_arch)
        next_tokens = get_last_token_logits(logits)
        if first_logits is None:
            first_logits = next_tokens

        input_ids = torch.cat([input_ids, torch.argmax(next_tokens, dim=-1, keepdim=True)], dim=-1)

    assert first_logits is not None
    return logits, first_logits, input_ids[:, -num_tokens_to_generate:]


def trim_left_and_right_pad(tensor, trim_value=2, pad_value=0):
    """
    Trims leading specified values from each row and pads rows on the right to equalize their lengths.

    Args:
        tensor (torch.Tensor): The input tensor.
        trim_value (int): The value to trim from the start of each row (default is 2).
        pad_value (int): The value to use for padding on the right (default is 0).

    Returns:
        torch.Tensor: The processed tensor with trimmed rows and right padding.
    """
    # Remove leading trim_value from each row
    trimmed_rows = [row[torch.nonzero(row != trim_value, as_tuple=True)[0][0] :] for row in tensor]

    # Determine the maximum length after trimming
    max_length = max(len(row) for row in trimmed_rows)

    # Pad each row from the right to make all rows equal to the max length
    padded_tensor = torch.stack(
        [torch.cat([row, torch.full((max_length - len(row),), pad_value)]) for row in trimmed_rows]
    )

    return padded_tensor


def _generate_few_tokens(model, tokenizer, input_ids, new_max_tokens):
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=input_ids.size(1) + new_max_tokens,
            num_return_sequences=1,
            top_k=1,
            temperature=1.0,
        )
    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return "".join(generated_text)
