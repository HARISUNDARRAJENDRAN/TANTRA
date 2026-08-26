from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ..text.vocabulary import Vocabulary


def greedy_ctc_ids(logits: Tensor, lengths: Tensor, blank_id: int = 0) -> list[list[int]]:
    predictions = logits.argmax(dim=-1)
    decoded: list[list[int]] = []
    for row, length in zip(predictions, lengths, strict=True):
        previous = -1
        output: list[int] = []
        for token in row[: int(length)].tolist():
            if token != blank_id and token != previous:
                output.append(token)
            previous = token
        decoded.append(output)
    return decoded


def greedy_ctc_text(logits: Tensor, lengths: Tensor, vocabulary: Vocabulary) -> list[str]:
    return [vocabulary.decode(ids) for ids in greedy_ctc_ids(logits, lengths)]
