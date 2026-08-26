from __future__ import annotations

import torch
from torch import Tensor


def ctc_viterbi_alignment(log_probabilities: Tensor, targets: Tensor, blank_id: int = 0) -> Tensor:
    """Return one CTC state per frame using Viterbi dynamic programming.

    `log_probabilities` is `[frames, vocab]`; `targets` is `[tokens]`.
    """
    if log_probabilities.ndim != 2 or targets.ndim != 1:
        raise ValueError("Expected [T,V] log probabilities and [L] targets")
    frames = log_probabilities.shape[0]
    extended = torch.full((targets.numel() * 2 + 1,), blank_id, dtype=torch.long, device=targets.device)
    extended[1::2] = targets
    states = extended.numel()
    score = torch.full((frames, states), float("-inf"), device=log_probabilities.device)
    back = torch.zeros((frames, states), dtype=torch.int16, device=log_probabilities.device)
    score[0, 0] = log_probabilities[0, blank_id]
    if states > 1:
        score[0, 1] = log_probabilities[0, extended[1]]
    for time in range(1, frames):
        for state in range(states):
            candidates = [(score[time - 1, state], state)]
            if state > 0:
                candidates.append((score[time - 1, state - 1], state - 1))
            if state > 1 and extended[state] != blank_id and extended[state] != extended[state - 2]:
                candidates.append((score[time - 1, state - 2], state - 2))
            best_score, previous = max(candidates, key=lambda item: float(item[0]))
            score[time, state] = best_score + log_probabilities[time, extended[state]]
            back[time, state] = previous
    state = states - 1
    if states > 1 and score[-1, states - 2] > score[-1, state]:
        state = states - 2
    path = torch.empty(frames, dtype=torch.long, device=log_probabilities.device)
    for time in range(frames - 1, -1, -1):
        path[time] = state
        if time:
            state = int(back[time, state])
    return path


def token_durations_from_path(path: Tensor, target_count: int) -> Tensor:
    durations = torch.zeros(target_count, dtype=torch.long, device=path.device)
    for token_index in range(target_count):
        durations[token_index] = (path == (token_index * 2 + 1)).sum()
    return durations.clamp_min(1)
