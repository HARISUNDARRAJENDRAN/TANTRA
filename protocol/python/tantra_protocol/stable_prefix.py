from __future__ import annotations

from dataclasses import dataclass, field
import re
import time

_BOUNDARY = re.compile(r"(?:\s|[.!?।॥,:;])$")


@dataclass(slots=True)
class StablePrefixCommitter:
    """Commits only hypothesis prefixes that survived repeated ASR updates."""

    required_observations: int = 3
    minimum_stability_ms: int = 450
    _last_hypothesis: str = ""
    _stable_prefix: str = ""
    _stable_since_ms: int = 0
    _observations: int = 0
    _committed: str = ""

    @staticmethod
    def _lcp(left: str, right: str) -> str:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return left[:index]

    @staticmethod
    def _safe_boundary(prefix: str) -> int:
        for index in range(len(prefix), 0, -1):
            if _BOUNDARY.search(prefix[:index]):
                return index
        return 0

    def update(self, hypothesis: str, now_ms: int | None = None, final: bool = False) -> str:
        now_ms = now_ms if now_ms is not None else time.monotonic_ns() // 1_000_000
        hypothesis = " ".join(hypothesis.strip().split())
        common = self._lcp(self._last_hypothesis, hypothesis)

        if common == self._stable_prefix:
            self._observations += 1
        else:
            self._stable_prefix = common
            self._stable_since_ms = now_ms
            self._observations = 1

        self._last_hypothesis = hypothesis
        candidate = hypothesis if final else self._stable_prefix
        if not final:
            stable_long_enough = now_ms - self._stable_since_ms >= self.minimum_stability_ms
            if self._observations < self.required_observations and not stable_long_enough:
                return ""
            candidate = candidate[: self._safe_boundary(candidate)]

        if len(candidate) <= len(self._committed):
            return ""
        emitted = candidate[len(self._committed) :]
        self._committed = candidate
        return emitted.lstrip()

    def reset(self) -> None:
        self._last_hypothesis = ""
        self._stable_prefix = ""
        self._stable_since_ms = 0
        self._observations = 0
        self._committed = ""
