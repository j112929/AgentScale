"""Bounded, policy-versioned experience and explicit training/sync transactions."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Experience:
    sample_id: str
    policy_version: int
    reward: float
    payload: dict

    def __post_init__(self) -> None:
        if (
            not self.sample_id
            or type(self.policy_version) is not int
            or self.policy_version < 0
            or not math.isfinite(self.reward)
        ):
            raise ValueError("sample id, nonnegative policy version and finite reward required")


class ExperienceBuffer:
    def __init__(self, max_samples: int = 1024, max_staleness: int = 2):
        if (
            type(max_samples) is not int
            or max_samples < 1
            or type(max_staleness) is not int
            or max_staleness < 0
        ):
            raise ValueError("invalid experience limits")
        self.max_samples, self.max_staleness = max_samples, max_staleness
        self.policy_version = 0
        self._samples: deque[Experience] = deque()
        self._ids: set[str] = set()
        self._inflight: set[str] = set()
        self.stale_dropped = 0

    def push(self, sample: Experience) -> None:
        if sample.sample_id in self._ids or sample.policy_version > self.policy_version:
            raise ValueError("duplicate sample or future policy version")
        if len(self._ids) >= self.max_samples:
            raise OverflowError("experience buffer full")
        self._samples.append(sample)
        self._ids.add(sample.sample_id)

    def take(self, batch_size: int) -> tuple[Experience, ...]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch size must be positive")
        fresh = deque(
            s for s in self._samples if self.policy_version - s.policy_version <= self.max_staleness
        )
        for s in self._samples:
            if self.policy_version - s.policy_version > self.max_staleness:
                self._ids.remove(s.sample_id)
                self.stale_dropped += 1
        self._samples = fresh
        if len(fresh) < batch_size:
            return ()
        # IDs remain reserved while training is in progress.
        samples = tuple(self._samples.popleft() for _ in range(batch_size))
        self._inflight.update(s.sample_id for s in samples)
        return samples

    def commit(self, samples: tuple[Experience, ...], new_version: int) -> None:
        if type(new_version) is not int or new_version != self.policy_version + 1:
            raise ValueError("successful training must advance policy exactly once")
        self._check_inflight(samples)
        for sample in samples:
            self._ids.remove(sample.sample_id)
            self._inflight.remove(sample.sample_id)
        self.policy_version = new_version

    def restore(self, samples: tuple[Experience, ...]) -> None:
        self._check_inflight(samples)
        self._inflight.difference_update(s.sample_id for s in samples)
        self._samples.extendleft(reversed(samples))

    def _check_inflight(self, samples: tuple[Experience, ...]) -> None:
        ids = {s.sample_id for s in samples}
        if not samples or len(ids) != len(samples) or not ids.issubset(self._inflight):
            raise ValueError("batch does not match reserved in-flight samples")

    def __len__(self) -> int:
        return len(self._samples)


class TrainingHooks(Protocol):
    """Runtime-owned safe points; implementations must fence every rollout producer."""

    def quiesce_rollouts(self) -> None: ...
    def sleep_inference(self) -> None: ...
    def train(self, samples: tuple[Experience, ...]) -> None: ...
    def synchronize_weights(self, policy_version: int) -> None: ...
    def wake_inference(self) -> None: ...
    def resume_rollouts(self) -> None: ...


class TrainingCoordinator:
    """Experimental orchestration contract; no unverified GPU actuator is supplied."""

    def __init__(self, buffer: ExperienceBuffer, hooks: TrainingHooks):
        self.buffer, self.hooks = buffer, hooks
        self.faulted = False

    def step(self, batch_size: int) -> bool:
        if self.faulted:
            raise RuntimeError("training outcome requires operator reconciliation")
        samples = self.buffer.take(batch_size)
        if not samples:
            return False
        training_started = False
        quiesced = False
        sleeping = False
        try:
            # Mark attempted transitions before invoking hooks: a thrown call may be partial.
            quiesced = True
            self.hooks.quiesce_rollouts()
            sleeping = True
            self.hooks.sleep_inference()
            training_started = True
            self.hooks.train(samples)
            self.hooks.synchronize_weights(self.buffer.policy_version + 1)
            self.hooks.wake_inference()
            sleeping = False
            self.hooks.resume_rollouts()
            quiesced = False
            self.buffer.commit(samples, self.buffer.policy_version + 1)
            return True
        except Exception:
            # Do not retry a possibly executed optimizer update and silently train twice.
            self.faulted = training_started
            if not training_started:
                self.buffer.restore(samples)
            raise
        finally:
            try:
                if sleeping and not training_started:
                    self.hooks.wake_inference()
                if quiesced and not training_started:
                    self.hooks.resume_rollouts()
            except Exception:
                self.faulted = True
                raise
