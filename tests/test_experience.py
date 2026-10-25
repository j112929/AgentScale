import pytest

from agentscale.experience import Experience, ExperienceBuffer, TrainingCoordinator


def sample(name="a", version=0):
    return Experience(name, version, 1, {"test": "passed"})


def test_bounded_experience_reservation_and_staleness():
    b = ExperienceBuffer(max_samples=2, max_staleness=0)
    b.push(sample())
    b.push(sample("b"))
    batch = b.take(1)
    with pytest.raises(OverflowError):
        b.push(sample("c"))
    with pytest.raises(ValueError):
        b.push(sample())
    b.commit(batch, 1)
    assert b.take(1) == () and b.stale_dropped == 1
    with pytest.raises(ValueError):
        b.push(sample("future", 2))
    b.push(sample("fresh", 1))
    batch = b.take(1)
    b.restore(batch)
    with pytest.raises(ValueError):
        b.restore(batch)
    assert b.take(1) == batch


class Hooks:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def call(self, name):
        self.calls.append(name)
        if self.fail == name:
            self.fail = None
            raise RuntimeError("injected " + name)

    def quiesce_rollouts(self):
        self.call("quiesce")

    def sleep_inference(self):
        self.call("sleep")

    def train(self, samples):
        self.call("train")

    def synchronize_weights(self, version):
        self.call("sync")

    def wake_inference(self):
        self.call("wake")

    def resume_rollouts(self):
        self.call("resume")


def test_training_commit_occurs_after_weight_sync_and_resume():
    b, h = ExperienceBuffer(), Hooks()
    b.push(sample())
    c = TrainingCoordinator(b, h)
    assert c.step(1)
    assert h.calls == ["quiesce", "sleep", "train", "sync", "wake", "resume"]
    assert b.policy_version == 1 and len(b) == 0
    assert not c.step(1)


@pytest.mark.parametrize("failure", ["quiesce", "sleep", "train", "sync", "wake", "resume"])
def test_failed_training_transaction_compensates_without_duplicate_optimizer_step(failure):
    b, h = ExperienceBuffer(), Hooks(failure)
    b.push(sample())
    c = TrainingCoordinator(b, h)
    with pytest.raises(RuntimeError):
        c.step(1)
    assert b.policy_version == 0
    if failure in ("quiesce", "sleep"):
        assert h.calls[-1] == "resume"
        assert len(b) == 1 and not c.faulted
    else:
        assert c.faulted
        assert h.calls[-1] == failure
        with pytest.raises(RuntimeError, match="reconciliation"):
            c.step(1)
