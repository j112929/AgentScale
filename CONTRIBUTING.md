# Contributing

Use Python 3.11–3.13 and a virtual environment:

```bash
python -m pip install -e '.[dev,release]'
pytest -q -m 'not ray and not docker'
ruff check agentscale tests examples tools
mypy agentscale
```

Optional integration checks:

```bash
python -m pip install -e '.[ray]'
pytest -q -m ray
# Pull the exact image reference printed by the following command:
python -c 'from agentscale.sandbox import DEFAULT_IMAGE; print(DEFAULT_IMAGE)'
docker pull python@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
pytest -q -m docker
python tools/check_verl_contract.py --download --execute
```

Keep adapters optional. Never replace unavailable runtime measurements with zero or
fabricated success. Include resource/failure tests for control-plane changes. State
upstream versions, backend provenance, seeds, warm-up and incomplete milestones in
benchmark reports. GPU migration requires a runtime-owned admission fence and verified
safe-point implementation. Use pull requests and describe the behavior plus validation.

Contributions are submitted under Apache-2.0. Do not include credentials, model weights,
private task contents or external data whose redistribution terms are unknown.
