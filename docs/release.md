# v0.1.0rc1 release procedure

This is a CPU/container-validated research release candidate. A stable release must
not imply that GPU inference, verl training, GPU sharing or native verl benchmarks
have been executed when they have not.

```bash
python -m pip install -e '.[dev,release]' 'setuptools>=77'
pytest -q -m 'not ray and not docker'
ruff check agentscale tests examples tools
mypy agentscale
python tools/build_release.py --output dist
```

Run real Ray and Docker jobs as documented in CONTRIBUTING.md. CI builds a wheel,
source archive and SHA256SUMS artifact, and separately runs container isolation and
upstream tool-contract validation. The package has no mandatory third-party runtime
dependencies; Ray, verl and vLLM remain optional deployment choices.

The build script fixes SOURCE_DATE_EPOCH/PYTHONHASHSEED, normalizes source tar/gzip
metadata and checks distribution metadata with Twine. To check reproducibility, build
twice with the same Python/build-tool versions into two empty directories and compare
artifact hashes. Source archives include tests, configs, examples, docs and benchmark
records; wheels include the Python package and license metadata. Bit identity across
different setuptools/build versions is not promised.

Do not publish a stable `v0.1.0` tag until the documented candidate checks pass and
maintainers accept its explicit experimental-runtime scope. Candidate artifacts may
be published as a prerelease. GitHub artifacts/releases do not upload to PyPI; PyPI
publishing requires separately configured trusted publishing and is not enabled here.

## Dependency and distribution scope

AgentScale uses Apache-2.0. The optional Ray 2.50.1 runtime is Apache-2.0; the verl
and vLLM integrations refer to Apache-2.0 upstream APIs. The Docker Python image has
its own Python/Debian component licenses and is pulled separately, never embedded
in the wheel or source distribution. Development tools are not vendored. Model
weights and task datasets are not distributed. Review image/model dependencies and
their license notices when building a downstream runtime distribution.
