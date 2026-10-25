"""Build normalized source/wheel artifacts and verify distribution metadata."""

import argparse
import gzip
import hashlib
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

EPOCH = 1792886400


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(EPOCH), PYTHONHASHSEED="0")
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(args.output)],
        env=env,
        check=True,
    )
    archives = sorted(args.output.glob("agentscale-*.tar.gz"))
    wheels = sorted(args.output.glob("agentscale-*.whl"))
    if len(archives) != 1 or len(wheels) != 1:
        raise RuntimeError("output directory must contain exactly one source archive and one wheel")
    archive = archives[0]
    output = io.BytesIO()
    with (
        tarfile.open(archive, "r:gz") as source,
        gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=EPOCH) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target,
    ):
        for member in sorted(source.getmembers(), key=lambda m: m.name):
            member.mtime, member.uid, member.gid = EPOCH, 0, 0
            member.uname, member.gname, member.pax_headers = "root", "root", {}
            stream = source.extractfile(member) if member.isfile() else None
            target.addfile(member, stream)
    archive.write_bytes(output.getvalue())
    subprocess.run(
        [sys.executable, "-m", "twine", "check", str(archive), str(wheels[0])], check=True
    )
    artifacts = [archive, wheels[0]]
    checksums = "\n".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name for p in artifacts
    )
    (args.output / "SHA256SUMS").write_text(checksums + "\n")
    print(checksums)


if __name__ == "__main__":
    main()
