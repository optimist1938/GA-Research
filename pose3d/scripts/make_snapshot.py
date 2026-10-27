"""Write the repo snapshot that an offline Kaggle kernel runs: `ga-research-<commit>.tar.gz`.

Kaggle GPU kernels with the RTX Pro 6000 have no internet, so they cannot `git clone` the branch.
The runner notebooks (notebooks/i2s-symsol-runner.ipynb, notebooks/symsol-stf-runner.ipynb) instead
look for a Kaggle dataset holding either an auto-extracted folder that contains `pose3d/pose3d/config.py`
or a tarball named `ga-research-*.tar.gz`, extract it to /kaggle/working and run `python -m pose3d` from
`/kaggle/working/GA-Research/pose3d`. This script writes that tarball.

What goes in: every file git tracks at HEAD of the current checkout (read from the working tree, so the
tarball is what you have on disk), under the top folder `GA-Research/`, minus files larger than
--max-mb (default 5, which drops stray binaries and keeps all code, tests, notebooks and reports) and
minus `awesome-reference/` (reading material, not needed to run). A file `GA-Research/SNAPSHOT_COMMIT`
records the commit. A dirty working tree is refused unless --allow-dirty, so the tarball always matches
a commit that can be named in a report.

    python pose3d/scripts/make_snapshot.py [--out DIR] [--max-mb 5] [--allow-dirty]

Then upload the tarball as (a new version of) a private Kaggle dataset, e.g. with
`kaggle datasets version -p <folder holding the tarball> -m "<commit>"`, and attach it to the kernel.
"""

import argparse
import io
import os
import subprocess
import sys
import tarfile


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=".", help="folder to write ga-research-<commit>.tar.gz into")
    parser.add_argument("--max-mb", type=float, default=5.0, help="skip tracked files larger than this")
    parser.add_argument("--allow-dirty", action="store_true", help="snapshot even with uncommitted changes")
    args = parser.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    root = git("rev-parse", "--show-toplevel", cwd=here).strip()
    commit = git("rev-parse", "--short=10", "HEAD", cwd=root).strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=root).strip()
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=root).strip()
    if dirty and not args.allow_dirty:
        sys.exit("uncommitted changes (commit them, or pass --allow-dirty):\n" + dirty)

    files = git("ls-files", "-z", cwd=root).split("\0")
    out = os.path.join(args.out, f"ga-research-{commit}.tar.gz")
    os.makedirs(args.out, exist_ok=True)
    kept = skipped = 0
    with tarfile.open(out, "w:gz") as tar:
        for rel in files:
            path = os.path.join(root, rel)
            if not rel or rel.startswith("pose3d/awesome-reference/") or not os.path.isfile(path):
                continue
            if os.path.getsize(path) > args.max_mb * 2**20:
                skipped += 1
                continue
            tar.add(path, arcname=f"GA-Research/{rel}")
            kept += 1
        note = f"{commit} ({branch}){' + uncommitted changes' if dirty else ''}\n".encode()
        info = tarfile.TarInfo("GA-Research/SNAPSHOT_COMMIT")
        info.size = len(note)
        tar.addfile(info, io.BytesIO(note))
    print(f"{out}: {kept} files from {commit} ({branch}), {skipped} larger than {args.max_mb} MB skipped, "
          f"{os.path.getsize(out) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
