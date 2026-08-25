"""Install the license-gated SMPL-X / SMPL body models from their download zips.

The two archives cannot be fetched automatically -- both require a registration
and an accepted licence -- but everything AFTER the download is mechanical and
easy to get subtly wrong:

* the zips nest the models several directories deep, differently from each other;
* SMPL ships its files as ``basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`` while
  every consumer in this pipeline expects ``SMPL_NEUTRAL.pkl``;
* a wrong layout does not fail here, it fails much later inside the pose stage
  with a SetupError that names a path rather than the mistake.

So this script extracts, renames to the canonical layout, and then VALIDATES by
actually loading each file and checking it carries the arrays a body model must
have. A truncated download is otherwise indistinguishable from a good one until
inference crashes.

Usage:
    python scripts/00b_install_body_models.py                    # scan ~/Downloads
    python scripts/00b_install_body_models.py a.zip b.zip        # explicit
    python scripts/00b_install_body_models.py --dest data/body_models
"""
from __future__ import annotations

import argparse
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

# Canonical destination name -> substrings identifying the member inside a zip.
# Matching is on the LOWERCASED member name and every substring must appear, so
# these stay robust to the archives' differing directory nesting.
_SMPLX_WANTED = {
    "smplx/SMPLX_NEUTRAL.npz": ("smplx_neutral.npz",),
    "smplx/SMPLX_MALE.npz": ("smplx_male.npz",),
    "smplx/SMPLX_FEMALE.npz": ("smplx_female.npz",),
}
_SMPL_WANTED = {
    "smpl/SMPL_NEUTRAL.pkl": ("basicmodel_neutral", ".pkl"),
    "smpl/SMPL_MALE.pkl": ("basicmodel_m", ".pkl"),
    "smpl/SMPL_FEMALE.pkl": ("basicmodel_f", ".pkl"),
}

# Every body model carries these, whatever the format. Checking them catches a
# truncated or wrong-file download, which a size check alone does not.
_REQUIRED_ARRAYS = ("v_template", "shapedirs", "J_regressor")



class _Stub:
    """Stands in for a class the reader does not have installed."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _StubUnpickler(pickle.Unpickler):
    """Read a pickle without importing the classes it references.

    The SMPL .pkl files are built on chumpy, an unmaintained package that does
    not install on current Pythons. Requiring it just to CHECK a download would
    trade one gated obstacle for another, and installing an old package to
    execute its constructors on a freshly downloaded file is the wrong trade.
    Stubbing every unresolvable class lets the container structure -- which is
    all this check needs -- be read safely.
    """

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Stub


def _members(zf: zipfile.ZipFile):
    return [m for m in zf.namelist() if not m.endswith("/")]


def _find(zf: zipfile.ZipFile, needles) -> str | None:
    for member in _members(zf):
        low = member.lower()
        if all(n in low for n in needles):
            return member
    return None


def _extract(zf: zipfile.ZipFile, member: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest, "wb") as out:
        data = src.read()
        out.write(data)
    return len(data)


def validate(path: Path) -> str:
    """Load a body model and confirm it carries the arrays one must have."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as handle:
            keys = set(handle.files)
    else:
        with open(path, "rb") as handle:
            obj = _StubUnpickler(handle, encoding="latin1").load()
        keys = set(obj.keys()) if isinstance(obj, dict) else set(dir(obj))

    missing = [k for k in _REQUIRED_ARRAYS if k not in keys]
    if missing:
        raise SystemExit(
            f"{path} loaded but is missing {missing}. That usually means the "
            "wrong file was matched, or the download is truncated -- re-download "
            "and re-run.")
    return f"{path.name}: ok ({len(keys)} arrays, {path.stat().st_size/1e6:.1f} MB)"


def install(zips, dest: Path) -> list[str]:
    done, notes = {}, []
    for zip_path in zips:
        if not zipfile.is_zipfile(zip_path):
            notes.append(f"skip (not a zip): {zip_path}")
            continue
        with zipfile.ZipFile(zip_path) as zf:
            for wanted in (_SMPLX_WANTED, _SMPL_WANTED):
                for rel, needles in wanted.items():
                    if rel in done:
                        continue
                    member = _find(zf, needles)
                    if member is None:
                        continue
                    size = _extract(zf, member, dest / rel)
                    done[rel] = size
                    notes.append(f"{zip_path.name}: {member} -> {rel} "
                                 f"({size/1e6:.1f} MB)")
    return notes, done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zips", nargs="*", type=Path,
                        help="downloaded archives; default scans ~/Downloads")
    parser.add_argument("--dest", type=Path, default=Path("data/body_models"))
    args = parser.parse_args()

    zips = args.zips
    if not zips:
        downloads = Path.home() / "Downloads"
        zips = sorted(p for p in downloads.glob("*.zip")
                      if any(t in p.name.lower()
                             for t in ("smpl", "models_smplx")))
        if not zips:
            print(f"no SMPL/SMPL-X zips found in {downloads}. Pass paths "
                  "explicitly, or download them first (SETUP.md section 2).",
                  file=sys.stderr)
            return 2
        print(f"found {len(zips)} archive(s) in {downloads}")

    notes, done = install(zips, args.dest)
    for note in notes:
        print("  " + note)

    print("\nvalidating:")
    required = ["smplx/SMPLX_NEUTRAL.npz", "smpl/SMPL_NEUTRAL.pkl"]
    missing = []
    for rel in list(_SMPLX_WANTED) + list(_SMPL_WANTED):
        path = args.dest / rel
        if path.exists():
            print("  " + validate(path))
        elif rel in required:
            missing.append(rel)
        else:
            print(f"  {rel}: absent (optional -- only NEUTRAL is required)")

    if missing:
        print(f"\nMISSING REQUIRED: {missing}", file=sys.stderr)
        print("The pose stage checks data/body_models/smplx/SMPLX_NEUTRAL.npz "
              "at startup. See SETUP.md section 2.", file=sys.stderr)
        return 1
    print("\nbody models installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
