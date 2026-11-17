"""Build the 10x PARRM-artifact recording variant and its feature caches.

Run from the project root:
    python Task3/task3_make_semisynthetic_h5_v1.py

Only stimulation-ON recordings are modified. Original exports and PARRM files
are read-only. See semi_synthetic/parrm_amp10/README.md after generation.
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

DEFAULT_SCALE = 10.0
CHUNK_SAMPLES = 25000
METADATA_FIELDS = (
    "session", "split", "condition", "am_freq_hz", "off_reference",
    "artifact_scale", "construction", "modified_channels", "total_channels",
    "samples", "lfp_fs_hz", "reconstruction_relative_rms_error",
    "original_export", "parrm_estimate", "synthetic_export",
    "base_features", "hf_features",
)

README = """# PARRM artifact amplification: 10×

These are modified versions of the stimulation-ON recordings from the Butters dataset. The 3 Hz and 20 Hz AM conditions use the same transformation. No movement labels or trial timings were changed.

For the 35 fixed channels used by the decoder, each sample is computed as:

    modified LFP = recorded LFP + 9 × PARRM-estimated artifact

The 10× figure refers to the estimated artifact component, not the stimulation current or the full LFP. The other channels in each 64-channel export are left as recorded. The files have the same datasets and metadata as the original exports, plus attributes documenting the modification.

The scale was chosen after a validation sweep. Relative to the recorded stimulation-ON signals (1×), 10× lowered mean test velocity R² by about 0.028 for both 3 Hz and 20 Hz AM with the fixed GRU. This is a semi-synthetic challenge condition, not a recording from stronger stimulation.

## Files

- `h5_export_outputs/<session>.h5`: modified stimulation-ON recordings; unchanged HDF5 layout.
- `metadata.csv`: source files, split, stimulation frequency, scale, and reconstruction check for each session.
- `features/task3_parrm_amp_s0010p000/`: LMP and seven 1–200 Hz power bands from the modified recordings.
- `features/task3_parrm_amp_hf_s0010p000/`: additional 200–400 Hz power for the GRU.

The last two paths are relative to the project root, not this README. The CSV contains the actual project-relative file paths.

## Using the files

Keep `splits.json`, `EECS182_data_manifest.csv`, trial CSVs, and `kinematics_v3/bins50ms` unchanged. Use the synthetic HDF5 only for stimulation-ON sessions. Stimulation-OFF sessions continue to use `export_outputs/h5_export_outputs`.

The current data loader does not automatically search both directories. Resolve each session path before loading it:

```python
from pathlib import Path

def recording_path(root, session):
    root = Path(root)
    modified = root / "semi_synthetic/parrm_amp10/h5_export_outputs" / f"{session}.h5"
    if modified.is_file():
        return modified
    return root / "export_outputs/h5_export_outputs" / f"{session}.h5"
```

For comparisons with the original GRU, use the fixed 35-channel mask and the existing calibration/query trial split. The feature caches contain unnormalized values; normalization must be fitted from the calibration pool, not the query labels.

Do not treat the original `parrm_outputs` files as the result of applying PARRM anew to these modified recordings. Re-run PARRM on the modified input if that specific baseline is needed.

This dataset is separate from the planned shift generator that adds stimulation templates to OFF trials for in-context model training. It provides modified real stimulation-ON sessions for evaluation, not new OFF-derived training episodes.
"""


def project_root():
    return Path(__file__).resolve().parent.parent


def tag(scale):
    return f"s{scale:08.3f}".replace(".", "p")


def attr_scalar(value):
    values = np.asarray(value)
    if values.size != 1:
        raise ValueError(f"Expected one scalar attribute, got shape {values.shape}")
    return float(values.reshape(-1)[0])


def source_paths(root, session):
    original = root / "export_outputs" / "h5_export_outputs" / f"{session}.h5"
    choices = [root / "parrm_outputs" / "parrm_h5" / f"{session}.h5",
               root / "parrm_outputs" / f"{session}.h5"]
    parrm = next((path for path in choices if path.is_file()), None)
    if not original.is_file():
        raise FileNotFoundError(f"Missing source export: {original}")
    if parrm is None:
        raise FileNotFoundError(f"Missing PARRM export for {session}; checked {choices}")
    return original, parrm


def select_sessions(root, split):
    import json
    with (root / "splits.json").open() as fh:
        splits = json.load(fh)
    wanted = {"val-test": ("val", "test"), "all": ("train", "val", "test")}
    selections = wanted.get(split, (split,))
    with (root / "EECS182_data_manifest.csv").open(newline="") as fh:
        manifest = {row["session"]: row for row in csv.DictReader(fh)}
    entries = []
    for part in selections:
        for session in splits["sessions"][part]:
            if session not in manifest:
                raise ValueError(f"Session not in manifest: {session}")
            row = manifest[session]
            if row["condition"] not in ("am3hz", "am20hz"):
                continue
            entries.append((session, part, row, splits.get("off_reference", {}).get(session) or ""))
    if not entries:
        raise ValueError(f"No stimulation-ON recordings for {split}")
    return entries


def verify_input(src, parrm, session):
    if "lfp" not in src or "lfp" not in parrm or "artifact" not in parrm:
        raise ValueError(f"{session}: missing /lfp or PARRM /artifact")
    original = src["lfp"]
    clean = parrm["lfp"]
    artifact = parrm["artifact"]
    if original.ndim != 2 or original.shape != clean.shape or original.shape != artifact.shape:
        raise ValueError(f"{session}: LFP/PARRM shape mismatch")
    if original.shape[0] != 64:
        raise ValueError(f"{session}: expected 64-channel (channels, samples) export")
    if not np.issubdtype(original.dtype, np.floating):
        raise ValueError(f"{session}: expected floating-point LFP, found {original.dtype}")
    if "t" not in src or len(src["t"]) != original.shape[1]:
        raise ValueError(f"{session}: /t does not match /lfp samples")
    channels = np.asarray(src["channel_ids"][:], dtype=int)
    keep = np.asarray(src["channel_keep_fixed"][:], dtype=bool)
    if len(keep) != 64 or len(channels) != 64 or keep.sum() != 35:
        raise ValueError(f"{session}: expected 35 fixed channels within 64 channels")
    if "channel_ids" in parrm and not np.array_equal(channels, parrm["channel_ids"][:]):
        raise ValueError(f"{session}: export/PARRM channel IDs differ")
    fs = attr_scalar(src.attrs["lfp_fs"])
    if not np.isfinite(fs) or fs <= 800:
        raise ValueError(f"{session}: sampling rate {fs} is invalid for the 200–400 Hz band")
    return np.flatnonzero(keep), channels, fs


def make_recording(root, session, scale, tolerance, overwrite):
    source, parrm_path = source_paths(root, session)
    folder = root / "semi_synthetic" / "parrm_amp10" / "h5_export_outputs"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{session}.h5"
    if dest.exists() and not overwrite:
        with h5py.File(dest, "r") as existing:
            previous = attr_scalar(existing.attrs["semi_synthetic_artifact_scale"])
            if not np.isclose(previous, scale):
                raise ValueError(f"Existing file has wrong scale: {dest}")
            if existing.attrs.get("semi_synthetic_method") != "parrm_artifact_addition":
                raise ValueError(f"Existing output is not from this generator: {dest}")
            error = attr_scalar(existing.attrs["semi_synthetic_reconstruction_relative_rms_error"])
            if error > tolerance:
                raise ValueError(f"Previous reconstruction check failed: {dest}")
            return dest, source, parrm_path, dict(
                modified_channels=len(existing.attrs["semi_synthetic_channels"]),
                total_channels=existing["lfp"].shape[0],
                samples=existing["lfp"].shape[1],
                fs=attr_scalar(existing.attrs["lfp_fs"]), error=error), False

    temp = dest.with_suffix(".h5.partial")
    if temp.exists():
        temp.unlink()
    try:
        # Preserve every HDF5 dataset and attribute, then modify only kept channels.
        shutil.copyfile(source, temp)
        with h5py.File(source, "r") as src, h5py.File(parrm_path, "r") as parrm, \
                h5py.File(temp, "r+") as output:
            keep, channels, fs = verify_input(src, parrm, session)
            raw = src["lfp"]
            clean = parrm["lfp"]
            artifact = parrm["artifact"]
            n = raw.shape[1]
            ss_error = 0.0
            ss_signal = 0.0
            for start in range(0, n, CHUNK_SAMPLES):
                stop = min(n, start + CHUNK_SAMPLES)
                # Fancy-index reads are ordered and small enough to fit in memory.
                a = np.asarray(artifact[keep, start:stop], dtype=np.float64)
                x = np.asarray(raw[keep, start:stop], dtype=np.float64)
                c = np.asarray(clean[keep, start:stop], dtype=np.float64)
                if not (np.isfinite(x).all() and np.isfinite(a).all() and np.isfinite(c).all()):
                    raise ValueError(f"{session}: non-finite PARRM or LFP values")
                ss_error += np.square(x - c - a).sum(dtype=np.float64)
                ss_signal += np.square(x).sum(dtype=np.float64)
                augmented = (x + (scale - 1.0) * a).astype(raw.dtype)
                if not np.isfinite(augmented).all():
                    raise ValueError(f"{session}: scaled LFP contains non-finite values")
                output["lfp"][keep, start:stop] = augmented
            error = float(np.sqrt(ss_error / max(ss_signal, 1e-24)))
            if error > tolerance:
                raise ValueError(f"{session}: exported LFP != cleaned + artifact "
                                 f"({error:.2%} relative RMS; tolerance {tolerance:.2%})")
            output.attrs["semi_synthetic_method"] = "parrm_artifact_addition"
            output.attrs["semi_synthetic_artifact_scale"] = scale
            output.attrs["semi_synthetic_formula"] = "recorded_lfp + (scale - 1) * parrm_artifact"
            output.attrs["semi_synthetic_channels"] = channels[keep]
            output.attrs["semi_synthetic_reconstruction_relative_rms_error"] = error
            output.attrs["semi_synthetic_source_export"] = str(source.relative_to(root))
            output.attrs["semi_synthetic_source_parrm"] = str(parrm_path.relative_to(root))
            output.flush()
        os.replace(temp, dest)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    return dest, source, parrm_path, dict(modified_channels=len(keep),
                total_channels=64, samples=n, fs=fs, error=error), True


def feature_paths(root, session, scale):
    t = tag(scale)
    base = root / "features" / f"task3_parrm_amp_{t}" / f"{session}.h5"
    hf = root / "features" / f"task3_parrm_amp_hf_{t}" / f"{session}.h5"
    return base, hf


def make_features(root, session, dest, scale, overwrite=False):
    project = str(project_root())
    if project not in sys.path:
        sys.path.insert(0, project)
    # Use the same LMP, band-power windows and sample alignment as the GRU sweep.
    import EECS182_NHP_data_loader as loader
    import EECS182_NHP_features as features
    base, hf = feature_paths(root, session, scale)
    if base.exists() and hf.exists() and not overwrite:
        with h5py.File(base, "r") as b, h5py.File(hf, "r") as h:
            if not (np.isclose(attr_scalar(b.attrs["artifact_scale"]), scale) and
                    np.isclose(attr_scalar(h.attrs["artifact_scale"]), scale)):
                raise ValueError(f"Existing feature cache has wrong artifact scale: {session}")
            if not np.array_equal(b["channel_ids"][:], h["channel_ids"][:]):
                raise ValueError(f"Feature caches have different channel order: {session}")
        return base, hf, False
    with h5py.File(dest, "r") as source:
        channels = np.asarray(source["channel_ids"][:], dtype=int)
        keep = np.asarray(source["channel_keep_fixed"][:], dtype=bool)
        lfp = np.asarray(source["lfp"][np.flatnonzero(keep), :], dtype=np.float32)
        t = source["t"][:]
        fs = attr_scalar(source.attrs["lfp_fs"])
    bins = np.arange(int(np.ceil(t[0] / loader.BIN_S - 1e-9)),
                     int(np.floor(t[-1] / loader.BIN_S + 1e-9)))
    ends = np.searchsorted(t, (bins + 1) * loader.BIN_S - 0.5 / fs) - 1
    session_data = SimpleNamespace(lfp=lfp, t=t, fs=fs, cache={})
    lmp = loader.bin_lmp(session_data, bins)
    bands = np.stack([features._band_power(lfp, ends, fs, win, low, high)
                      for low, high, win in features.BANDS], axis=1)
    hfpower = features._band_power(lfp, ends, fs, 0.256, 200, 400)
    base.parent.mkdir(parents=True, exist_ok=True)
    hf.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = base.with_suffix(".h5.partial")
    tmp_hf = hf.with_suffix(".h5.partial")
    # save_session uses its own atomic .tmp extension; write there then replace.
    try:
        base_payload = dict(session=session, bin_id=bins, lmp=lmp, logpow=bands,
                            channels=channels[keep], fs=fs,
                            source=f"semi_synthetic_PARRM_artifact_scale_{scale:g}",
                            reference="none")
        features.save_session(base_payload, str(tmp_base))
        with h5py.File(tmp_base, "a") as file:
            file.attrs["artifact_scale"] = scale
            file.attrs["reconstruction_formula"] = "raw + (artifact_scale - 1) * PARRM_artifact"
        with h5py.File(tmp_hf, "w") as file:
            file.create_dataset("bin_id", data=bins)
            file.create_dataset("channel_ids", data=channels[keep])
            file.create_dataset("logpow", data=hfpower, compression="gzip", compression_opts=1)
            file.attrs["artifact_scale"] = scale
            file.attrs["band_hz"] = (200, 400)
            file.attrs["reconstruction_formula"] = "raw + (artifact_scale - 1) * PARRM_artifact"
        os.replace(tmp_base, base)
        os.replace(tmp_hf, hf)
    except Exception:
        for p in (tmp_base, tmp_hf, Path(str(tmp_base) + ".tmp")):
            if p.exists():
                p.unlink()
        raise
    return base, hf, True


def rel(path, root):
    return str(path.relative_to(root))


def update_metadata(folder, rows):
    target = folder / "metadata.csv"
    existing = {}
    if target.exists():
        with target.open(newline="") as fh:
            for row in csv.DictReader(fh):
                existing[row["session"]] = row
    for row in rows:
        existing[row["session"]] = row
    temp = folder / "metadata.csv.partial"
    with temp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(existing.values(), key=lambda r: (r["split"], r["session"])))
    os.replace(temp, target)
    readme = folder / "README.md"
    if not readme.exists():
        readme.write_text(README, encoding="utf-8")
    return target


def build(args):
    root = args.root.expanduser().resolve()
    if not root.is_dir() or not (root / "splits.json").is_file():
        raise SystemExit(f"Not a project root: {root}")
    if not np.isclose(args.scale, DEFAULT_SCALE):
        raise SystemExit("This release packages the validated 10x condition only")
    entries = select_sessions(root, args.split)
    folder = root / "semi_synthetic" / "parrm_amp10"
    records = []
    print(f"Selected {len(entries)} ON sessions from {args.split}")
    for ix, (session, part, manifest_row, off_reference) in enumerate(entries, 1):
        dest, original, parrm, info, created = make_recording(
            root, session, args.scale, args.tolerance, args.overwrite)
        base, hf = feature_paths(root, session, args.scale)
        features_made = False
        if not args.h5_only:
            base, hf, features_made = make_features(root, session, dest,
                                                   args.scale, args.overwrite)
        records.append(dict(
            session=session, split=part, condition=manifest_row["condition"],
            am_freq_hz=manifest_row["am_freq_hz"], off_reference=off_reference,
            artifact_scale=f"{args.scale:g}",
            construction="recorded_lfp + (scale - 1) * parrm_artifact",
            modified_channels=info["modified_channels"],
            total_channels=info["total_channels"], samples=info["samples"],
            lfp_fs_hz=f"{info['fs']:.6f}",
            reconstruction_relative_rms_error=f"{info['error']:.9g}",
            original_export=rel(original, root), parrm_estimate=rel(parrm, root),
            synthetic_export=rel(dest, root),
            base_features=rel(base, root) if base.exists() else "",
            hf_features=rel(hf, root) if hf.exists() else "",
        ))
        print(f"  {ix}/{len(entries)} {session}: H5 {'written' if created else 'exists'}, "
              f"features {'written' if features_made else ('exists' if base.exists() and hf.exists() else 'skipped')}, "
              f"PARRM check {info['error']:.3%}", flush=True)
        # Preserve metadata for completed sessions if a subsequent session fails.
        update_metadata(folder, [records[-1]])
    print(f"\nData: {folder / 'h5_export_outputs'}")
    print(f"Metadata: {folder / 'metadata.csv'}")
    print(f"README: {folder / 'README.md'}")
    if args.h5_only:
        print("Features skipped (--h5-only)")


def self_test():
    project = str(project_root())
    if project not in sys.path:
        sys.path.insert(0, project)
    import json
    import tempfile
    import EECS182_NHP_features as features
    import EECS182_NHP_data_loader as loader
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "export_outputs/h5_export_outputs").mkdir(parents=True)
        (root / "parrm_outputs/parrm_h5").mkdir(parents=True)
        (root / "EECS182_data_manifest.csv").write_text(
            "session,condition,am_freq_hz\nToy-3Hz,am3hz,3\nToy-20Hz,am20hz,20\n", encoding="utf-8")
        (root / "splits.json").write_text(json.dumps({
            "sessions": {"train": [], "val": ["Toy-3Hz"], "test": ["Toy-20Hz"]},
            "off_reference": {} }), encoding="utf-8")
        rng = np.random.default_rng(2026)
        for session in ("Toy-3Hz", "Toy-20Hz"):
            n = 2000
            raw = rng.normal(size=(64, n)).astype(np.float32)
            art = rng.normal(size=(64, n)).astype(np.float32) * 0.1
            clean = raw - art
            channels = np.arange(1, 65)
            kept = np.arange(64) < 35
            with h5py.File(root / "export_outputs/h5_export_outputs" / f"{session}.h5", "w") as f:
                f.create_dataset("lfp", data=raw)
                f.create_dataset("channel_ids", data=channels)
                f.create_dataset("channel_keep_fixed", data=kept)
                f.create_dataset("t", data=np.arange(n) / 1000)
                f.create_dataset("stim/envelope", data=np.ones(n))
                f.attrs["lfp_fs"] = np.array([1000.])
            with h5py.File(root / "parrm_outputs/parrm_h5" / f"{session}.h5", "w") as f:
                f.create_dataset("lfp", data=clean)
                f.create_dataset("artifact", data=art)
                f.create_dataset("channel_ids", data=channels)
        args = argparse.Namespace(root=root, scale=10., split="val-test",
                                  tolerance=.03, overwrite=False, h5_only=False)
        build(args)
        with h5py.File(root / "semi_synthetic/parrm_amp10/h5_export_outputs/Toy-3Hz.h5", "r") as f:
            with h5py.File(root / "export_outputs/h5_export_outputs/Toy-3Hz.h5", "r") as o, \
                    h5py.File(root / "parrm_outputs/parrm_h5/Toy-3Hz.h5", "r") as p:
                np.testing.assert_allclose(f["lfp"][:35], o["lfp"][:35] + 9*p["artifact"][:35], atol=1e-6)
                np.testing.assert_array_equal(f["lfp"][35:], o["lfp"][35:])
                np.testing.assert_array_equal(f["stim/envelope"][:], o["stim/envelope"][:])
        base, hf = feature_paths(root, "Toy-3Hz", 10.)
        with h5py.File(base) as f, h5py.File(hf) as g:
            assert f["logpow"].shape[1] == len(features.BANDS)
            assert g["logpow"].shape[0] == 35
            assert len(f["bin_id"]) == 39
            np.testing.assert_array_equal(f["bin_id"][:], g["bin_id"][:])
        build(args)
        with (root / "semi_synthetic/parrm_amp10/metadata.csv").open() as fh:
            assert len(list(csv.DictReader(fh))) == 2
    print("Self-test passed: full HDF5 copy, artifact formula, feature alignment, and metadata")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--split", choices=("val-test", "train", "val", "test", "all"),
                        default="val-test")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--tolerance", type=float, default=0.03,
                        help="maximum relative RMS error in raw = clean + artifact")
    parser.add_argument("--h5-only", action="store_true", help="skip the LFP feature caches")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        build(args)


if __name__ == "__main__":
    main()
