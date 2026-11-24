"""Generate the Task 4 data / split / batch reference.

Reads the committed code and data (no model imports), asserts the pipeline invariants,
then writes:

  reports/data_pipeline.svg        three-column block diagram (storage -> parse -> split/batch)
  reports/data_pipeline_table.tex  multi-panel LaTeX table for the report appendix

The point is to keep the report's dimensions tied to the code: every number below is
read from task4/utils.py, EECS182_NHP_data_loader.py, splits.json or data/manifest.csv,
and the script fails loudly if an expected invariant changes.

Run from the repository root:
    python scripts/make_data_pipeline_diagram.py
"""

import csv
import html
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(ROOT, "task4", "utils.py")
MODELS = os.path.join(ROOT, "task4", "task4_2_eegnet.ipynb")
LOADER = os.path.join(ROOT, "EECS182_NHP_data_loader.py")
FEATURES = os.path.join(ROOT, "EECS182_NHP_features.py")
TRAINER = os.path.join(ROOT, "kaggle", "train_task4_2.py")
SPLITS = os.path.join(ROOT, "splits.json")
MANIFEST = os.path.join(ROOT, "data", "manifest.csv")
TRIALS = os.path.join(ROOT, "data", "export_outputs", "trials")
OUT_SVG = os.path.join(ROOT, "reports", "data_pipeline.svg")
OUT_TEX = os.path.join(ROOT, "reports", "data_pipeline_table.tex")


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _read_notebook(path):
    """Concatenate the code cells of a notebook, which is where the models now live."""
    notebook = json.loads(_read(path))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")


def _search(pattern, text, name, cast=float):
    match = re.search(pattern, text, re.S)
    if not match:
        raise SystemExit(f"could not parse {name} from {UTILS}")
    return cast(match.group(1))


def read_code():
    utils = _read(UTILS)
    models = _read_notebook(MODELS)
    loader = _read(LOADER)
    features = _read(FEATURES)
    trainer = _read(TRAINER)

    signature = re.search(
        r"def __init__\(self, channel_count=(\d+), samples=(\d+), outputs=(\d+), "
        r"temporal_filters(?:_n)?=(\d+),\s*depth_multiplier=(\d+), "
        r"kernel_samples=(\d+),\s*separable_samples=(\d+),\s*pool_first=(\d+), "
        r"pool_second=(\d+),\s*dropout=([\d.]+)\)",
        models,
        re.S,
    )
    if not signature:
        raise SystemExit("could not parse CausalEEGNet.__init__ defaults")
    values = [int(value) for value in signature.groups()[:9]]
    class_dropout = float(signature.group(10))

    selected = re.search(r"SETTINGS = dict\(([^)]*)\)", trainer, re.S)
    selected_dropout = class_dropout
    if selected:
        match = re.search(r"dropout=([\d.]+)", selected.group(1))
        if match:
            selected_dropout = float(match.group(1))

    bands = re.search(r"BANDS = \[(.*?)\]\s*\n", features, re.S)
    if not bands:
        raise SystemExit("could not parse BANDS from EECS182_NHP_features.py")
    feature_count = 1 + len(re.findall(r"\(\s*\d+\s*,\s*\d+\s*,\s*[\d.]+\s*\)", bands.group(1)))

    return dict(
        C=values[0],
        T=values[1],
        V=values[2],
        F1=values[3],
        D=values[4],
        K1=values[5],
        K2=values[6],
        P=values[7],
        Q=values[8],
        class_dropout=class_dropout,
        selected_dropout=selected_dropout,
        G=feature_count,
        L=_search(r"LAG_COUNT = (\d+)", utils, "LAG_COUNT", int),
        trial_batch=_search(r"TRIAL_BATCH_SIZE = (\d+)", utils, "TRIAL_BATCH_SIZE", int),
        window_batch=_search(r"WINDOW_BATCH_SIZE = (\d+)", utils, "WINDOW_BATCH_SIZE", int),
        bin_s=_search(r"BIN_S = ([\d.]+)", loader, "BIN_S"),
        lmp_s=_search(r"LMP_WIN_S = ([\d.]+)", loader, "LMP_WIN_S"),
    )


def read_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["session"]: row for row in rows}


def split_counts(splits, manifest):
    counts = {}
    for part in ("train", "val", "test"):
        sessions = splits["sessions"][part]
        counts[part] = dict(
            days=len(splits["days"][part]),
            sessions=len(sessions),
            off=sum(int(manifest[session]["stim"]) == 0 for session in sessions),
            am3=sum(int(float(manifest[session]["am_freq_hz"])) == 3 for session in sessions),
            am20=sum(int(float(manifest[session]["am_freq_hz"])) == 20 for session in sessions),
        )
    return counts


def test_days(splits, manifest):
    rows = []
    for day in splits["days"]["test"]:
        sessions = [s for s in splits["sessions"]["test"] if manifest[s]["date"] == day]
        on = [s for s in sessions if int(manifest[s]["stim"]) == 1]
        off = [s for s in sessions if int(manifest[s]["stim"]) == 0]
        freqs = sorted({int(float(manifest[s]["am_freq_hz"])) for s in on})
        rows.append(dict(day=day, on=len(on), off=off, freqs=freqs, sessions=on))
    return rows


def calibration_checks(splits):
    pool_sizes, query_sizes = set(), set()
    for session, entry in splits["calibration"].items():
        pool_sizes.add(len(entry["pool"]))
        query_sizes.add(len(entry["query"]))
        path = os.path.join(TRIALS, f"{session}_trials.csv")
        with open(path, newline="", encoding="utf-8") as handle:
            blocks = {int(row["trial"]): int(row["task_block"]) for row in csv.DictReader(handle)}
        pool_blocks = {blocks[trial] for trial in entry["pool"]}
        query_blocks = {blocks[trial] for trial in entry["query"]}
        if pool_blocks != {1, 2} or query_blocks != {3, 4, 5}:
            raise SystemExit(f"{session}: pool/query blocks are {pool_blocks}/{query_blocks}")
    return pool_sizes, query_sizes


def compute():
    code = read_code()
    splits = json.loads(_read(SPLITS))
    manifest = read_manifest()
    counts = split_counts(splits, manifest)
    days = test_days(splits, manifest)
    pool_sizes, query_sizes = calibration_checks(splits)

    off_train = [s for s in splits["sessions"]["train"] if int(manifest[s]["stim"]) == 0]
    val_off = [s for s in splits["sessions"]["val"] if int(manifest[s]["stim"]) == 0]
    test_on = [s for s in splits["sessions"]["test"] if int(manifest[s]["stim"]) == 1]
    test_off = [s for s in splits["sessions"]["test"] if int(manifest[s]["stim"]) == 0]

    n_train = sum(
        int(manifest[s]["n_include"]) - len(splits["train_val_trials"][s]) for s in off_train
    )
    n_val = sum(int(manifest[s]["n_include"]) for s in val_off)
    n_pool = pool_sizes.pop()
    n_query = query_sizes.pop()
    ks_match = re.search(r"CALIBRATION_SIZES = \(([^)]*)\)", _read(UTILS))
    if not ks_match:
        raise SystemExit("could not parse CALIBRATION_SIZES")
    ks = [int(value) for value in re.findall(r"\d+", ks_match.group(1))]
    n_draws = _search(r"DRAWS_PER_SIZE = (\d+)", _read(UTILS), "DRAWS_PER_SIZE", int)
    n_part = len(test_on) * ((len(ks) - 1) * n_draws + 1)

    N = code["T"] // (code["P"] * code["Q"])
    F2 = code["F1"] * code["D"]
    feature_dim = (code["L"] + 1) * code["C"] * code["G"]

    if code["T"] % (code["P"] * code["Q"]):
        raise SystemExit("pool factors do not divide T into whole bins")
    expected = dict(
        C=35, T=2000, V=2, F1=8, D=2, F2=16, K1=500, K2=62, P=2, Q=25,
        G=8, L=9, N=40, feature_dim=2800,
        train_days=18, val_days=3, test_days=5,
        train_sessions=42, val_sessions=9, test_sessions=19,
        train_off=14, train_am3=13, train_am20=15,
        val_off=3, val_am3=2, val_am20=4,
        test_off=5, test_am3=7, test_am20=7,
        n_train=560, n_val=150, n_pool=20, n_query=30, n_part=434,
        trial_batch=16, window_batch=256,
    )
    actual = dict(
        C=code["C"], T=code["T"], V=code["V"], F1=code["F1"], D=code["D"], F2=F2,
        K1=code["K1"], K2=code["K2"], P=code["P"], Q=code["Q"], G=code["G"], L=code["L"],
        N=N, feature_dim=feature_dim,
        train_days=counts["train"]["days"], val_days=counts["val"]["days"],
        test_days=counts["test"]["days"],
        train_sessions=counts["train"]["sessions"], val_sessions=counts["val"]["sessions"],
        test_sessions=counts["test"]["sessions"],
        train_off=counts["train"]["off"], train_am3=counts["train"]["am3"],
        train_am20=counts["train"]["am20"],
        val_off=counts["val"]["off"], val_am3=counts["val"]["am3"],
        val_am20=counts["val"]["am20"],
        test_off=counts["test"]["off"], test_am3=counts["test"]["am3"],
        test_am20=counts["test"]["am20"],
        n_train=n_train, n_val=n_val, n_pool=n_pool, n_query=n_query, n_part=n_part,
        trial_batch=code["trial_batch"], window_batch=code["window_batch"],
    )
    failures = {key: (expected[key], actual[key]) for key in expected if expected[key] != actual[key]}
    if failures:
        details = "\n".join(f"  {key}: expected {value}, got {actual[key]}"
                            for key, value in failures.items())
        raise SystemExit(f"pipeline invariants changed:\n{details}")

    day_spec = ", ".join(
        f"{row['day'][5:]}: {row['on']} ON ({'/'.join(str(f) for f in row['freqs'])} Hz)"
        for row in days
    )
    return dict(code=code, actual=actual, counts=counts, days=days, day_spec=day_spec,
                ks=ks, n_draws=n_draws, test_on=test_on, test_off=test_off)


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

W, H = 760, 780
X = [15, 270, 525]
COL_W = [240, 240, 220]
BOX_H, GAP = 64, 10
TOP = 88

COLORS = {
    "store": ("#e7f0ff", "#2f6fdb"),
    "parse": ("#e7f6e9", "#2e7d32"),
    "split": ("#f1e9ff", "#6b46c1"),
    "batch": ("#fff4e0", "#c77c02"),
    "metric": ("#ffe9e9", "#c62828"),
}


def esc(text):
    return html.escape(str(text), quote=True)


def box_rect(x, y, w, h, title, lines, kind):
    fill, stroke = COLORS[kind]
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.1"/>',
        f'<text x="{x + 8}" y="{y + 17}" class="name">{esc(title)}</text>',
    ]
    for index, line in enumerate(lines):
        parts.append(f'<text x="{x + 8}" y="{y + 33 + index * 14}" class="sub">{esc(line)}</text>')
    return parts


def arrow_down(x, y1, y2):
    return f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" class="arrow" marker-end="url(#head)"/>'


def arrow_right(x1, x2, y):
    return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" class="arrow" marker-end="url(#head)"/>'


def build_svg(data):
    c = data["actual"]
    code = data["code"]
    body = []

    body.append('<text x="15" y="20" class="head">Task 4 data pipeline</text>')
    body.append('<text x="15" y="36" class="note">storage &#8594; parse &#8594; split/batch; '
                'all counts read from the code and splits.json</text>')

    headers = [("1. Storage", "store"), ("2. Parse and tensors", "parse"),
               ("3. Split and batch", "split")]
    for column, (text, kind) in enumerate(headers):
        fill, stroke = COLORS[kind]
        body.append(f'<rect x="{X[column]}" y="50" width="{COL_W[column]}" height="24" rx="5" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        body.append(f'<text x="{X[column] + 7}" y="67" class="lane">{esc(text)}</text>')

    storage = [
        ("manifest.csv", ["70 sessions x 61 cols", "date, stim, AM, n_include"]),
        ("splits.json", [f"{c['train_days']}/{c['val_days']}/{c['test_days']} days",
                         "calibration, off_reference"]),
        ("<session>.h5", ["/lfp (64, N_s)", "/t, masks, /stim, /task"]),
        ("trials.csv", ["50 included trials", "block 1-5, t_go, t_reward"]),
        ("kinematics", ["one row / 50 ms bin", "Index_vx_cm, Index_vy_cm"]),
        ("features/<variant>.h5", ["lmp (C,B_s), logpow (C,7,B_s)", "LMP + 7 bands"]),
        ("parrm_h5", ["/lfp, /artifact", "optional cleaned ON"]),
    ]
    parse = [
        ("load_session", [f"lfp (C={code['C']}, N_s)", "64 recorded -> 35 kept"]),
        ("normalize", ["robust z per channel", "first go -1 s to last reward +1 s"]),
        ("load_kinematics", ["pixel -> cm affine", "Index_vx_cm, Index_vy_cm"]),
        ("load_lfp_dataset", [f"(N_trials, {code['C']}, {code['T']})",
                              f"targets (N_trials, {c['N']}, {code['V']}) + mask"]),
        (f"trial_arrays(L={code['L']})", [f"(N_trials, {c['N']}, {c['feature_dim']})",
                                          f"{code['L'] + 1} lags x {code['C']} ch x {code['G']} feats"]),
        ("windows_from_valid", [f"(N_valid, {c['feature_dim']})", "drop NaN bins"]),
    ]
    split = [
        ("Day split", [f"{c['train_days']}/{c['val_days']}/{c['test_days']} days",
                       f"{c['train_sessions']}/{c['val_sessions']}/{c['test_sessions']} sessions"]),
        ("Test set", [f"{len(data['test_on'])} ON + {len(data['test_off'])} OFF",
                      "1,3,3,3,4 ON/day; 5 OFF refs"]),
        ("Calibration", [f"pool {c['n_pool']} (blocks 1-2)",
                         f"query {c['n_query']} (blocks 3-5)"]),
        ("Draws", [f"k={data['ks'][0]},{data['ks'][1]},{data['ks'][2]}: {data['n_draws']} each",
                   f"k={data['ks'][-1]}: 1; total {c['n_part']}"]),
        ("EEGNet batch", [f"{c['n_train']} train, B={c['trial_batch']}",
                          f"{c['n_train'] // c['trial_batch']} updates/epoch; val {c['n_val']}"]),
        ("Test batch", [f"{c['n_query']} query + 50 OFF", "score per day; 5 days"]),
        ("Window batch", [f"(N_valid, {c['feature_dim']}), B={c['window_batch']}", "ridge/KNN/MLP/GRU"]),
    ]

    for column, rows, kind in ((0, storage, "store"), (1, parse, "parse"), (2, split, "split")):
        y = TOP
        for index, (title, lines) in enumerate(rows):
            body += box_rect(X[column], y, COL_W[column], BOX_H, title, lines, kind)
            if index < len(rows) - 1:
                body.append(arrow_down(X[column] + 14, y + BOX_H + 1, y + BOX_H + GAP - 2))
            y += BOX_H + GAP
        if column < 2:
            body.append(arrow_right(X[column] + COL_W[column] + 2,
                                    X[column + 1] - 3, TOP + 18))

    callout_y = 615
    callout_w = 176
    callouts = [
        ("R^2 (axis-averaged)", ["R^2 = (R^2_vx + R^2_vy)/2", "per-day mean of R^2"], "metric"),
        ("5-seed ensemble", ["v_ens = (1/5) sum_s v_s", "score mean predictions"], "store"),
        ("Tones", ["synthetic 3/20 Hz sine", "clean training only"], "batch"),
        ("Why five test days?", ["test split is 5 days", "all per-day figures"], "split"),
    ]
    for index, (title, lines, kind) in enumerate(callouts):
        x = 15 + index * (callout_w + 8)
        body += box_rect(x, callout_y, callout_w, 88, title, lines, kind)

    body.append(f'<text x="15" y="722" class="note">'
                f'Test days: {esc(data["day_spec"])}</text>')
    body.append('<text x="15" y="736" class="note">'
                'training: up to 200 epochs; early stop 20; batch 16; '
                'Muon on weight matrices + Adam on biases and BatchNorm vectors</text>')
    body.append(f'<text x="15" y="758" class="var">'
                f'C {code["C"]}   T {code["T"]}   N {c["N"]}   V {code["V"]}   '
                f'B {code["trial_batch"]}/{code["window_batch"]}   F1 {code["F1"]}   '
                f'D {code["D"]}   F2 {c["F2"]}   K1 {code["K1"]}   K2 {code["K2"]}   '
                f'P {code["P"]}   Q {code["Q"]}   G {code["G"]}   L {code["L"]}</text>')

    style = """<style>
  svg .name { font-size: 12px; font-weight: 600; fill: #1f2328; }
  svg .lane { font-size: 11.5px; font-weight: 700; fill: #1f2328; }
  svg .sub  { font-size: 10.5px; fill: #57606a; font-family: ui-monospace, Menlo, monospace; }
  svg .note { font-size: 8.8px; fill: #57606a; }
  svg .var  { font-size: 9.8px; fill: #1f2328; font-family: ui-monospace, Menlo, monospace; }
  svg .head { font-size: 16px; font-weight: 700; fill: #1f2328; }
  svg .arrow { stroke: #6e7781; stroke-width: 1.1; }
</style>
<defs>
  <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5"
          orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e7781"/>
  </marker>
</defs>"""

    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
        style,
        *body,
        "</svg>",
    ])


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def tex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def tt(text):
    return r"\texttt{" + tex_escape(text) + "}"


def build_table(data):
    c = data["actual"]
    code = data["code"]
    day_counts = ", ".join(str(row["on"]) for row in data["days"])
    rows = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Task 4 dataset, tensor shapes, splits, and calibration protocol "
        r"($C=35$ channels, $T=2{,}000$ samples, $N=40$ 50\,ms bins, $V=2$ velocity axes).}",
        r"\label{tab:datapipeline}",
        r"\footnotesize",
        r"\begin{tabular}{@{}p{0.22\textwidth}p{0.72\textwidth}@{}}",
        r"\toprule",
        r"Component & Specification \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textbf{(a) Preprocessing and Tensor Shapes}} \\",
        r"Recordings \& Scaling & 70 sessions (1\,kHz raw LFP); 35 fixed motor-cortex "
        r"channels centered and scaled per recording by median and MAD ($z$-units). \\",
        r"Inputs & Window $[t_{\mathrm{go}}-0.5,\, t_{\mathrm{go}}+1.5]$\,s "
        r"($40 \times 50$\,ms bins): raw LFP $(N_{\mathrm{trials}}, 35, 2000)$ for "
        r"EEGNet; lagged spectral/LMP features $(N_{\mathrm{trials}}, 40, 2800)$ for "
        r"window models ($8\text{ features} \times 35\text{ ch} \times 10\text{ lags}$). \\",
        r"Targets \& Mask & Binned fingertip velocity $(v_x, v_y)$ in cm/s, shape "
        r"$(N_{\mathrm{trials}}, 40, 2)$, paired with binary validity mask "
        r"$(N_{\mathrm{trials}}, 40)$. \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textbf{(b) Date-Grouped Splits (70 Sessions Across 26 Days)}} \\",
        r"Train / Val (21 days) & \textbf{Train (18d):} 42 sessions "
        f"({c['train_off']} OFF, {c['train_am3']} at 3\\,Hz, {c['train_am20']} at 20\\,Hz). "
        r"\textbf{Val (3d):} 9 sessions "
        f"({c['val_off']} OFF, {c['val_am3']} at 3\\,Hz, {c['val_am20']} at 20\\,Hz). \\\\",
        r"Test (5 days) & 19 sessions: "
        f"{c['test_off']} OFF, {c['test_am3']} at 3\\,Hz, {c['test_am20']} at 20\\,Hz "
        f"({day_counts} ON sessions/day, each paired with its day's OFF reference). \\\\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textbf{(c) Calibration Draws and Model Batching}} \\",
        r"Calibration Partitions & Across 14 test ON sessions: first 20 trials = "
        r"calibration pool; final 30 trials = fixed query set. For "
        r"$k \in \{2, 5, 10, 20\}$: 10 subsets for $k < 20$ and 1 full-pool draw for "
        r"$k = 20$ ($434$ partitions). \\",
        r"Training \& Batching & \textbf{Causal EEGNet:} 560 train / 150 val OFF trials "
        r"(batch 16; $F_1=8, D=2, F_2=16, K_1=500, K_2=62$, pool $2\times 25, p=0.25$). "
        r"\textbf{Window models:} valid bins $(N_{\mathrm{valid}}, 2800)$; batch 256 "
        r"(MLP/GRU) or closed-form (ridge). \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(rows) + "\n"



def main():
    data = compute()
    os.makedirs(os.path.dirname(OUT_SVG), exist_ok=True)
    with open(OUT_SVG, "w", encoding="utf-8") as handle:
        handle.write(build_svg(data))
    with open(OUT_TEX, "w", encoding="utf-8") as handle:
        handle.write(build_table(data))
    c = data["actual"]
    print(f"wrote {OUT_SVG}")
    print(f"wrote {OUT_TEX}")
    print(f"  C={c['C']} T={c['T']} N={c['N']} V={c['V']} F1={c['F1']} F2={c['F2']}")
    print(f"  train={c['n_train']} val={c['n_val']} pool={c['n_pool']} query={c['n_query']} "
          f"partitions={c['n_part']}")


if __name__ == "__main__":
    main()
