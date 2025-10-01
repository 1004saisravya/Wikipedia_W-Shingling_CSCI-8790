import hashlib
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from collections import defaultdict
from pathlib import Path

def read_input_file(file_name):
    """
    Read the entire contents of a text file.
    Parameters
    ----------
    file_name : str
        Path to the text file to read.

    Returns
    -------
    str
        The full contents of the file as a single string.
    """
    with open(file_name, "r", encoding="utf-8") as file:
        content = file.read()
    return content

def w_shingles(text, w, lam):
    """
    Generate (w, λ)-shingles from input text.
    This function splits the text into tokens (by whitespace)
    and then slides a window of size `w` over the tokens,
    stepping by `lam` each time to create shingles.

    Parameters
    ----------
    text : str
        The input text to shingle.
    w : int
        The window size (number of tokens per shingle).
    lam : int
        The step size between consecutive shingles.

    Returns
    -------
    list of tuple
        Each tuple contains `w` consecutive tokens.
    """
    # Tokenize based on spaces
    tokens = text.split()
    shingles = []

    # Slide window of size w across tokens with step lam
    for i in range(0, len(tokens) - w + 1, lam):
        shingle = tuple(tokens[i:i + w])  # make tuple
        shingles.append(shingle)

    return shingles


def hash_shingles(shingles):
    """
    Convert shingles to MD5 hashes.
    Each shingle is assumed to be a tuple of tokens (strings).
    The function joins the tokens with a separator, encodes to bytes,
    and computes an MD5 hash for each shingle.

    Parameters
    ----------
    shingles : list of tuple
        List of shingles, where each shingle is a tuple of tokens.

    Returns
    -------
    list of str
        List of MD5 hash strings corresponding to each shingle.
    """
    hashes = []
    for shingle in shingles:
        # Join tokens using a separator; you can lowercase here if desired
        shingle_str = "|".join(shingle)  # skip .lower() if you care about case
        # Encode to bytes and hash using MD5
        shingle_hash = hashlib.md5(shingle_str.encode("utf-8")).hexdigest()
        hashes.append(shingle_hash)

    return hashes


def union(list1, list2):
    """
    Compute the union of two lists.

    Parameters
    ----------
    list1 : list
    list2 : list

    Returns
    -------
    list
        Elements present in either list1 or list2 (unique).
    """
    return list(set(list1) | set(list2))


def intersection(list1, list2):
    """
    Compute the intersection of two lists.

    Parameters
    ----------
    list1 : list
    list2 : list

    Returns
    -------
    list
        Elements present in both list1 and list2 (unique).
    """
    return list(set(list1) & set(list2))

# -----------------------------
# Everything below is the added/modified logic
# -----------------------------

# Patterns to support both naming styles in the data folders
_PAT_C_CURR = re.compile(r"^(.+)_C\.txt$")
_PAT_C_OLD  = re.compile(r"^(.+)_C-(\d+)\.txt$")
_PAT_VC_CURR = re.compile(r"^VC_T\.txt$")
_PAT_VC_OLD  = re.compile(r"^VC_T-(\d+)\.txt$")

def _lambda_min(hashes, lam):
    """
    λ-min selection on hashed shingles.
    Keep the lexicographically smallest λ hash strings to form a compact
    sketch of the document. If λ is ∞, keep all hashes.
    """
    if lam == float("inf"):
        return list(hashes)
    lam = int(lam)
    return sorted(hashes)[:lam]

def _jaccard(hashes_a, hashes_b):
    """
    Jaccard similarity between two hash sets.
    """
    A, B = set(hashes_a), set(hashes_b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B) if (A or B) else 0.0

def _find_city_versions(city_dir: Path):
    """
    Locate the current file and all older versions for a city.
    Supports both naming schemes:
      1) <City_State>_C.txt , <City_State>_C-3.txt , ...
      2) VC_T.txt , VC_T-3.txt , ...
    Returns (current_path, [(version_label, path, lag_int), ...]) sorted by lag.
    """
    files = [p for p in city_dir.iterdir() if p.is_file() and p.suffix == ".txt"]
    current = None
    older = []

    for p in files:
        if _PAT_C_CURR.match(p.name) or _PAT_VC_CURR.match(p.name):
            current = p
            break
    if not current:
        raise FileNotFoundError(f"Missing current file in {city_dir} (need *_C.txt or VC_T.txt)")

    for p in files:
        m1 = _PAT_C_OLD.match(p.name)
        m2 = _PAT_VC_OLD.match(p.name)
        if m1:
            lag = int(m1.group(2)); older.append((f"C-{lag}", p, lag))
        elif m2:
            lag = int(m2.group(1)); older.append((f"VC_T-{lag}", p, lag))

    older.sort(key=lambda x: x[2])
    return current, older

def _compute_city_jaccards(city_dir: Path, w_values, lam_values):
    """
    Computes per-city similarities without writing CSV.
    Returns: dict like results[w][lam] -> list[(version_label, jaccard_float)]
    """
    current, older = _find_city_versions(city_dir)
    text_cur = read_input_file(str(current)).lower()

    # Precompute current shingles & hashes per w to reuse across λ
    cur_hashes_by_w = {}
    for w in w_values:
        cur_sh = w_shingles(text_cur, w, 1)  # stride=1 per spec
        cur_h  = hash_shingles(cur_sh)
        cur_hashes_by_w[w] = cur_h

    # For each older version, compute hashes per w, then jaccard for each λ
    results = {w: {lam: [] for lam in lam_values} for w in w_values}

    for ver_label, ver_path, _lag in older:
        text_old = read_input_file(str(ver_path)).lower()
        for w in w_values:
            old_sh = w_shingles(text_old, w, 1)
            old_h  = hash_shingles(old_sh)

            # reuse current hashes for this w
            cur_h = cur_hashes_by_w[w]
            for lam in lam_values:
                cur_sel = _lambda_min(cur_h, lam)
                old_sel = _lambda_min(old_h, lam)
                sim = _jaccard(cur_sel, old_sel)
                results[w][lam].append((ver_label, sim))

    # Ensure version order by numeric lag in the label, e.g., C-3, VC_T-6
    lag_num = lambda v: int(re.search(r"(\d+)$", v).group(1)) if re.search(r"(\d+)$", v) else 0
    for w in w_values:
        for lam in lam_values:
            results[w][lam].sort(key=lambda kv: lag_num(kv[0]))

    return results

def _plot_city_w_facets(city: str, results, out_dir: Path, lam_values, w):
    """
    For (city, w), produce one PNG with 5 subplots (one per λ).
    Saves to: output/plots/{city}_w{w}_lambdas_facet.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build x-axis (shared across λ) using any λ group
    any_lam = next(iter(results[w].keys()))
    xs = [v for v, _ in results[w][any_lam]]

    # 5 subplots stacked (5 rows x 1 column)
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 14), sharex=True, constrained_layout=True)

    for idx, lam in enumerate(lam_values):
        ax = axes[idx]
        ys = [s for _, s in results[w][lam]]
        lam_str = "inf" if lam == float("inf") else str(lam)
        ax.plot(xs, ys, marker="o")
        ax.set_title(f"w={w}, λ={lam_str}")
        ax.set_ylabel("Jaccard")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Older version (C-3, C-6, …)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    outfile = out_dir / f"{city}_w{w}_lambdas_facet.png"
    fig.suptitle(f"{city} — Jaccard vs Version (5 λ subplots)", fontsize=14)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[plot] {outfile}")

# ---------- ADDED: closest-λ CSV (computed from in-memory results) ----------
def _write_closest_lambda_from_results(city: str, results, lam_values, out_dir: Path):
    """
    For a city's in-memory results:
      For each w, find finite λ whose mean Jaccard is closest to λ=∞ mean.
    Writes: output/results/closest_lambda/closest_lambda_<city>.csv
    Columns: w,best_lambda,abs_gap_vs_inf,inf_mean,lambda_mean
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"closest_lambda_{city}.csv"

    rows = []
    for w, lam_dict in results.items():
        # Must have ∞ to compare against
        if float("inf") not in lam_dict:
            continue
        inf_mean = mean([s for _, s in lam_dict[float("inf")]])
        best = None
        for lam in lam_values:
            if lam == float("inf"):
                continue
            lam_mean = mean([s for _, s in lam_dict[lam]])
            gap = abs(lam_mean - inf_mean)
            if best is None or gap < best[2]:
                best = (lam, lam_mean, gap)
        if best:
            rows.append((w, best[0], best[2], inf_mean, best[1]))

    if rows:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("w,best_lambda,abs_gap_vs_inf,inf_mean,lambda_mean\n")
            for w, lam, gap, inf_mean, lam_mean in rows:
                lam_str = "inf" if lam == float("inf") else str(lam)
                f.write(f"{w},{lam_str},{gap:.6f},{inf_mean:.6f},{lam_mean:.6f}\n")
        for w, lam, gap, inf_mean, lam_mean in rows:
            lam_str = "inf" if lam == float("inf") else str(lam)
            print(f"[closest-λ] {city}: w={w} → λ={lam_str} (|{lam_mean:.4f}−{inf_mean:.4f}|={gap:.4f})")

# ---------- ADDED: timing over entire corpus + timing plot ----------
def _collect_all_texts(root: Path, cities):
    """
    Gather every text from all cities (current + each older version).
    Returns: list[str] lowercased texts.
    """
    texts = []
    for city in cities:
        cdir = root / city
        if not cdir.exists():
            continue
        current, older = _find_city_versions(cdir)
        for p in [current] + [vp for _, vp, _ in older]:
            texts.append(read_input_file(str(p)).lower())
    return texts

def _timing_over_corpus(cities, w_values, lam_values, root: Path, out_dir: Path, runs=3):
    """
    For each (w, λ), measure average time (seconds) of:
      shingling (stride=1) → hashing → λ-min selection,
    aggregated across all documents (all cities, all versions).
    Writes: output/results/timing.csv with columns w,lambda,avg_seconds
    """
    texts = _collect_all_texts(root, cities)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "timing.csv"

    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("w,lambda,avg_seconds\n")
        for w in w_values:
            for lam in lam_values:
                times = []
                for _ in range(runs):
                    t0 = time.perf_counter()
                    for tx in texts:
                        sh = w_shingles(tx, w, 1)
                        hh = hash_shingles(sh)
                        _ = _lambda_min(hh, lam)
                    times.append(time.perf_counter() - t0)
                lam_str = "inf" if lam == float("inf") else str(lam)
                f.write(f"{w},{lam_str},{sum(times)/len(times):.6f}\n")
    print(f"[timing] results → {csv_file}")
    return csv_file

def _plot_timing_from_csv(csv_path: Path, out_png: Path):
    """
    Read timing.csv and plot avg_seconds vs λ, one line per w.
    """
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return
    rows = [ln.split(",") for ln in lines[1:] if ln.strip()]

    # organize into series per w with λ on x-axis in file order
    by_w = defaultdict(list)
    lam_axis = []
    for w, lam, secs in rows:
        if lam not in lam_axis:
            lam_axis.append(lam)           # preserves file order: 8,16,32,64,inf
        by_w[int(w)].append(float(secs))

    plt.figure()
    for w in sorted(by_w.keys()):
        plt.plot(lam_axis, by_w[w], marker="o", label=f"w={w}")
    plt.title("Shingling time over corpus vs λ")
    plt.xlabel("λ (inf = all hashes)")
    plt.ylabel("Average seconds")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[plot] {out_png}")

def main():
    # Data root detection compatible with your structure
    root = Path("data")
    if not root.exists():
        alt = Path("Data")
        root = alt if alt.exists() else root

    if not root.exists():
        raise FileNotFoundError("Could not find 'data/' or 'Data/' directory.")

    out_plots = Path("output/plots")
    out_results = Path("output/results")
    out_closest = out_results / "closest_lambda"

    cities = [d.name for d in root.iterdir() if d.is_dir()]
    if not cities:
        print("[warn] No city folders found under", root)
        return

    # Required parameter grid
    w_values   = [25, 50]
    lam_values = [8, 16, 32, 64, float("inf")]

    # Per-city processing: plots + closest-λ CSV
    for city in cities:
        city_dir = root / city
        try:
            results = _compute_city_jaccards(city_dir, w_values, lam_values)
        except FileNotFoundError as e:
            print(f"[skip] {city}: {e}")
            continue

        # Create one PNG per w, each with 5 subplots
        for w in w_values:
            _plot_city_w_facets(city, results, out_plots, lam_values, w)

        # NEW: write closest-λ CSV for this city (from in-memory results)
        _write_closest_lambda_from_results(city, results, lam_values, out_closest)

    # NEW: timing over corpus + timing plot
    timing_csv = _timing_over_corpus(cities, w_values, lam_values, root=root, out_dir=out_results, runs=3)
    _plot_timing_from_csv(Path(timing_csv), out_png=out_plots / "timing.png")

if __name__ == "__main__":
    main()