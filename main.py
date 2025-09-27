import hashlib
import re
import time
import matplotlib.pyplot as plt
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

    Parameters
    ----------
    hashes : list[str]
        MD5 hex strings for a document's shingles.
    lam : int or float('inf')
        Number of smallest hashes to keep (∞ → all).

    Returns
    -------
    list[str]
        Selected hash strings.
    """
    if lam == float("inf"):
        return list(hashes)
    lam = int(lam)
    return sorted(hashes)[:lam]

def _jaccard(hashes_a, hashes_b):
    """
    Jaccard similarity between two hash sets.

    Parameters
    ----------
    hashes_a : list[str]
    hashes_b : list[str]

    Returns
    -------
    float
        |A ∩ B| / |A ∪ B| where A,B are sets of hash strings.
    """
    A, B = set(hashes_a), set(hashes_b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B) if (A or B) else 0.0

def _find_city_versions(city_dir: Path):
    """
    Locate the current file and all older versions for a city.

    Supports both naming schemes used in the project:
      1) <City_State>_C.txt , <City_State>_C-3.txt , ...
      2) VC_T.txt , VC_T-3.txt , ...

    Parameters
    ----------
    city_dir : Path
        Directory containing the city's text files.

    Returns
    -------
    tuple[Path, list[tuple[str, Path, int]]]
        current_path,
        list of (version_name, version_path, numeric_lag) sorted by lag.
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

def _process_city(city, w_values, lam_values, root: Path, out_dir: Path):
    """
    Compute Jaccard(current, older) for all (w, λ) and save CSV.

    Parameters
    ----------
    city : str
        City folder name under the data root.
    w_values : list[int]
        Window sizes to evaluate (e.g., [25, 50]).
    lam_values : list[Union[int, float]]
        λ-min sizes (e.g., [8,16,32,64,float('inf')]).
    root : Path
        Data root (./data or ./Data).
    out_dir : Path
        Output directory for CSV files.

    Returns
    -------
    Path | None
        Path to the city's CSV if processed, else None.
    """
    city_dir = root / city
    if not city_dir.exists():
        print(f"[skip] {city} not found in {root}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / f"{city}.csv"

    current, older = _find_city_versions(city_dir)
    text_cur = read_input_file(str(current)).lower()

    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("city,version,w,lambda,jaccard\n")
        for w in w_values:
            cur_sh = w_shingles(text_cur, w, 1)   # stride=1 per spec
            cur_h  = hash_shingles(cur_sh)
            for lam in lam_values:
                cur_sel = _lambda_min(cur_h, lam)
                for ver_name, ver_path, _ in older:
                    text_old = read_input_file(str(ver_path)).lower()
                    old_sh = w_shingles(text_old, w, 1)
                    old_h  = hash_shingles(old_sh)
                    old_sel = _lambda_min(old_h, lam)
                    sim = _jaccard(cur_sel, old_sel)
                    lam_str = "inf" if lam == float("inf") else str(lam)
                    f.write(f"{city},{ver_name},{w},{lam_str},{sim:.6f}\n")
    print(f"[{city}] results → {csv_file}")
    return csv_file

def _collect_all_texts(root: Path, cities):
    """
    Gather every text from all cities (current + each older version).

    Used for the timing experiment to measure total shingling + λ-min
    cost over the whole corpus.

    Parameters
    ----------
    root : Path
        Data root (./data or ./Data).
    cities : list[str]
        City folder names.

    Returns
    -------
    list[str]
        Lowercased document texts.
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

def _timing(cities, w_values, lam_values, root: Path, out_dir: Path, runs=3):
    """
    Simple timing experiment over the corpus.

    For each (w, λ), measure average time (seconds) of:
      shingling (stride=1) → hashing → λ-min selection,
    aggregated across all documents (all cities, all versions).

    Parameters
    ----------
    cities : list[str]
        City folder names to include.
    w_values : list[int]
        Window sizes to evaluate.
    lam_values : list[Union[int, float]]
        λ values to evaluate (including ∞).
    root : Path
        Data root.
    out_dir : Path
        Output directory for timing.csv.
    runs : int
        Repeat count to average timings.

    Returns
    -------
    Path
        Path to timing.csv.
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

# ---- simple plotting (uses only matplotlib) ----
def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None

def _plot_city_from_csv(city, csv_path: Path, out_dir: Path):
    """
    Read similarities_<city>.csv and plot Jaccard vs version
    for each (w, λ). One PNG per (w, λ).
    """
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plot] matplotlib not available; skipping city plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return
    header = lines[0].split(",")
    rows = [ln.split(",") for ln in lines[1:] if ln.strip()]

    # group by (w, lambda)
    from collections import defaultdict
    groups = defaultdict(list)   # (w, lam) -> [(version, jaccard_float)]
    for city_, version, w, lam, jac in rows:
        if city_ != city:
            continue
        groups[(int(w), lam)].append((version, float(jac)))

    # sort by numeric lag in the version name (e.g., C-3, VC_T-6, ...)
    import re
    for (w, lam), pairs in groups.items():
        pairs.sort(key=lambda x: int(re.search(r"(\d+)$", x[0]).group(1)) if re.search(r"(\d+)$", x[0]) else 0)
        xs = [v for v, _ in pairs]
        ys = [s for _, s in pairs]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.title(f"{city} — Jaccard vs Version (w={w}, λ={lam})")
        plt.xlabel("Older version (C-3, C-6, ...)")
        plt.ylabel("Jaccard similarity")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_png = out_dir / f"{city}_w{w}_lam{lam}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[plot] {out_png}")

def _plot_timing_from_csv(csv_path: Path, out_png: Path):
    """
    Read timing.csv and plot avg_seconds vs λ, one line per w.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plot] matplotlib not available; skipping timing plot")
        return

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return
    rows = [ln.split(",") for ln in lines[1:] if ln.strip()]

    # organize into series per w with λ on x-axis in file order
    from collections import defaultdict
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
    root = Path("data")
    if not root.exists():
        alt = Path("Data")
        root = alt if alt.exists() else root

    out_results = Path("output/results")
    out_plots   = Path("output/plots")
    cities = ["Jacksonville_FL", "Berkeley_CA", "Edinburg_TX", "Winter Graden_FL"]

    w_values   = [25, 50]
    lam_values = [8, 16, 32, 64, float("inf")]

    city_csv_paths = []
    for city in cities:
        csv_path = _process_city(city, w_values, lam_values, root=root, out_dir=out_results)
        if csv_path:
            city_csv_paths.append((city, csv_path))
    
    t_csv = _timing(cities, w_values, lam_values, root=root, out_dir=out_results, runs=3)

    for city, csv_path in city_csv_paths:
        _plot_city_from_csv(city, csv_path, out_dir=out_plots)
    _plot_timing_from_csv(t_csv, out_png=out_plots / "timing.png")

if __name__ == "__main__":
    main()
