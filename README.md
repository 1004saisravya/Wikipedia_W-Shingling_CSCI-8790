The objective of this project is to investigate the evolution of Wikipedia city pages across time pressures utilizing W-shingling and λ-min sketching. For these city pages we maintain both the current version and prior versions of each page and calculate Jaccard similarity against two values of shingle window sizes, w ∈ {25, 50}, and λ-min sketching against five sketch sizes, λ ∈ {8, 16, 32, 64, ∞}. The code provides CSV based results, individual city plots, and a timing plot indicator of generally performance.

The pipeline reads a corpus of plain-text Wikipedia pages for various cities, converts each page to lowercase, separates it into words, and generates w-shingles at stride 1. A new ID is created for each shingle by hashing its content with MD5 to get a fixed-length ID. A λ-min sketch is then formed by keeping the lexicographically smallest λ hashes (λ = ∞ will simply keep all the hashes). Jaccard similarity is computed between the newest page and every previous page version. Finally, it will output CSVs (incoming similarities plus “closest-λ”/midpoint summaries), generate city-specific PNG plots with five stacked subplots (one for each λ), and output a timing plot reporting average seconds versus λ (two curves per page version for w=25 and w=50).

We consider two sizes of the shingle window, w = 25 and w = 50, and five sizes for the sketches, λ = 8, 16, 32, 64, ∞ (where λ = ∞ implies we keep all the hashes with no sketching). For each (w,λ) combination, we compute the Jaccard similarity between the current page and all historical versions of the page, and report the historical versions in chronological order according to numeric lag so that the x-axis reflect the timeline of edits.

MD5 (Message Digest 5) is a fast, deterministic hash function that maps any input (like a text shingle) to a fixed-length 128-bit “fingerprint.” In this project, we use MD5 to turn each shingle into a compact, fixed-size ID so we can compare sets efficiently and build λ-min sketches.


REQUIREMENTS:
  1.Python 3.9+
  2.Python packages:
    -> numpy, pandas, matplotlib


COMMAND FOR EXECUTION:
  python main.py --data_dir ./data --out_dir ./output  ## It reads the full pipeline i.e.,read data from ./data and writes output in ./output






