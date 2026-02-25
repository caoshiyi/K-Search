#!/usr/bin/env bash
set -e

# Use the upstream reference-kernels evaluator (clone if needed)
TRIMUL_DIR="/tmp/reference-kernels/problems/bioml/trimul"
if [ ! -d "$TRIMUL_DIR" ]; then
    echo "Cloning gpu-mode/reference-kernels..."
    git clone --depth 1 https://github.com/gpu-mode/reference-kernels.git /tmp/reference-kernels
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE_DIR="/tmp/triton_cache"

NUM_ITERS=3

# Create benchmark test cases file (matches upstream task.yml benchmarks)
BENCHFILE=$(mktemp /tmp/trimul_bench_XXXXXX.txt)
cat > "$BENCHFILE" <<'EOF'
seqlen: 256; bs: 2; dim: 128; hiddendim: 128; seed: 9371; nomask: True; distribution: normal
seqlen: 768; bs: 1; dim: 128; hiddendim: 128; seed: 381; nomask: True; distribution: cauchy
seqlen: 256; bs: 2; dim: 384; hiddendim: 128; seed: 2301; nomask: False; distribution: normal
seqlen: 512; bs: 1; dim: 128; hiddendim: 128; seed: 12819; nomask: True; distribution: normal
seqlen: 1024; bs: 1; dim: 128; hiddendim: 128; seed: 381; nomask: True; distribution: cauchy
seqlen: 768; bs: 1; dim: 384; hiddendim: 128; seed: 481; nomask: False; distribution: normal
seqlen: 1024; bs: 1; dim: 384; hiddendim: 128; seed: 23291; nomask: True; distribution: normal
EOF

# Auto-discover all .py solution files in the script directory (exclude bench scripts)
KERNELS=()
KERNEL_FILES=()
for pyfile in "$SCRIPT_DIR"/*.py; do
    [ -f "$pyfile" ] || continue
    name="$(basename "$pyfile" .py)"
    KERNELS+=("$name")
    KERNEL_FILES+=("$pyfile")
done

if [ ${#KERNELS[@]} -eq 0 ]; then
    echo "No .py kernel files found in $SCRIPT_DIR"
    exit 1
fi

echo "Found ${#KERNELS[@]} kernels: ${KERNELS[*]}"
echo ""

RESULTS_DIR=$(mktemp -d /tmp/trimul_results_XXXXXX)

for i in "${!KERNELS[@]}"; do
    name="${KERNELS[$i]}"
    kfile="${KERNEL_FILES[$i]}"

    echo "============================================================"
    echo "  Kernel: $name  (${NUM_ITERS} runs)"
    echo "============================================================"

    # Clean triton cache before each kernel for fair autotuning
    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"
    export TRITON_CACHE_DIR="$CACHE_DIR"

    for iter in $(seq 1 $NUM_ITERS); do
        outfile="$RESULTS_DIR/${name}_run${iter}.txt"

        cd "$TRIMUL_DIR"
        cp "$kfile" submission.py

        POPCORN_FD=3 python eval.py benchmark "$BENCHFILE" 3>"$outfile" 2>&1 || true
        rm -f submission.py

        echo "  Run $iter done"
    done
    echo ""
done

echo ""
echo "============================================================"
echo "  Parsing results..."
echo "============================================================"

# Pass the kernel names as a comma-separated string
KERNEL_LIST=$(IFS=,; echo "${KERNELS[*]}")

python - "$RESULTS_DIR" "$NUM_ITERS" "$KERNEL_LIST" <<'PYEOF'
import sys, os, math

results_dir = sys.argv[1]
num_iters = int(sys.argv[2])
kernels = sys.argv[3].split(",")
num_bm = 7

bm_labels = [
    "seq=256  bs=2 dim=128 nmsk=T norm",
    "seq=768  bs=1 dim=128 nmsk=T cchy",
    "seq=256  bs=2 dim=384 nmsk=F norm",
    "seq=512  bs=1 dim=128 nmsk=T norm",
    "seq=1024 bs=1 dim=128 nmsk=T cchy",
    "seq=768  bs=1 dim=384 nmsk=F norm",
    "seq=1024 bs=1 dim=384 nmsk=T norm",
]

def parse_file(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ": " in line:
                k, v = line.split(": ", 1)
                data[k.strip()] = v.strip()
    return data

# Collect: kernel -> run -> bm_idx -> mean_ms
all_means = {}   # kernel -> list of list  [run][bm]
all_geos = {}    # kernel -> list of geo means

for kname in kernels:
    all_means[kname] = []
    all_geos[kname] = []
    for r in range(1, num_iters + 1):
        path = os.path.join(results_dir, f"{kname}_run{r}.txt")
        data = parse_file(path)
        means = []
        for bi in range(num_bm):
            m = data.get(f"benchmark.{bi}.mean")
            if m is not None:
                means.append(float(m) / 1e6)  # ns -> ms
            else:
                means.append(None)
        all_means[kname].append(means)
        valid = [v for v in means if v is not None]
        if valid:
            geo = math.exp(sum(math.log(v) for v in valid) / len(valid))
        else:
            geo = None
        all_geos[kname].append(geo)

# Determine column width
col_w = max(max(len(k) for k in kernels) + 4, 26)

# Per-benchmark stats across runs
print()
header_text = f"PER-BENCHMARK RESULTS (mean +/- std across {num_iters} run(s), ms)"
total_w = 40 + 2 + col_w * len(kernels)
print("=" * total_w)
print(f"{header_text:^{total_w}s}")
print("=" * total_w)
hdr = f"{'Benchmark':>40s}"
for k in kernels:
    hdr += f"  {k:^{col_w}s}"
print(hdr)
print("-" * total_w)

for bi in range(num_bm):
    row = f"{bm_labels[bi]:>40s}"
    for k in kernels:
        vals = [all_means[k][r][bi] for r in range(num_iters) if all_means[k][r][bi] is not None]
        if vals:
            m = sum(vals) / len(vals)
            s = math.sqrt(sum((v - m)**2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0
            cell = f"{m:8.3f} +/- {s:<5.3f} ms"
            row += f"  {cell:^{col_w}s}"
        else:
            row += f"  {'ERR':^{col_w}s}"
    print(row)

# Geo mean stats
print()
print("=" * total_w)
geo_text = f"GEOMETRIC MEAN ACROSS {num_iters} RUN(S)"
print(f"{geo_text:^{total_w}s}")
print("=" * total_w)
hdr2 = f"{'':>15s}"
for k in kernels:
    hdr2 += f"  {k:^{col_w}s}"
print(hdr2)
sep_w = 15 + 2 + col_w * len(kernels)
print("-" * sep_w)

for r in range(num_iters):
    row = f"{'Run ' + str(r+1):>15s}"
    for k in kernels:
        g = all_geos[k][r]
        if g is not None:
            cell = f"{g:8.3f} ms"
            row += f"  {cell:^{col_w}s}"
        else:
            row += f"  {'ERR':^{col_w}s}"
    print(row)

print("-" * sep_w)
row_avg = f"{'Avg Geo Mean':>15s}"
row_std = f"{'Std Geo Mean':>15s}"
for k in kernels:
    geos = [g for g in all_geos[k] if g is not None]
    if geos:
        avg = sum(geos) / len(geos)
        std = math.sqrt(sum((g - avg)**2 for g in geos) / (len(geos) - 1)) if len(geos) > 1 else 0
        avg_cell = f"{avg:8.3f} ms"
        std_cell = f"{std:8.4f} ms"
        row_avg += f"  {avg_cell:^{col_w}s}"
        row_std += f"  {std_cell:^{col_w}s}"
    else:
        row_avg += f"  {'N/A':^{col_w}s}"
        row_std += f"  {'N/A':^{col_w}s}"
print(row_avg)
print(row_std)
print()

PYEOF

rm -rf "$RESULTS_DIR"
rm -f "$BENCHFILE"
echo "Done."
