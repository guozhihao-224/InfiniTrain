#!/usr/bin/env bash
#
# Run the Generator test suite twice and compare results.
#
# Usage: ./scripts/check_reproducibility.sh [build_dir]
#   build_dir defaults to "build". Pass "build_cpu" for CPU-only trees.
#
# Exit code 0 means the two ctest runs produced the same per-test outcomes
# (Pass/Fail/Skip lines). Non-zero means non-reproducible (or build missing).
#
# Reference: spec §9.1 in
#   docs/superpowers/specs/2026-05-19-generator-abstraction-design.md

set -euo pipefail

BUILD_DIR="${1:-build}"
FILTER='^test_generator_'

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: build dir '$BUILD_DIR' not found" >&2
    echo "Hint: cmake -B $BUILD_DIR ... && cmake --build $BUILD_DIR -j" >&2
    exit 2
fi

TMP_DIR="$(mktemp -d -t generator-repro-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

LOG1="$TMP_DIR/run1.log"
LOG2="$TMP_DIR/run2.log"

run_ctest() {
    local out="$1"
    ctest --test-dir "$BUILD_DIR" --output-on-failure -R "$FILTER" -O "$out" \
        > "${out}.stdout" 2>&1 || true
}

# Filter to per-test verdict lines only (drop timestamps / durations).
# Example matched line:
#   Test #42: GeneratorSeedTest/CPU.ManualSeedReseedsState ...   Passed    0.04 sec
filter_verdicts() {
    grep -E '^[[:space:]]*Test #[0-9]+: .* (Passed|Failed|Skipped)' "$1" \
        | sed -E 's/[[:space:]]+[0-9]+\.[0-9]+ sec$//'
}

echo "[1/2] running ctest in $BUILD_DIR (filter $FILTER)"
run_ctest "$LOG1"
echo "[2/2] running ctest again"
run_ctest "$LOG2"

V1="$TMP_DIR/v1"
V2="$TMP_DIR/v2"
filter_verdicts "$LOG1" > "$V1"
filter_verdicts "$LOG2" > "$V2"

if [[ ! -s "$V1" || ! -s "$V2" ]]; then
    echo "ERROR: no Generator test verdicts captured; is the suite built?" >&2
    echo "  see ${LOG1}.stdout and ${LOG2}.stdout for raw output" >&2
    exit 3
fi

if diff -u "$V1" "$V2" >/dev/null; then
    n=$(wc -l < "$V1")
    echo "PASS: $n test outcomes match across two ctest runs"
    exit 0
fi

echo "FAIL: ctest verdicts differ between runs (non-reproducible)"
diff -u "$V1" "$V2" || true
exit 1
