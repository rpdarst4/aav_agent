#!/usr/bin/env bash
# Build the small demo contaminant BLAST database used when full NT is not available.
#
# Usage:
#   scripts/build_demo_db.sh [--fasta-dir DIR] [--manifest FILE] [--out DIR]
#                            [--taxdump-dir DIR] [--download-taxdump]
#
# Defaults (override with flags or env vars):
#   --fasta-dir   fixtures/contaminants
#   --manifest    <fasta-dir>/manifest.tsv     (columns: fasta, taxid, organism)
#   --out         ${BLASTDB_DIR:-data/blastdb}
#   --taxdump-dir ${TAXDUMP_DIR:-data/taxonomy}
#
# Outputs (in --out):
#   demo_contaminants.*         BLAST nucleotide database (with taxids)
#   demo_contaminants.build.txt provenance: tool version, FASTA checksums, organisms
#
# The taxdump (nodes.dmp, names.dmp, merged.dmp) is needed by classify_reads.py.
# It is only downloaded when --download-taxdump is passed.
#
# Compatible with the bash 3.2 that ships with macOS.

set -euo pipefail

FASTA_DIR="fixtures/contaminants"
MANIFEST=""
OUT="${BLASTDB_DIR:-data/blastdb}"
TAXDUMP_DIR="${TAXDUMP_DIR:-data/taxonomy}"
DOWNLOAD_TAXDUMP=0
DB_NAME="demo_contaminants"

while [ $# -gt 0 ]; do
    case "$1" in
        --fasta-dir) FASTA_DIR="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --taxdump-dir) TAXDUMP_DIR="$2"; shift 2 ;;
        --download-taxdump) DOWNLOAD_TAXDUMP=1; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$MANIFEST" ] || MANIFEST="$FASTA_DIR/manifest.tsv"

command -v makeblastdb >/dev/null 2>&1 || { echo "makeblastdb not found (install BLAST+ or run inside the project image)" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "Manifest not found: $MANIFEST" >&2; exit 1; }

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p "$OUT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
COMBINED="$WORK/combined.fa"
TAXMAP="$WORK/taxid_map.txt"
INFO="$OUT/$DB_NAME.build.txt"
: > "$COMBINED"; : > "$TAXMAP"

{
    echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "version: $(makeblastdb -version | head -1)"
    echo "manifest: $MANIFEST"
    echo "fasta_sha256:"
} > "$INFO"

# Read manifest, skipping the header line and blank lines.
n=0
while IFS=$'\t' read -r fasta taxid organism; do
    [ "$fasta" = "fasta" ] && continue
    [ -n "$fasta" ] || continue
    case "$taxid" in ''|*[!0-9]*) echo "Bad taxid '$taxid' for $fasta in $MANIFEST" >&2; exit 1 ;; esac
    path="$FASTA_DIR/$fasta"
    [ -f "$path" ] || { echo "FASTA listed in manifest is missing: $path" >&2; exit 1; }
    cat "$path" >> "$COMBINED"
    # Last line of a FASTA may lack a newline; make sure records do not run together.
    printf '\n' >> "$COMBINED"
    # taxid map: "<sequence id> <taxid>" for every record (id = first word of header).
    awk -v t="$taxid" '/^>/ { id = substr($1, 2); print id, t }' "$path" >> "$TAXMAP"
    echo "  $(sha256 "$path")  $fasta  taxid=$taxid  $organism" >> "$INFO"
    n=$((n + 1))
done < "$MANIFEST"

[ "$n" -gt 0 ] || { echo "No FASTA files listed in $MANIFEST" >&2; exit 1; }

dups="$(cut -d' ' -f1 "$TAXMAP" | sort | uniq -d)"
if [ -n "$dups" ]; then
    echo "Duplicate sequence ids across FASTA files (makeblastdb would fail):" >&2
    echo "$dups" >&2
    exit 1
fi

# Drop blank lines introduced by the separators above.
grep -v '^$' "$COMBINED" > "$COMBINED.clean" && mv "$COMBINED.clean" "$COMBINED"

makeblastdb -in "$COMBINED" -dbtype nucl -parse_seqids \
    -taxid_map "$TAXMAP" -title "AAV demo contaminants" \
    -out "$OUT/$DB_NAME" >/dev/null

echo "sequences: $(wc -l < "$TAXMAP" | tr -d ' ')" >> "$INFO"
echo "Built $OUT/$DB_NAME from $n FASTA files ($(wc -l < "$TAXMAP" | tr -d ' ') sequences)"

if [ "$DOWNLOAD_TAXDUMP" -eq 1 ]; then
    mkdir -p "$TAXDUMP_DIR"
    echo "Downloading NCBI taxdump to $TAXDUMP_DIR ..."
    curl -fsSL "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz" -o "$WORK/taxdump.tar.gz"
    tar -xzf "$WORK/taxdump.tar.gz" -C "$TAXDUMP_DIR" nodes.dmp names.dmp merged.dmp
    echo "taxdump: downloaded $(date -u +%Y-%m-%d)" >> "$INFO"
elif [ ! -f "$TAXDUMP_DIR/nodes.dmp" ]; then
    echo "Note: no taxdump in $TAXDUMP_DIR; re-run with --download-taxdump before classifying." >&2
fi
