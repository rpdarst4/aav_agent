#!/usr/bin/env bash
# Run blastn on a FASTA of unmapped reads with the columns classify_reads.py expects.
#
# Usage: scripts/run_blastn.sh QUERIES.fa OUT.tsv [DB]
#
#   DB defaults to ${BLAST_DB:-data/blastdb/demo_contaminants}. To use NCBI nt or
#   core_nt instead, point BLASTDB at the directory holding it and pass the name:
#       BLASTDB=/mnt/ncbi/blast scripts/run_blastn.sh reads.fa hits.tsv core_nt
#   The v5 NCBI databases carry taxids, so staxids is populated automatically.
#
# Environment: THREADS (default 4), EVALUE (default 1e-5), MAX_TARGETS (default 500).
# MAX_TARGETS is deliberately high: nt holds many near-identical strains per
# species, and classify_reads.py needs all the near-best subjects for its LCA.
set -euo pipefail

[ $# -ge 2 ] || { sed -n '2,13p' "$0"; exit 2; }
QUERIES="$1"; OUT="$2"; DB="${3:-${BLAST_DB:-data/blastdb/demo_contaminants}}"

command -v blastn >/dev/null 2>&1 || { echo "blastn not found" >&2; exit 1; }
[ -s "$QUERIES" ] || { echo "No reads to BLAST in $QUERIES (nothing unmapped?); writing empty result"; : > "$OUT"; exit 0; }

blastn -query "$QUERIES" -db "$DB" -task megablast \
    -evalue "${EVALUE:-1e-5}" -max_target_seqs "${MAX_TARGETS:-500}" \
    -num_threads "${THREADS:-4}" \
    -outfmt "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore staxids" \
    -out "$OUT"
echo "Wrote $(wc -l < "$OUT" | tr -d ' ') HSPs to $OUT"
