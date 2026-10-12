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
# Also writes OUT.meta.json (database title/date, BLAST version, parameters).
#
# Environment: THREADS (default 4), EVALUE (default 1e-5), MAX_TARGETS (default 500).
# MAX_TARGETS is deliberately high: nt holds many near-identical strains per
# species, and classify_reads.py needs all the near-best subjects for its LCA.
set -euo pipefail

[ $# -ge 2 ] || { sed -n '2,15p' "$0"; exit 2; }
QUERIES="$1"; OUT="$2"; DB="${3:-${BLAST_DB:-data/blastdb/demo_contaminants}}"

command -v blastn >/dev/null 2>&1 || { echo "blastn not found" >&2; exit 1; }

# Provenance sidecar (OUT.meta.json): which database and tool version produced these hits.
# classify_reads.py picks it up automatically so reports state the database actually used.
write_meta() {
    command -v python3 >/dev/null 2>&1 || { echo "python3 not found; skipping $OUT.meta.json" >&2; return 0; }
    DB="$DB" BLASTN_VERSION="$(blastn -version | head -1)" \
    DBINFO="$(blastdbcmd -db "$DB" -info 2>/dev/null || true)" \
    EVALUE="${EVALUE:-1e-5}" MAX_TARGETS="${MAX_TARGETS:-500}" \
    python3 - "$OUT.meta.json" <<'PY'
import datetime, json, os, re, sys
info = os.environ["DBINFO"]
title = re.search(r"^Database:\s*(.+)$", info, re.M)
date = re.search(r"^\s*Date:\s*(.+?)\s{2,}", info, re.M) or re.search(r"^\s*Date:\s*(.+)$", info, re.M)
seqs = re.search(r"([\d,]+)\s+sequences", info)
json.dump({
    "db_path": os.environ["DB"],
    "db_name": os.path.basename(os.environ["DB"]),
    "db_title": title.group(1).strip() if title else "",
    "db_date": date.group(1).strip() if date else "",
    "db_sequences": int(seqs.group(1).replace(",", "")) if seqs else None,
    "blastn": os.environ["BLASTN_VERSION"],
    "evalue": os.environ["EVALUE"],
    "max_target_seqs": int(os.environ["MAX_TARGETS"]),
    "run_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, open(sys.argv[1], "w"), indent=2)
PY
}

if [ ! -s "$QUERIES" ]; then
    echo "No reads to BLAST in $QUERIES (nothing unmapped?); writing empty result"
    : > "$OUT"; write_meta; exit 0
fi

blastn -query "$QUERIES" -db "$DB" -task megablast \
    -evalue "${EVALUE:-1e-5}" -max_target_seqs "${MAX_TARGETS:-500}" \
    -num_threads "${THREADS:-4}" \
    -outfmt "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore staxids" \
    -out "$OUT"
write_meta
echo "Wrote $(wc -l < "$OUT" | tr -d ' ') HSPs to $OUT"
