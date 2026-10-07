import argparse
import pysam

from pathlib import Path

import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bamfile", help="Path to a LAAVA output name-sorted BAM file")
    parser.add_argument("tsvfile", help="Path to a LAAVA output per-read TSV")
    parser.add_argument("fastafile", help="Path to FASTA file of unmapped reads to write")
    args = parser.parse_args()
    write_fasta(
        args.fastafile,
        read_tsv(
            args.bamfile,
            args.tsvfile))


def read_tsv(bamfile: str, tsvfile: str):
    for path in (bamfile, tsvfile):
        if not Path(path).exists():
            raise FileNotFoundError(f"Cannot find {path}")
    df = pd.read_csv(tsvfile, sep='\t')
    unmapped = set(df[df["reference_label"] == "(unmapped)"]["read_id"])
    reads = pysam.AlignmentFile(bamfile, "rb", check_sq=False)
    for read in reads:
        if read.query_name in unmapped and read.is_unmapped:
            yield SeqRecord(
                Seq(read.query_sequence),
                id=read.query_name,
                name="",
                description="",
            )


def write_fasta(fastafile: str, records):
    if not records:
        return
    with open(fastafile, 'w') as handle:
        SeqIO.write(list(records), handle, "fasta")


if __name__ == "__main__":
    main()
