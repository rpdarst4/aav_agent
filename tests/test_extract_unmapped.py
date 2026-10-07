import tempfile as tf

from Bio.SeqRecord import SeqRecord

from src.aav_agent.tools import extract_unmapped


def test_extract_unmapped():
    bamfile = "tests/fixtures/sc.subsample005.sort_by_name.bam"
    tsvfile = "tests/fixtures/sc.subsample005.per_read.tsv.gz"
    reads = list(extract_unmapped.read_tsv(bamfile, tsvfile))
    assert all(isinstance(read, SeqRecord) for read in reads)
    assert len(reads) == 1
    with tf.NamedTemporaryFile("wt") as fastafile:
        extract_unmapped.write_fasta(fastafile.name, reads)
