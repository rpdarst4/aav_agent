Usage

```shell
poetry run pytest
(cd ../laava && NXF_SYNTAX_PARSER=v1 nextflow run -profile local main.nf \
  -params-file ../aav_agent/fixtures/spike_in/params-local-ss.json \
  --output ../aav_agent/data/laava/output -w ../aav_agent/data/laava/work \
  --cpus 4 --memory '6 GB')
python src/aav_agent/tools/extract_unmapped.py data/laava/output/alignment/ss.spikein.sort_by_name.bam \
    data/laava/output/report/ss.spikein.per_read.tsv.gz data/ss_unmapped.fa
bash scripts/run_blastn.sh data/ss_unmapped.fa data/ss_unmapped.tsv
python src/aav_agent/tools/classify_reads.py --blast data/ss_unmapped.tsv --queries data/ss_unmapped.fa \
  --taxdump data/taxonomy --categories scripts/categories.yaml \
  --truth fixtures/spike_in/ss.spikein.truth.tsv \
  --out-tsv data/ss_blasted.tsv --out-summary data/ss_blasted.json
python src/aav_agent/tools/add_contaminants_to_report.py data/laava/output/report/ss.spikein_AAV_report.html \
  data/ss_blasted.tsv data/ss_blasted.json data/ss.subsample005_AAV_report.spliced.html \
    --laava-per-read data/laava/output/report/ss.spikein.per_read.tsv.gz --db-label "NCBI core_nt" \
    --show-validation
```