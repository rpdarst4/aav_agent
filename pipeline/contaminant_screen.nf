#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * Contaminant screen for a finished LAAVA run.
 *
 * Consumes LAAVA's published outputs (name-sorted BAM, per-read TSV, HTML report),
 * BLASTs the reads LAAVA could not map, classifies them by organism type, and writes a
 * copy of the LAAVA report with a "Contaminant screening" section added.
 *
 * Why a separate workflow instead of including LAAVA's `laava` workflow: LAAVA does not
 * emit its HTML report as a channel (it only publishes it), so a downstream process could
 * not reliably wait for it. Running after LAAVA completes avoids that race, and it lets you
 * re-screen an existing run against a different database without re-running LAAVA.
 *
 * Directives that use input variables (tag, publishDir) are written as closures so the
 * script also parses under Nextflow's strict syntax.
 *
 * Expected LAAVA layout (as published by `--output`):
 *   <laava_output>/alignment/<sample_id>.sort_by_name.bam
 *   <laava_output>/report/<sample_id>.per_read.tsv.gz
 *   <laava_output>/report/<sample_id>_AAV_report.html
 */

process EXTRACT_UNMAPPED {
    tag { sample_id }
    publishDir path: { "${params.outdir}/${sample_id}/unmapped" }, mode: 'copy'

    input:
    tuple val(sample_id), path(name_sorted_bam), path(per_read_tsv)

    output:
    tuple val(sample_id), path("${sample_id}.unmapped.fa"), emit: fasta

    script:
    """
    python -m aav_agent.tools.extract_unmapped ${name_sorted_bam} ${per_read_tsv} ${sample_id}.unmapped.fa
    """
}

process RUN_BLASTN {
    tag { sample_id }
    publishDir path: { "${params.outdir}/${sample_id}/blast" }, mode: 'copy'

    input:
    tuple val(sample_id), path(unmapped_fa)
    path blast_db_dir
    val blast_db_name

    output:
    tuple val(sample_id), path("${sample_id}.blast.tsv"), path("${sample_id}.blast.tsv.meta.json"), emit: hits

    script:
    """
    THREADS=${task.cpus} run_blastn.sh ${unmapped_fa} ${sample_id}.blast.tsv ${blast_db_dir}/${blast_db_name}
    """
}

process CLASSIFY {
    tag { sample_id }
    publishDir path: { "${params.outdir}/${sample_id}/classification" }, mode: 'copy'

    input:
    tuple val(sample_id), path(hits), path(hits_meta), path(unmapped_fa)
    path taxdump_dir
    path categories
    path truth

    output:
    tuple val(sample_id), path("${sample_id}.classified.tsv"), path("${sample_id}.classified.json"), emit: result

    script:
    def truth_arg = truth.name == 'NO_FILE' ? '' : "--truth ${truth}"
    """
    python -m aav_agent.tools.classify_reads \\
        --blast ${hits} --blast-meta ${hits_meta} --queries ${unmapped_fa} \\
        --taxdump ${taxdump_dir} --categories ${categories} \\
        --out-tsv ${sample_id}.classified.tsv --out-summary ${sample_id}.classified.json ${truth_arg}
    """
}

process ADD_TO_REPORT {
    tag { sample_id }
    publishDir path: { "${params.outdir}/${sample_id}/report" }, mode: 'copy'

    input:
    tuple val(sample_id), path(report_html), path(per_read_tsv), path(classified_tsv), path(summary_json)

    output:
    tuple val(sample_id), path("${sample_id}_AAV_report.contaminants.html"), emit: report

    script:
    def label_arg = params.db_label ? "--db-label '${params.db_label}'" : ''
    def validation_arg = params.show_validation ? '--show-validation' : ''
    """
    python -m aav_agent.tools.add_contaminants_to_report ${report_html} ${classified_tsv} ${summary_json} \\
        ${sample_id}_AAV_report.contaminants.html --laava-per-read ${per_read_tsv} ${label_arg} ${validation_arg}
    """
}

workflow {
    if( !params.laava_output ) { error "Provide --laava_output (the output directory of a LAAVA run)" }
    if( !params.sample_id )    { error "Provide --sample_id (comma-separated for several samples)" }
    if( !params.blast_db_dir ) { error "Provide --blast_db_dir (directory containing the BLAST database)" }
    if( !params.taxdump_dir )  { error "Provide --taxdump_dir (directory with nodes.dmp and names.dmp)" }

    def laava_out   = params.laava_output
    def blast_db    = file(params.blast_db_dir, type: 'dir', checkIfExists: true)
    def taxdump     = file(params.taxdump_dir, type: 'dir', checkIfExists: true)
    def categories  = file(params.categories, checkIfExists: true)
    def truth_file  = params.truth ? file(params.truth, checkIfExists: true) : file("${projectDir}/assets/NO_FILE")

    def laava_files = channel.fromList( params.sample_id.toString().tokenize(',') ).map { id ->
        tuple(
            id,
            file("${laava_out}/alignment/${id}.sort_by_name.bam", checkIfExists: true),
            file("${laava_out}/report/${id}.per_read.tsv.gz", checkIfExists: true),
            file("${laava_out}/report/${id}_AAV_report.html", checkIfExists: true)
        )
    }

    EXTRACT_UNMAPPED( laava_files.map { id, bam, tsv, html -> tuple(id, bam, tsv) } )
    RUN_BLASTN( EXTRACT_UNMAPPED.out.fasta, blast_db, params.blast_db_name )
    CLASSIFY( RUN_BLASTN.out.hits.join( EXTRACT_UNMAPPED.out.fasta ), taxdump, categories, truth_file )
    ADD_TO_REPORT(
        laava_files.map { id, bam, tsv, html -> tuple(id, html, tsv) }.join( CLASSIFY.out.result )
    )
}
