# Image for the contaminant-screen workflow: BLAST+, Python deps, this repo's tools and scripts.
# Build from the repo root:
#   docker build -f docker/tools.Dockerfile -t aav-agent-tools:local .
# If the BLAST+ package has no build for your CPU (e.g. Apple Silicon), add --platform linux/amd64.
FROM mambaorg/micromamba:2.0.5

USER root
COPY docker/tools-env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean -a -y

# Nextflow starts tasks without the image's entrypoint, so put the environment on PATH directly.
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/opt/aav_agent/src \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

COPY src/ /opt/aav_agent/src/
COPY scripts/run_blastn.sh scripts/build_demo_db.sh /usr/local/bin/
# Tasks run as the host user (-u uid:gid), so everything must be world-readable regardless of the
# file modes in the build context.
RUN chmod 755 /usr/local/bin/run_blastn.sh /usr/local/bin/build_demo_db.sh \
 && chmod -R a+rX /opt/aav_agent
