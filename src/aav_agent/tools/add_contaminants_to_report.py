"""Append a contaminant-screening section to a LAAVA HTML report.

Takes the outputs of classify_reads.py (per-read TSV and summary JSON), draws a
pie chart of unmapped reads by organism type, and splices a new section into
the LAAVA report just before "Methods", plus a matching table-of-contents entry.
The chart is embedded as a base64 PNG, so the report stays a single file. LAAVA
itself is not modified.

Running it again on its own output replaces the previous section (the inserted
HTML is bracketed by marker comments), so it is safe to re-run.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import html
import io
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

SECTION_ID = "contaminant-screening"
SECTION_START = "<!-- aav-agent:contaminants:start -->"
SECTION_END = "<!-- aav-agent:contaminants:end -->"
TOC_START = "<!-- aav-agent:contaminants-toc:start -->"
TOC_END = "<!-- aav-agent:contaminants-toc:end -->"

# Okabe-Ito colour-blind-safe palette; assigned by name so a category keeps its
# colour from run to run. Unlisted buckets cycle through the spare colours.
COLORS = {
    "host (human)": "#0072B2",
    "enterobacteria": "#D55E00",
    "soil bacteria": "#009E73",
    "other bacteria": "#56B4E9",
    "baculovirus": "#CC79A7",
    "other virus": "#E69F00",
    "insect": "#F0E442",
    "plant": "#44AA44",
    "other eukaryote": "#8C6BB1",
    "unclassified": "#BBBBBB",
    "ambiguous": "#999999",
    "mixed": "#777777",
    "low coverage": "#555555",
    "no hit": "#DDDDDD",
}
SPARE = ["#332288", "#88CCEE", "#AA4499", "#DDCC77", "#117733", "#882255"]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def read_tsv(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def laava_read_counts(per_read_tsv: Path) -> tuple[int, int]:
    """(total reads, unmapped reads) from LAAVA's per-read table."""
    rows = read_tsv(per_read_tsv)
    return len(rows), sum(1 for r in rows if r.get("reference_label") == "(unmapped)")


# --------------------------------------------------------------------------
# Chart and section
# --------------------------------------------------------------------------


def bucket_colors(buckets: list[str]) -> list[str]:
    spare = iter(SPARE)
    out = []
    for b in buckets:
        out.append(COLORS.get(b) or next(spare, "#666666"))
    return out


def make_pie_png(buckets: dict[str, dict], total: int) -> bytes:
    names = list(buckets)
    counts = [buckets[n]["reads"] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)
    wedges, _ = ax.pie(
        counts, colors=bucket_colors(names), startangle=90, counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    ax.set_title(f"Unmapped reads by organism type (n = {total:,})", fontsize=12)
    ax.legend(
        wedges,
        [f"{n}: {c:,} ({c / total:.0%})" if c / total >= 0.01 else f"{n}: {c:,} (<1%)"
         for n, c in zip(names, counts)],
        loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10,
    )
    ax.set_aspect("equal")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _td(text, align: str = "left") -> str:
    return f'<td align="{align}">{html.escape(str(text))}</td>'


def _table(caption: str, headers: list[str], rows: list[list], aligns: list[str]) -> str:
    out = ["<table>", f"<caption>{html.escape(caption)}</caption>", "<thead>", '<tr class="header">']
    out += [f'<th align="{a}">{html.escape(h)}</th>' for h, a in zip(headers, aligns)]
    out += ["</tr>", "</thead>", "<tbody>"]
    for i, r in enumerate(rows):
        out.append(f'<tr class="{"odd" if i % 2 == 0 else "even"}">')
        out += [_td(v, a) for v, a in zip(r, aligns)]
        out.append("</tr>")
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


def database_label(summary: dict, override: str = "") -> str:
    """Label for the database, taken from what blastn recorded unless overridden.

    An explicit override that disagrees with the recorded title is allowed but
    logged, because a report that names the wrong database is worse than none.
    """
    db = summary.get("database") or {}
    recorded = db.get("db_title") or db.get("db_name") or ""
    if recorded and db.get("db_date"):
        recorded_full = f"{recorded}, {db['db_date']}"
    else:
        recorded_full = recorded
    if override:
        if recorded and override.strip().lower() not in recorded.lower() \
                and recorded.lower() not in override.strip().lower():
            logger.warning("--db-label %r differs from the database recorded by blastn (%r); using the label "
                           "you gave", override, recorded)
        return override
    return recorded_full


def build_section(
    summary: dict,
    rows: list[dict],
    laava_total: int | None = None,
    db_label: str = "",
    show_validation: bool = False,
) -> str:
    buckets = summary["buckets"]
    total = summary["total_reads"]
    p = summary.get("params", {})

    parts = [SECTION_START, f'<div id="{SECTION_ID}" class="section level1">', "<h1>Contaminant screening</h1>"]

    context = f"LAAVA could not map {total:,} reads to any supplied reference"
    if laava_total:
        context += f" ({total / laava_total:.2%} of {laava_total:,} reads in the sample)"
    db_label = database_label(summary, db_label)
    parts.append(
        f"<p>{html.escape(context)}. These reads were compared with a nucleotide "
        f"database{' (' + html.escape(db_label) + ')' if db_label else ''} and assigned to an organism "
        "type. Reads that LAAVA assigned to the vector, packaging or helper plasmids, host, or "
        "chimeric types are <em>not</em> included here.</p>"
    )

    if total == 0:
        parts.append("<p>There were no unmapped reads, so there is nothing to screen.</p>")
    else:
        png = base64.b64encode(make_pie_png(buckets, total)).decode()
        parts.append(
            f'<p><img role="img" alt="Pie chart of unmapped reads by organism type" '
            f'src="data:image/png;base64,{png}" style="max-width:100%;height:auto" /></p>'
        )
        parts.append(_table(
            "Unmapped reads by organism type",
            ["Organism type", "Reads", "% of unmapped reads", "Bases", "% of unmapped bases"],
            [[b, f"{v['reads']:,}", f"{v['read_fraction']:.1%}", f"{v['bases']:,}", f"{v['base_fraction']:.1%}"]
             for b, v in buckets.items()],
            ["left", "right", "right", "right", "right"],
        ))

        ident = Counter(
            (r["lca_name"] or "-", r["bucket"]) for r in rows if r["lca_name"] and r["bucket"] not in ("no hit",)
        )
        if ident:
            parts.append(_table(
                "Taxa identified (lowest common ancestor of the best-matching sequences)",
                ["Taxon", "Organism type", "Reads"],
                [[t, b, f"{n:,}"] for (t, b), n in ident.most_common(15)],
                ["left", "left", "right"],
            ))

        unassigned = [(b, v["reads"]) for b, v in buckets.items()
                      if b in ("no hit", "ambiguous", "mixed", "low coverage", "unclassified")]
        notes = []
        if any(b == "no hit" for b, _ in unassigned):
            notes.append(
                "<em>no hit</em>: no significant match in the database used. This can mean low-complexity "
                "sequence or an organism the database does not contain, so a no-hit read is not evidence "
                "of absence of contamination."
            )
        if any(b == "ambiguous" for b, _ in unassigned):
            notes.append("<em>ambiguous</em>: best matches span unrelated groups of organisms.")
        if any(b == "mixed" for b, _ in unassigned):
            notes.append("<em>mixed</em>: different parts of the read match different organism types.")
        if any(b == "low coverage" for b, _ in unassigned):
            notes.append("<em>low coverage</em>: matches cover only a small part of the read.")
        if notes:
            parts.append("<p>" + "<br />\n".join(notes) + "</p>")

    if p:
        parts.append(
            "<p>Method: blastn (megablast), hits filtered at "
            f"&ge;{p.get('min_pident', '?')}% identity, &ge;{p.get('min_aln_len', '?')} bp alignment length and "
            f"e-value &le;{p.get('max_evalue', '?')}. Each read is assigned to the lowest common ancestor of all "
            f"sequences scoring within {float(p.get('lca_window', 0)):.0%} of its best match, then mapped to an "
            "organism type.</p>"
        )

    v = summary.get("validation")
    if show_validation and not v:
        logger.warning("--show-validation given but the summary has no validation results; "
                       "re-run classify_reads.py with --truth")
    if show_validation and v:
        parts.append(
            "<p><strong>Demonstration data:</strong> this sample includes synthetic reads spiked in from known "
            "organisms, so the table below compares what was found with what was added.</p>"
        )
        parts.append(_table(
            "Validation against spiked-in reads (demonstration data)",
            ["Expected", "Observed", "Reads"],
            [[c["expected"], c["observed"], c["reads"]] for c in v["confusion"]],
            ["left", "left", "right"],
        ))

    parts += ["</div>", SECTION_END]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# Splice
# --------------------------------------------------------------------------


def _strip_previous(report: str) -> str:
    for start, end in ((SECTION_START, SECTION_END), (TOC_START, TOC_END)):
        report = re.sub(re.escape(start) + r".*?" + re.escape(end) + r"\n?", "", report, flags=re.S)
    return report


def splice(report: str, section: str) -> str:
    """Insert the section before Methods and add a TOC entry; idempotent."""
    report = _strip_previous(report)

    toc_li = (
        f'{TOC_START}<li><a href="#{SECTION_ID}" id="toc-{SECTION_ID}">Contaminant screening</a></li>'
        f"{TOC_END}\n"
    )
    toc_anchor = '<li><a href="#methods"'
    if toc_anchor in report:
        report = report.replace(toc_anchor, toc_li + toc_anchor, 1)
    else:
        logger.warning("No TOC entry for Methods found; section added without a TOC link")

    anchor = '<div id="methods" class="section level1">'
    if anchor in report:
        return report.replace(anchor, section + anchor, 1)
    logger.warning("Methods section not found; appending the section before </body> instead")
    if "</body>" not in report:
        raise ValueError("Report has neither a Methods section nor a </body> tag; is it a LAAVA HTML report?")
    return report.replace("</body>", section + "</body>", 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", type=Path, help="LAAVA HTML report")
    ap.add_argument("classified", type=Path, help="Per-read TSV from classify_reads.py")
    ap.add_argument("summary", type=Path, help="Summary JSON from classify_reads.py")
    ap.add_argument("out", type=Path, help="Output HTML (may equal the input to update in place)")
    ap.add_argument("--laava-per-read", type=Path, help="LAAVA per-read TSV(.gz), for sample-wide context")
    ap.add_argument("--db-label", default="",
                    help="Override the database description (default: taken from the blastn provenance file "
                         "recorded in the summary JSON)")
    ap.add_argument("--show-validation", action="store_true", help="Include the spike-in validation table")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = json.loads(a.summary.read_text())
    rows = read_tsv(a.classified)
    laava_total = laava_read_counts(a.laava_per_read)[0] if a.laava_per_read else None
    section = build_section(summary, rows, laava_total, a.db_label, a.show_validation)
    a.out.write_text(splice(a.report.read_text(encoding="utf-8"), section), encoding="utf-8")
    logger.info("Wrote %s", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
