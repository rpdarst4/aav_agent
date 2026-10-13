"""Make a small structural copy of a LAAVA HTML report for tests.

Embedded images, inline CSS and scripts are removed; the document structure (table of contents,
section divs, tables, Methods) is kept. Tests that splice into the report use this instead of a
1-2 MB real report. Regenerate it from a report made by a new LAAVA version to see whether the
splice still finds its anchors:

    python scripts/make_report_skeleton.py ../laava/.../sample_AAV_report.html tests/fixtures/laava_report_skeleton.html
"""

import re
import sys
from pathlib import Path


def make_skeleton(html: str) -> str:
    html = re.sub(r"<style\b.*?</style>", "<style></style>", html, flags=re.S)
    html = re.sub(r"<script\b.*?</script>", "", html, flags=re.S)
    html = re.sub(r'src="data:image/[^"]*"', 'src="data:image/png;base64,"', html)
    html = re.sub(r"<link\b[^>]*>", "", html)
    note = "<!-- Structural copy of a real LAAVA report (images, CSS and scripts removed); see scripts/make_report_skeleton.py -->\n"
    return note + html


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    out = make_skeleton(src.read_text(encoding="utf-8", errors="replace"))
    for anchor in ('<div id="methods" class="section level1">', '<li><a href="#methods"'):
        if anchor not in out:
            sys.exit(f"warning: expected anchor not found in the report: {anchor}")
    dst.write_text(out, encoding="utf-8")
    print(f"wrote {dst} ({len(out):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
