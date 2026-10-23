#!/usr/bin/env bash
# Build the submission DOCX (and PDF if LibreOffice is installed). Run from the repository root: bash paper/build.sh
set -euo pipefail
cd "$(dirname "$0")"
NAME=ECIS2027_Statistically_Certified_Autonomy
python build_docx.py
python pack_docx.py build/tpl "build/$NAME.docx"
if command -v soffice > /dev/null; then
  soffice --headless --convert-to pdf --outdir build "build/$NAME.docx" > /dev/null 2>&1 && echo "PDF: paper/build/$NAME.pdf"
  command -v pdfinfo > /dev/null && pdfinfo "build/$NAME.pdf" | grep Pages
else
  echo "LibreOffice not found: open the DOCX in Word and export to PDF; check that it has at most 16 pages."
fi
