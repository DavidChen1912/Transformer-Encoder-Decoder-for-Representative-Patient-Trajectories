# Manuscript sources

- `manuscript.md`: English article text, equation source, figure placements and captions.
- `build_manuscript.py`: generates `../ot_transformer_manuscript.pdf`.
- `verify_pdf.py`: page rendering, geometry checks and independent numerical checks.
- `figures/` and `qa/`: generated equations, placement copies, and review previews. These directories are recreated by the scripts and excluded from Git.

The PDF uses embedded STIX General text fonts and vector STIX mathematics. All ten PNG figures from the root `chart/` and `docs/chart/` directories are reused without changing their source files. Conceptual figures remain labeled as illustrations.

Run from the repository root in the Ubuntu environment:

```bash
python3 docs/manuscript/build_manuscript.py
python3 docs/manuscript/verify_pdf.py
```

Python dependencies: reportlab, matplotlib, pillow, pdfrw, pymupdf and numpy. The builder uses Matplotlib's bundled STIX fonts and the system DejaVu Sans font for the architecture diagram. A temporary dependency path is appended when available; normal installed packages take precedence.

The research status reflects the author's report that initial training is complete and parameter comparisons are in progress. No unprovided fitted-model metrics or clinical labels are added. This revision does not execute model training or alter the model implementation.

Typography and placement conventions: display equations use a fixed 11.5 pt STIX type size without automatic downscaling. Running-text mathematics uses explicit `$...$` notation and baseline-aligned 600 dpi STIX glyphs at the surrounding text size (10.5 pt in prose; 8 pt in tables). Backticks distinguish code/data identifiers from mathematical variables. Existing PNG illustrations are cropped only at blank outer margins and placed at figure-specific sizes; original chart files remain unchanged. Tables stay intact, and headings remain with the following content.

