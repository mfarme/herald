# GPT-OSS-20B HERALD Evaluation - Quarto Document

This directory contains the converted Quarto (.qmd) version of the GPT-OSS-20B HERALD evaluation writeup.

## Files

- `gpt-oss-20b-writeup.qmd` - Main Quarto document with YAML frontmatter, proper citations, and cross-references
- `references.bib` - BibTeX bibliography file with all citations converted from numbered format
- `README.md` - This documentation file

## Key Changes from Original Markdown

1. **Format Conversion**: Converted from .md to .qmd (Quarto markdown)
2. **YAML Frontmatter**: Added comprehensive metadata including:
   - Title, author, and affiliation
   - Multiple output formats (HTML and PDF)
   - Bibliography and citation style configuration
   - Abstract and keywords
   - Content warning
3. **Citations**: Converted from numbered citations (1,2,3...) to BibTeX cite keys (@author2025title)
4. **Cross-references**: Added proper Quarto cross-references for tables and figures
5. **Table Formatting**: Converted the results table to proper Quarto table format with caption and ID
6. **Bibliography**: Created comprehensive BibTeX file with all 8 references

## Usage

To render this document:

```bash
quarto render gpt-oss-20b-writeup.qmd
```

Or to render to specific format:

```bash
quarto render gpt-oss-20b-writeup.qmd --to html
quarto render gpt-oss-20b-writeup.qmd --to pdf
```

## Notes

- The figure reference `escalation_metrics.png` was not found in the original figures directory
- Path updated to `../figures/escalation_metrics.png` to maintain correct relative path from findings directory
- All citations have been converted to proper academic format with DOIs where available