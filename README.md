# RNA-seq Analysis Pipeline

A one-click Streamlit web app that takes a raw **feature counts** file and runs a full
downstream RNA-seq analysis: differential expression, volcano plot, heatmap, GO/KEGG
enrichment, GSEA, Reactome pathways, splicing factors, drug targets, transcription
factors, and a STRING protein-protein interaction network — with a ZIP export of all
tables and figures and an auto-generated Methods + citations block.

## Features

- **Auto ID detection** — Entrez / Ensembl IDs are detected and converted to gene symbols automatically.
- **Choice of statistical test** — DESeq2 (recommended), Welch t-test, or Mann-Whitney U.
- **Multiple-testing correction** — Benjamini-Hochberg (FDR) or Bonferroni.
- **Custom group names** and editable significance thresholds.
- **Customizable plots** — volcano axes/labels/colors, heatmap color scale and colormap.
- **QC step** — PCA and sample-correlation before testing.
- **Per-plot and bundled CSV/Excel/figure downloads.**

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Deploy

This repo is ready for [Streamlit Community Cloud](https://share.streamlit.io):
point it at this repository with `app.py` as the main file.

## Citing

The app cites the underlying tools (PyDESeq2, Enrichr, GSEApy, Reactome, STRING, and
core Python libraries) for the steps you run, and generates a ready-to-paste Methods
paragraph. Author and website citation fields are configurable in the sidebar.

## Author

Khalid Al-Abdulla
