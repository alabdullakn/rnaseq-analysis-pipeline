"""
RNA-seq Analysis Pipeline - Streamlit App
Upload a feature counts file to get full downstream analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import os
import tempfile

st.set_page_config(
    page_title="RNA-seq Analysis Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Visual theme - injected CSS for a polished, modern look
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1180px; }

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #7C3AED 0%, #06B6D4 100%);
    padding: 2.3rem 2.5rem;
    border-radius: 20px;
    color: #fff;
    box-shadow: 0 12px 34px rgba(124, 58, 237, 0.28);
    margin-bottom: 1.8rem;
}
.hero h1 { color: #fff; font-size: 2.15rem; font-weight: 800; letter-spacing: -0.6px; margin: 0 0 0.4rem 0; }
.hero p { color: rgba(255,255,255,0.94); font-size: 1.07rem; margin: 0; line-height: 1.5; }

/* Section headers (st.header -> h2) */
h2 {
    font-weight: 800 !important;
    color: #1E293B !important;
    border-left: 5px solid #7C3AED;
    padding-left: 0.75rem !important;
    margin-top: 2.2rem !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #ECE9FB;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 2px 10px rgba(124, 58, 237, 0.06);
}
[data-testid="stMetricValue"] { color: #5B21B6; font-weight: 800; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(120deg, #7C3AED, #6D28D9);
    color: #fff;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    padding: 0.65rem 1.1rem;
    transition: all 0.16s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.32);
}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #C4B5FD;
    border-radius: 16px;
    background: #FAF9FF;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #FAF9FF; border-right: 1px solid #ECE9FB; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h3 { color: #5B21B6; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# Citation registry - every external method/tool this app uses,
# its formal reference, and a ready-to-paste Methods sentence.
# ============================================================
TOOL_CITATIONS = {
    "de": {
        "label": "Differential Expression",
        "tools": "PyDESeq2 / DESeq2",
        "methods": "Differential expression analysis was performed using PyDESeq2 "
                   "(Muzellec et al., 2023), a Python implementation of the DESeq2 "
                   "negative-binomial framework (Love et al., 2014).",
        "refs": [
            "Muzellec B, Teleńczuk M, Cabeli V, Andreux M. PyDESeq2: a python package for bulk RNA-seq differential expression analysis. Bioinformatics. 2023;39(9):btad547.",
            "Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biology. 2014;15(12):550.",
        ],
    },
    "enrichment": {
        "label": "GO / KEGG Enrichment",
        "tools": "Enrichr",
        "methods": "Gene Ontology and KEGG pathway over-representation analysis was "
                   "performed with Enrichr (Chen et al., 2013; Kuleshov et al., 2016).",
        "refs": [
            "Chen EY, Tan CM, Kou Y, et al. Enrichr: interactive and collaborative HTML5 gene list enrichment analysis tool. BMC Bioinformatics. 2013;14:128.",
            "Kuleshov MV, Jones MR, Rouillard AD, et al. Enrichr: a comprehensive gene set enrichment analysis web server 2016 update. Nucleic Acids Research. 2016;44(W1):W90-W97.",
        ],
    },
    "gsea": {
        "label": "GSEA",
        "tools": "GSEApy / MSigDB Hallmark",
        "methods": "Gene Set Enrichment Analysis was performed on genes ranked by "
                   "signed -log10(p-value) using GSEApy (Fang et al., 2023), implementing "
                   "the GSEA method (Subramanian et al., 2005) against MSigDB Hallmark "
                   "gene sets (Liberzon et al., 2015).",
        "refs": [
            "Fang Z, Liu X, Peltz G. GSEApy: a comprehensive package for performing gene set enrichment analysis in Python. Bioinformatics. 2023;39(1):btac757.",
            "Subramanian A, Tamayo P, Mootha VK, et al. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. PNAS. 2005;102(43):15545-15550.",
            "Liberzon A, Birger C, Thorvaldsdóttir H, et al. The Molecular Signatures Database Hallmark gene set collection. Cell Systems. 2015;1(6):417-425.",
        ],
    },
    "reactome": {
        "label": "Reactome Pathways",
        "tools": "Reactome",
        "methods": "Pathway enrichment was assessed using the Reactome knowledgebase "
                   "and its over-representation Analysis Service (Milacic et al., 2024).",
        "refs": [
            "Milacic M, Beavers D, Conley P, et al. The Reactome Pathway Knowledgebase 2024. Nucleic Acids Research. 2024;52(D1):D672-D678.",
        ],
    },
    "ppi": {
        "label": "PPI Network",
        "tools": "STRING",
        "methods": "Protein-protein interaction networks were retrieved from the STRING "
                   "database (Szklarczyk et al., 2023) and visualized with NetworkX "
                   "(Hagberg et al., 2008).",
        "refs": [
            "Szklarczyk D, Kirsch R, Koutrouli M, et al. The STRING database in 2023. Nucleic Acids Research. 2023;51(D1):D638-D646.",
            "Hagberg AA, Schult DA, Swart PJ. Exploring network structure, dynamics, and function using NetworkX. Proceedings of the 7th Python in Science Conference (SciPy). 2008:11-15.",
        ],
    },
    "base": {
        "label": "Core libraries",
        "tools": "pandas, NumPy, SciPy, Matplotlib, seaborn",
        "methods": "Data handling and visualization used pandas, NumPy, SciPy, "
                   "Matplotlib and seaborn.",
        "refs": [
            "Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. Nature. 2020;585:357-362.",
            "Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods. 2020;17:261-272.",
            "Hunter JD. Matplotlib: A 2D graphics environment. Computing in Science & Engineering. 2007;9(3):90-95.",
            "McKinney W. Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference (SciPy). 2010:56-61.",
            "Waskom ML. seaborn: statistical data visualization. Journal of Open Source Software. 2021;6(60):3021.",
        ],
    },
}


DE_METHODS = ["DESeq2 (recommended)", "Welch t-test", "Mann-Whitney U"]
CORRECTIONS = ["Benjamini-Hochberg (FDR)", "Bonferroni"]


def normalize_log_cpm(count_df):
    """Library-size normalize to CPM, then log2(x+1) - for QC and rank-based tests."""
    lib = count_df.sum(axis=0).replace(0, np.nan)
    cpm = count_df.divide(lib, axis=1) * 1e6
    return np.log2(cpm + 1).replace([np.inf, -np.inf], np.nan).fillna(0)


def adjust_pvalues(pvals, method):
    """Benjamini-Hochberg or Bonferroni adjustment, NaN-safe."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full(len(p), np.nan)
    mask = ~np.isnan(p)
    pm = p[mask]
    m = len(pm)
    if m == 0:
        return adj
    if method == "Bonferroni":
        adj[mask] = np.minimum(pm * m, 1.0)
        return adj
    order = np.argsort(pm)
    bh = pm[order] * m / np.arange(1, m + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(bh, 1.0)
    adj[mask] = out
    return adj


def compute_simple_de(count_df, group_map, control, treatment, test, correction):
    """Per-gene t-test / Mann-Whitney on log2-CPM. Returns a DESeq2-shaped results frame."""
    from scipy import stats

    logcpm = normalize_log_cpm(count_df)
    trt = logcpm[[s for s in count_df.columns if group_map[s] == treatment]].values
    ctrl = logcpm[[s for s in count_df.columns if group_map[s] == control]].values
    log2fc = trt.mean(axis=1) - ctrl.mean(axis=1)

    if test == "Welch t-test":
        _, pval = stats.ttest_ind(trt, ctrl, axis=1, equal_var=False)
    else:
        pval = np.array([
            stats.mannwhitneyu(trt[i], ctrl[i], alternative="two-sided")[1]
            if np.ptp(np.concatenate([trt[i], ctrl[i]])) > 0 else np.nan
            for i in range(trt.shape[0])
        ])

    df = pd.DataFrame({
        "baseMean": logcpm.mean(axis=1).values,
        "log2FoldChange": log2fc,
        "pvalue": pval,
        "padj": adjust_pvalues(pval, "Bonferroni" if correction.startswith("Bonferroni") else "BH"),
    }, index=count_df.index)
    return df.dropna(subset=["pvalue"]).sort_values("pvalue")


@st.cache_data(show_spinner="Running differential expression...")
def compute_de_cached(counts, group_items, control, treatment, method, correction, convert, id_type):
    """Cached DE (DESeq2 or simple test) + ID->symbol conversion. Status is applied later."""
    groups = dict(group_items)
    if method.startswith("DESeq2"):
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        metadata = pd.DataFrame({"condition": [groups[s] for s in counts.columns]},
                                index=counts.columns)
        count_matrix = counts.round().astype(int)
        dds = DeseqDataSet(counts=count_matrix.T, metadata=metadata, design_factors="condition")
        dds.deseq2()
        stat_res = DeseqStats(dds, contrast=["condition", treatment, control])
        stat_res.summary()
        de = stat_res.results_df.copy()
        if correction.startswith("Bonferroni"):
            de["padj"] = adjust_pvalues(de["pvalue"].values, "Bonferroni")
    else:
        de = compute_simple_de(counts, groups, control, treatment, method, correction)

    de = de.dropna(subset=["padj"]).sort_values("pvalue")

    # Keep the original IDs so downstream steps (e.g. heatmap) can look genes up in
    # the raw count matrix even after the index is relabeled to symbols.
    de["orig_id"] = de.index.astype(str)
    if convert:
        mapping = map_ids_to_symbols(de.index.tolist(), id_type)
        if mapping:
            de.index = [mapping.get(str(i), str(i)) for i in de.index]
            de = de[~de.index.duplicated(keep="first")]
    return de


# ============================================================
# Cached network/compute fetches — keyed on inputs so post-run
# tweaks (top-N, plot styling) never re-hit the external APIs.
# ============================================================
ENRICHR_LIBRARIES = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "KEGG_2021_Human",
]


@st.cache_data(show_spinner=False)
def fetch_enrichr(genes):
    """Submit a gene list to Enrichr; return a tidy DataFrame of enriched terms."""
    import requests

    genes = [g for g in genes if not str(g).isdigit()]
    if not genes:
        return None
    base = "https://maayanlab.cloud/Enrichr"
    try:
        resp = requests.post(f"{base}/addList",
                             files={"list": (None, "\n".join(genes)), "description": (None, "query")},
                             timeout=30)
        if resp.status_code != 200:
            return None
        uid = resp.json()["userListId"]
        rows = []
        for lib in ENRICHR_LIBRARIES:
            r2 = requests.get(f"{base}/enrich?userListId={uid}&backgroundType={lib}", timeout=30)
            if r2.status_code == 200:
                for t in r2.json().get(lib, []):
                    rows.append({
                        "Rank": t[0], "Term": t[1], "P-value": t[2], "Z-score": t[3],
                        "Combined Score": t[4], "Genes": ";".join(t[5]),
                        "Overlap_count": len(t[5]), "Adjusted P-value": t[6], "Library": lib,
                    })
        return pd.DataFrame(rows)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def run_gsea_cached(ranked_items, db):
    """Run GSEA prerank for one library; return a normalized Term/NES/FDR table.

    gseapy versions differ: some put the gene-set name in the index, some use
    lowercase columns. Normalize so the caller always gets the same schema.
    """
    import gseapy as gp

    rnk = pd.Series(dict(ranked_items)).sort_values(ascending=False)
    pre = gp.prerank(rnk=rnk, gene_sets=db, outdir=None, min_size=10, max_size=500,
                     permutation_num=50, seed=42, threads=1, verbose=False)

    res = pre.res2d.copy()
    if res.index.name or not isinstance(res.index, pd.RangeIndex):
        res = res.reset_index()
    cols = {c.lower(): c for c in res.columns}
    term = next((cols[k] for k in ("term", "name", "index") if k in cols), res.columns[0])
    nes = next((cols[k] for k in ("nes",) if k in cols), None)
    fdr = next((cols[k] for k in ("fdr q-val", "fdr", "fdr_q-val") if k in cols), None)
    if nes is None or fdr is None:
        return pd.DataFrame(columns=["Term", "NES", "FDR"])
    return pd.DataFrame({
        "Term": res[term].astype(str),
        "NES": pd.to_numeric(res[nes], errors="coerce"),
        "FDR": pd.to_numeric(res[fdr], errors="coerce"),
    })


@st.cache_data(show_spinner=False)
def fetch_reactome(genes):
    """Query Reactome over-representation; return a DataFrame of pathways (empty if none)."""
    import requests

    try:
        resp = requests.post("https://reactome.org/AnalysisService/identifiers/projection/",
                             headers={"Content-Type": "text/plain"}, data="\n".join(genes), timeout=60)
        if resp.status_code != 200:
            return None
        pathways = resp.json().get("pathways", [])
        return pd.DataFrame([{
            "Pathway": p["name"], "ID": p["stId"],
            "P-value": p["entities"]["pValue"], "FDR": p["entities"]["fdr"],
            "Found": p["entities"]["found"], "Total": p["entities"]["total"],
            "Ratio": p["entities"]["found"] / p["entities"]["total"] if p["entities"]["total"] > 0 else 0,
        } for p in pathways])
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fetch_string(genes, species=9606):
    """Fetch the full STRING interaction network (unfiltered); confidence filter is applied later."""
    import requests

    try:
        resp = requests.post("https://string-db.org/api/json/network",
                             data={"identifiers": "\r".join(genes), "species": species,
                                   "caller_identity": "rnaseq_app"}, timeout=60)
        if resp.status_code != 200:
            return None
        interactions = resp.json()
        if not interactions:
            return None
        return pd.DataFrame(interactions)[["preferredName_A", "preferredName_B", "score"]].rename(
            columns={"preferredName_A": "Protein_A", "preferredName_B": "Protein_B", "score": "Confidence"}
        )
    except Exception:
        return None


def build_methods_text(active_keys, de_method=DE_METHODS[0], correction=CORRECTIONS[0],
                       author="", app_name="", app_url=""):
    """Assemble a Methods paragraph + numbered reference list for the steps run."""
    keys = [k for k in TOOL_CITATIONS if k in active_keys or k == "base"]
    corr_phrase = "Benjamini-Hochberg FDR" if correction.startswith("Benjamini") else "Bonferroni"

    parts, refs, seen = [], [], set()
    for k in keys:
        if k == "de" and not de_method.startswith("DESeq2"):
            test_phrase = "Welch's t-test" if "Welch" in de_method else "the Mann-Whitney U test"
            parts.append(
                f"Differential expression was assessed on log2 counts-per-million values "
                f"using {test_phrase} per gene, with {corr_phrase} multiple-testing "
                f"correction (Virtanen et al., 2020)."
            )
        else:
            text = TOOL_CITATIONS[k]["methods"]
            if k == "de":
                text += f" Multiple testing was controlled using the {corr_phrase} procedure."
            parts.append(text)
        for r in TOOL_CITATIONS[k]["refs"]:
            if r not in seen:
                seen.add(r)
                refs.append(r)

    methods = " ".join(parts)
    ref_block = "\n".join(f"{i}. {r}" for i, r in enumerate(refs, 1))

    cite_self = ""
    if author.strip():
        name = app_name.strip() or "RNA-seq Analysis Pipeline"
        citation = f"{author.strip()}. {name} [web application]. 2026."
        if app_url.strip():
            citation += f" Available from: {app_url.strip()}."
        cite_self = (
            f"\nTO CITE THIS WEBSITE\n\n{citation} When reporting results produced here, "
            f"please credit this website and also cite the tools and packages listed "
            f"in the references below.\n"
        )
    return f"METHODS\n\n{methods}\n{cite_self}\nREFERENCES\n\n{ref_block}\n"


def store_fig(store, name, fig):
    """Render-then-free: save a figure to PNG/PDF bytes and close it to release memory."""
    import matplotlib.pyplot as plt

    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    pdf = io.BytesIO()
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    store[name] = (png.getvalue(), pdf.getvalue())
    plt.close(fig)


def download_table(df, basename, key, label="this table"):
    """Render side-by-side CSV and Excel download buttons for a DataFrame."""
    c1, c2 = st.columns(2)
    c1.download_button(f"Download {label} (CSV)", df.to_csv().encode(),
                       f"{basename}.csv", "text/csv", key=f"{key}_csv", use_container_width=True)
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=basename[:31])
    c2.download_button(f"Download {label} (Excel)", xbuf.getvalue(), f"{basename}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=f"{key}_xlsx", use_container_width=True)


# ============================================================
# Selectable pipeline steps (shared by the landing page and the
# post-upload panel; session_state keeps both views in sync)
# ============================================================
STEPS = [
    ("qc", "QC: PCA & correlation", "Sample quality check before testing"),
    ("de", "1. Differential Expression", "PyDESeq2 / t-test / Mann-Whitney"),
    ("volcano", "2. Volcano Plot", "Significance vs fold change"),
    ("heatmap", "3. Heatmap", "Top DEGs, z-scored"),
    ("enrichment", "4. GO / KEGG Enrichment", "Enrichr functional terms"),
    ("gsea", "5. GSEA", "Ranked gene-set enrichment"),
    ("reactome", "6. Reactome Pathways", "Curated pathway hits"),
    ("splicing", "7. Splicing Factors", "hnRNPs, SR proteins, spliceosome"),
    ("drug", "8. Drug Targets", "Actionable gene-drug links"),
    ("tf", "9. Transcription Factors", "Regulatory program shifts"),
    ("ppi", "10. PPI Network", "STRING interaction hubs"),
]


def init_step_state():
    for key, _, _ in STEPS:
        st.session_state.setdefault(f"step_{key}", True)


def render_step_selector(ncols=3):
    """Grid of bordered, checkbox-backed cards for choosing which steps run."""
    cols = st.columns(ncols)
    for i, (key, label, desc) in enumerate(STEPS):
        with cols[i % ncols].container(border=True):
            st.checkbox(label, key=f"step_{key}")
            st.caption(desc)


# ============================================================
# Helper: Entrez ID to Gene Symbol conversion
# ============================================================
@st.cache_data(show_spinner=False)
def convert_entrez_to_symbol(entrez_ids):
    """Convert Entrez IDs to gene symbols using NCBI API."""
    import requests
    mapping = {}
    ids = [str(x) for x in entrez_ids if str(x).isdigit()]
    if not ids:
        return mapping

    # Process in batches of 500
    for i in range(0, len(ids), 500):
        batch = ids[i:i+500]
        try:
            resp = requests.post(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                data={"db": "gene", "id": ",".join(batch), "retmode": "json"},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                for uid, info in data.get("result", {}).items():
                    if uid == "uids":
                        continue
                    if isinstance(info, dict) and "name" in info:
                        mapping[uid] = info["name"]
        except Exception:
            pass
    return mapping


def detect_id_type(index):
    """Guess whether gene identifiers are Entrez (numeric), Ensembl (ENSG...), or symbols."""
    sample = [str(x) for x in index[:300]]
    if not sample:
        return "Gene Symbol"
    n = len(sample)
    if sum(1 for v in sample if v.isdigit()) / n > 0.8:
        return "Entrez ID"
    if sum(1 for v in sample if v.upper().startswith("ENSG")) / n > 0.8:
        return "Ensembl ID"
    return "Gene Symbol"


@st.cache_data(show_spinner=False)
def map_ids_to_symbols(ids, id_type):
    """Return {original_id: gene_symbol} for human Entrez or Ensembl IDs (NCBI + MyGene.info)."""
    import requests

    ids = [str(x) for x in ids]
    mapping = {}
    if id_type == "Entrez ID":
        mapping.update(convert_entrez_to_symbol(ids))

    scopes = {"Entrez ID": "entrezgene", "Ensembl ID": "ensembl.gene"}.get(id_type)
    if scopes is None:
        return mapping

    # Query whatever NCBI did not resolve via MyGene (also the only path for Ensembl)
    query_for = {i: (i.split(".")[0] if id_type == "Ensembl ID" else i)
                 for i in ids if i not in mapping}
    q_to_orig = {}
    for orig, q in query_for.items():
        q_to_orig.setdefault(q, orig)

    qlist = list(q_to_orig)
    for k in range(0, len(qlist), 1000):
        batch = qlist[k:k + 1000]
        try:
            resp = requests.post(
                "https://mygene.info/v3/query",
                data={"q": ",".join(batch), "scopes": scopes,
                      "fields": "symbol", "species": "human"},
                timeout=30,
            )
            if resp.status_code == 200:
                for item in resp.json():
                    sym = item.get("symbol")
                    q = str(item.get("query"))
                    if sym and q in q_to_orig:
                        mapping[q_to_orig[q]] = sym
        except Exception:
            pass
    return mapping

# ============================================================
# Helper: safe imports
# ============================================================
def check_imports():
    missing = []
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    try:
        import seaborn
    except ImportError:
        missing.append("seaborn")
    try:
        import gseapy
    except ImportError:
        missing.append("gseapy")
    try:
        import pydeseq2
    except ImportError:
        missing.append("pydeseq2")
    try:
        import requests
    except ImportError:
        missing.append("requests")
    return missing


# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("# 🧬 RNA-seq Pipeline")
st.sidebar.caption("Configure your analysis, then upload data on the right.")
st.sidebar.markdown("---")

missing = check_imports()
if missing:
    st.sidebar.warning(f"Missing packages: {', '.join(missing)}\n\nInstall with:\n```\npip install {' '.join(missing)}\n```")

init_step_state()
st.sidebar.markdown("### Analysis Steps")
st.sidebar.caption("Pick steps on the main page cards. Quick actions:")
sb_all, sb_none = st.sidebar.columns(2)
if sb_all.button("Select all", use_container_width=True):
    for step_key, _, _ in STEPS:
        st.session_state[f"step_{step_key}"] = True
if sb_none.button("Clear all", use_container_width=True):
    for step_key, _, _ in STEPS:
        st.session_state[f"step_{step_key}"] = False

st.sidebar.markdown("---")
st.sidebar.markdown("### Significance Thresholds")
pval_cutoff = st.sidebar.slider("Adjusted P-value cutoff", 0.001, 0.1, 0.05, 0.001,
                                help="Genes with padj below this are called significant.")
lfc_cutoff = st.sidebar.slider("|log2 Fold Change| cutoff", 0.0, 2.0, 1.0, 0.1,
                               help="Minimum absolute log2 fold change to call a gene up/down.")

st.sidebar.markdown("### Statistical Test")
de_method = st.sidebar.selectbox(
    "Differential expression test", DE_METHODS,
    help="DESeq2: negative-binomial model on raw counts (best for RNA-seq replicates). "
         "Welch t-test / Mann-Whitney: simpler tests on log2-CPM - fast, fewer assumptions, "
         "but less powerful with few replicates.",
)
correction = st.sidebar.selectbox(
    "Multiple-testing correction", CORRECTIONS,
    help="Benjamini-Hochberg controls the false discovery rate (standard for RNA-seq). "
         "Bonferroni controls the family-wise error rate (stricter).",
)

st.sidebar.markdown("### Cite This Website")
# Fixed author credit - not user-editable
author_name = "Khalid Al-Abdulla"
st.sidebar.markdown(f"**Author:** {author_name}")
app_name = st.sidebar.text_input("Website / app name", "hnRNPB1 RNA-seq Analysis Pipeline")
app_url = st.sidebar.text_input(
    "Website URL", "http://localhost:8501",
    help="Where your website/app is hosted. Included in the citation so others can "
         "credit and find it, alongside the underlying tool/package citations.",
)

st.sidebar.markdown("---")
with st.sidebar.expander("About & How to Cite"):
    st.markdown(
        "This website orchestrates established, peer-reviewed tools. "
        "If you use its output, **cite this website** (author and URL set on the left) "
        "**and** the underlying methods. A ready-to-paste citation plus the full "
        "reference list is generated for the steps you run and included in your "
        "results download.\n\n"
        "**Methods used:**"
    )
    for entry in TOOL_CITATIONS.values():
        st.markdown(f"- **{entry['label']}** - {entry['tools']}")
    st.caption(
        "Authorship of *your* manuscript follows ICMJE criteria: substantial "
        "contribution to design/analysis/interpretation, drafting/revising, final "
        "approval, and accountability. Discuss authorship with your supervisor."
    )

# ============================================================
# Main
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🧬 RNA-seq Analysis Pipeline</h1>
        <p>From raw feature counts to publication-ready figures - differential expression,
        pathway enrichment, GSEA, splicing factors, drug targets, and interaction networks,
        all in one click.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- File upload ---
col1, col2 = st.columns([1.1, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload feature counts (CSV / TSV / XLSX)",
        type=["csv", "tsv", "txt", "xlsx"],
        help="Rows = genes, Columns = samples. First column = gene IDs."
    )
with col2:
    st.markdown("**Expected format**")
    st.code("GeneID  Control_1  Control_2  Control_3  Treatment_1  Treatment_2  Treatment_3", language=None)
    st.caption("First column = gene IDs · remaining columns = raw counts per sample.")

if uploaded_file is None:
    st.markdown(
        '<div style="margin-top:1.4rem;font-weight:700;font-size:1.15rem;color:#1E293B;">'
        "Choose what you'll get</div>",
        unsafe_allow_html=True,
    )
    st.caption("Tick the analyses you want to run. You can adjust them again after uploading.")
    render_step_selector()
    st.info("Upload a feature counts file above to begin.")
    st.stop()

# --- Parse file ---
@st.cache_data
def load_data(file):
    name = file.name.lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(file, index_col=0)
    elif name.endswith(".tsv") or name.endswith(".txt"):
        df = pd.read_csv(file, sep="\t", index_col=0)
    else:
        df = pd.read_csv(file, index_col=0)
    # Drop non-numeric columns (like gene length, chr, etc.)
    df = df.select_dtypes(include=[np.number])
    # Drop rows with all zeros
    df = df.loc[df.sum(axis=1) > 0]
    return df

counts = load_data(uploaded_file)

st.header("Data Preview")
mcol1, mcol2 = st.columns(2)
mcol1.metric("Genes detected", f"{counts.shape[0]:,}")
mcol2.metric("Samples", counts.shape[1])
st.dataframe(counts.head(10), use_container_width=True)

# --- Sample grouping ---
st.header("Define Groups")
gname_col1, gname_col2 = st.columns(2)
gname_control = gname_col1.text_input("Name for baseline group", "Control").strip() or "Control"
gname_treatment = gname_col2.text_input("Name for compared group", "Treatment").strip() or "Treatment"

if gname_control == gname_treatment:
    st.error("The two group names must be different.")
    st.stop()

st.markdown(f"Assign each sample to **{gname_control}** or **{gname_treatment}**. "
            "Groups are auto-guessed from sample names - adjust as needed.")

samples = counts.columns.tolist()
group_assignments = {}

# Try to auto-detect groups
cols = st.columns(min(len(samples), 4))
for i, sample in enumerate(samples):
    with cols[i % len(cols)]:
        # Guess default group
        name_lower = sample.lower()
        default = gname_control
        for kw in ["treat", "ko", "kd", "si", "shrna", "amo", "b1", "drug", "mut", "test", "exp"]:
            if kw in name_lower:
                default = gname_treatment
                break
        group_assignments[sample] = st.selectbox(
            sample, [gname_control, gname_treatment],
            index=0 if default == gname_control else 1,
            key=f"group_{sample}"
        )

control_samples = [s for s, g in group_assignments.items() if g == gname_control]
treatment_samples = [s for s, g in group_assignments.items() if g == gname_treatment]

if len(control_samples) < 2 or len(treatment_samples) < 2:
    st.error("Need at least 2 samples per group.")
    st.stop()

st.success(f"{gname_control}: {len(control_samples)} samples | {gname_treatment}: {len(treatment_samples)} samples")

# --- Gene ID type ---
st.header("Gene Identifiers")
detected_id_type = detect_id_type(counts.index)
id_choice = st.radio(
    "Gene ID type in your data:",
    ["Auto-detect (recommended)", "Gene Symbol", "Ensembl ID", "Entrez ID"],
    horizontal=True,
)
gene_id_type = detected_id_type if id_choice.startswith("Auto") else id_choice
convert_ids = gene_id_type in ("Entrez ID", "Ensembl ID")
if convert_ids:
    st.caption(f"Detected **{gene_id_type}**. These will be auto-converted to gene "
               "symbols so enrichment, GSEA, and network steps work.")
else:
    st.caption(f"Detected **{gene_id_type}**. No conversion needed.")

# --- Analysis steps (same selector as the landing page, kept in sync) ---
st.header("Analysis Steps to Run")
st.caption("These carry over from the landing page. Adjust any time, then run.")
render_step_selector()

run_qc = st.session_state["step_qc"]
run_de = st.session_state["step_de"]
run_volcano = st.session_state["step_volcano"]
run_enrichment = st.session_state["step_enrichment"]
run_gsea = st.session_state["step_gsea"]
run_reactome = st.session_state["step_reactome"]
run_splicing = st.session_state["step_splicing"]
run_drug = st.session_state["step_drug"]
run_heatmap = st.session_state["step_heatmap"]
run_tf = st.session_state["step_tf"]
run_ppi = st.session_state["step_ppi"]

# --- Run button ---
# Persist "ran" in session_state so post-run controls (top-N, plot customization)
# survive Streamlit reruns instead of vanishing when the button returns False.
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Run Analysis", type="primary", use_container_width=True):
    st.session_state["analysis_ran"] = True

if not st.session_state.get("analysis_ran"):
    st.info("Ready when you are - press **Run Analysis** to start the pipeline.")
    st.stop()

# ============================================================
# Store all results for export
# ============================================================
results = {}
figures = {}

# Free matplotlib figures left open by the previous rerun. Without this, every
# threshold tweak leaks ~12 figures and the app eventually runs out of memory.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.close("all")

progress = st.progress(0, text="Starting analysis...")

# ============================================================
# QC: PCA & SAMPLE CORRELATION
# ============================================================
if run_qc:
    progress.progress(3, text="Running QC (PCA & sample correlation)...")
    st.header("Quality Control: PCA & Sample Correlation")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    logcpm_qc = normalize_log_cpm(counts)
    top_var = logcpm_qc.loc[logcpm_qc.var(axis=1).sort_values(ascending=False).head(500).index]
    centered = top_var.subtract(top_var.mean(axis=1), axis=0).T.values
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # The default LAPACK driver (gesdd) can fail to converge; gesvd is slower but robust
        from scipy.linalg import svd as _scipy_svd
        u, s, _ = _scipy_svd(centered, full_matrices=False, lapack_driver="gesvd")
    pcs = u * s
    var_pct = (s ** 2) / (s ** 2).sum() * 100

    qc_col1, qc_col2 = st.columns(2)

    fig_pca, ax_pca = plt.subplots(figsize=(6, 5))
    for grp, color in [(gname_control, "#3498db"), (gname_treatment, "#e74c3c")]:
        idx = [i for i, sample in enumerate(counts.columns) if group_assignments[sample] == grp]
        if idx:
            ax_pca.scatter(pcs[idx, 0], pcs[idx, 1], s=140, c=color,
                           edgecolors="white", linewidth=1.5, label=grp, zorder=3)
    for i, sample in enumerate(counts.columns):
        ax_pca.annotate(sample, (pcs[i, 0], pcs[i, 1]), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")
    ax_pca.set_xlabel(f"PC1 ({var_pct[0]:.1f}%)", fontsize=11, fontweight="bold")
    ax_pca.set_ylabel(f"PC2 ({var_pct[1]:.1f}%)", fontsize=11, fontweight="bold")
    ax_pca.set_title("PCA (top 500 variable genes)", fontsize=12, fontweight="bold")
    ax_pca.legend()
    ax_pca.spines["top"].set_visible(False)
    ax_pca.spines["right"].set_visible(False)
    qc_col1.pyplot(fig_pca)
    store_fig(figures, "pca", fig_pca)

    corr = logcpm_qc.corr(method="spearman")
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, cmap="viridis", annot=True, fmt=".2f", square=True,
                ax=ax_corr, cbar_kws={"label": "Spearman r"})
    ax_corr.set_title("Sample-sample correlation", fontsize=12, fontweight="bold")
    qc_col2.pyplot(fig_corr)
    store_fig(figures, "sample_correlation", fig_corr)

    st.caption("Samples should cluster by group on PC1/PC2; low cross-group correlation flags outliers.")

# ============================================================
# 1. DIFFERENTIAL EXPRESSION (PyDESeq2)
# ============================================================
if run_de:
    progress.progress(5, text="Running differential expression...")
    st.header("1. Differential Expression")

    try:
        de_results = compute_de_cached(
            counts, tuple(sorted(group_assignments.items())),
            gname_control, gname_treatment, de_method, correction,
            convert_ids, gene_id_type,
        ).copy()
        st.caption(f"Test: {de_method}  |  Correction: {correction}")

        # Classify genes against the current thresholds (cheap, kept out of the cache)
        de_results["status"] = "Not significant"
        de_results.loc[
            (de_results["padj"] < pval_cutoff) & (de_results["log2FoldChange"] > lfc_cutoff), "status"
        ] = "Upregulated"
        de_results.loc[
            (de_results["padj"] < pval_cutoff) & (de_results["log2FoldChange"] < -lfc_cutoff), "status"
        ] = "Downregulated"

        n_up = (de_results["status"] == "Upregulated").sum()
        n_down = (de_results["status"] == "Downregulated").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total DEGs", n_up + n_down)
        col2.metric("Upregulated", n_up)
        col3.metric("Downregulated", n_down)

        sig_table = de_results[de_results["status"] != "Not significant"].drop(columns=["orig_id"], errors="ignore")
        top_n = st.number_input("Show top N significant genes in table", 5, 500, 20, 5)
        st.dataframe(sig_table.head(int(top_n)), use_container_width=True)

        results["de_results"] = de_results
        results["de_significant"] = de_results[de_results["status"] != "Not significant"]

    except ImportError:
        st.error("PyDESeq2 not installed. Run: `pip install pydeseq2`")
        st.stop()
    except Exception as e:
        st.error(f"DE analysis failed: {e}")
        st.stop()
else:
    st.warning("Differential expression skipped. Upload pre-computed DE results to continue.")
    st.stop()

# ============================================================
# 2. VOLCANO PLOT
# ============================================================
if run_volcano and "de_results" in results:
    progress.progress(15, text="Generating volcano plot...")
    st.header("2. Volcano Plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    de = results["de_results"].copy()
    de["neg_log10_p"] = -np.log10(de["pvalue"].clip(lower=1e-300))

    with st.expander("Customize plot (axes, labels, style)"):
        c1, c2, c3 = st.columns(3)
        vol_title = c1.text_input("Title", f"Volcano Plot: {gname_treatment} vs {gname_control}", key="vol_title")
        vol_point_size = c1.slider("Point size", 5, 60, 20, 5, key="vol_psize")
        vol_label_n = c2.slider("Genes to label (per direction)", 0, 25, 8, 1, key="vol_labn")
        data_xmax = float(np.ceil(de["log2FoldChange"].abs().max()))
        data_ymax = float(np.ceil(de["neg_log10_p"].max()))
        vol_xlim = c2.slider("X-axis range (|log2FC|)", 0.5, max(2.0, data_xmax), data_xmax, 0.5, key="vol_xlim")
        vol_ymax = c3.slider("Y-axis max (-log10 p)", 1.0, max(2.0, data_ymax), data_ymax, 1.0, key="vol_ymax")
        vol_label_col = c3.color_picker("Up color", "#e74c3c", key="vol_upcol")
        vol_down_col = c3.color_picker("Down color", "#3498db", key="vol_downcol")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Not significant - light gray, small
    ns = de[de["status"] == "Not significant"]
    ax.scatter(ns["log2FoldChange"], ns["neg_log10_p"], s=6, c="#d5d5d5", alpha=0.4, zorder=1)

    # Upregulated - red
    up = de[de["status"] == "Upregulated"]
    ax.scatter(up["log2FoldChange"], up["neg_log10_p"], s=vol_point_size, c=vol_label_col, alpha=0.75, zorder=2)

    # Downregulated - blue
    down = de[de["status"] == "Downregulated"]
    ax.scatter(down["log2FoldChange"], down["neg_log10_p"], s=vol_point_size, c=vol_down_col, alpha=0.75, zorder=2)

    # Cutoff lines
    ax.axhline(-np.log10(pval_cutoff), ls="--", c="gray", lw=0.7, alpha=0.6)
    ax.axvline(lfc_cutoff, ls="--", c="gray", lw=0.7, alpha=0.6)
    ax.axvline(-lfc_cutoff, ls="--", c="gray", lw=0.7, alpha=0.6)

    # Label top genes with colored text (no boxes)
    top_up = up.nlargest(vol_label_n, "neg_log10_p")
    top_down = down.nlargest(vol_label_n, "neg_log10_p")

    for idx, row in top_up.iterrows():
        ax.annotate(idx, (row["log2FoldChange"], row["neg_log10_p"]),
                     fontsize=7.5, fontweight="bold", color=vol_label_col,
                     ha="left", va="bottom", xytext=(4, 4), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=vol_label_col, alpha=0.85, lw=0.5))

    for idx, row in top_down.iterrows():
        ax.annotate(idx, (row["log2FoldChange"], row["neg_log10_p"]),
                     fontsize=7.5, fontweight="bold", color=vol_down_col,
                     ha="right", va="bottom", xytext=(-4, 4), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=vol_down_col, alpha=0.85, lw=0.5))

    ax.set_xlabel(f"log$_2$ Fold Change ({gname_treatment} vs {gname_control})", fontsize=13, fontweight="bold")
    ax.set_ylabel("-log$_{10}$(P-value)", fontsize=13, fontweight="bold")
    ax.set_title(vol_title, fontsize=14, fontweight="bold")
    ax.set_xlim(-vol_xlim, vol_xlim)
    ax.set_ylim(0, vol_ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    # Legend at bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d5d5d5', markersize=8,
               label=f'Not significant (n={len(ns)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=vol_down_col, markersize=8,
               label=f'Downregulated (n={len(down)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=vol_label_col, markersize=8,
               label=f'Upregulated (n={len(up)})'),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=10, framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    st.pyplot(fig)
    store_fig(figures, "volcano_plot", fig)

    vol_data = de[["log2FoldChange", "pvalue", "padj", "status", "neg_log10_p"]]
    st.markdown("**Download volcano data** (all genes with fold change, p-values, and significance call)")
    download_table(vol_data, "volcano_data", "vol", label="volcano data")

# ============================================================
# 3. HEATMAP - TOP DEGs
# ============================================================
if run_heatmap and "de_results" in results:
    progress.progress(20, text="Generating heatmap...")
    st.header("3. Heatmap - Top DEGs")

    import matplotlib.pyplot as plt
    import seaborn as sns

    de = results["de_results"]
    sig = de[de["status"] != "Not significant"].copy()

    hc1, hc2, hc3 = st.columns(3)
    n_top = hc1.slider("Top genes per direction", 10, 50, 25, 5, key="heatmap_n")
    heat_scale = hc2.slider(
        "Color scale (±z-score)", 1.0, 4.0, 2.0, 0.5, key="heat_scale",
        help="Clamps the color range so every heatmap uses the same scale. "
             "Lower = more uniform colors; a few extreme genes won't dominate.",
    )
    heat_cmap = hc3.selectbox("Color map", ["RdBu_r", "vlag", "coolwarm", "viridis", "magma"], key="heat_cmap")

    top_up = sig[sig["status"] == "Upregulated"].nsmallest(n_top, "pvalue")
    top_down = sig[sig["status"] == "Downregulated"].nsmallest(n_top, "pvalue")
    top_genes = pd.concat([top_up, top_down])

    try:
        # Look genes up in the raw counts by their ORIGINAL id (counts aren't relabeled),
        # then display them with their gene symbol. No whole-matrix ID conversion needed.
        counts_idx = counts.copy()
        counts_idx.index = counts_idx.index.astype(str)
        if "orig_id" in top_genes.columns:
            wanted = [(o, sym) for o, sym in zip(top_genes["orig_id"].astype(str), top_genes.index)
                      if o in counts_idx.index]
        else:
            wanted = [(g, g) for g in top_genes.index.astype(str) if g in counts_idx.index]

        if wanted:
            heat_data = counts_idx.loc[[o for o, _ in wanted]].copy()
            heat_data.index = [sym for _, sym in wanted]
            heat_data = heat_data[~heat_data.index.duplicated(keep="first")]

            # Log2 transform + z-score per gene
            heat_data = np.log2(heat_data + 1)
            heat_data = heat_data.subtract(heat_data.mean(axis=1), axis=0).div(heat_data.std(axis=1), axis=0)
            heat_data = heat_data.replace([np.inf, -np.inf], np.nan).fillna(0)

            col_colors = [("#3498db" if group_assignments[s] == gname_control else "#e74c3c") for s in heat_data.columns]

            g = sns.clustermap(heat_data, cmap=heat_cmap, center=0,
                               vmin=-heat_scale, vmax=heat_scale,
                               col_colors=col_colors, row_cluster=len(heat_data) > 1, col_cluster=False,
                               yticklabels=True, xticklabels=True,
                               figsize=(max(6, len(heat_data.columns) * 0.8), max(8, len(heat_data) * 0.25)),
                               cbar_pos=(0.02, 0.83, 0.03, 0.13),
                               dendrogram_ratio=(0.14, 0.04))
            g.ax_cbar.set_title("Z-score", fontsize=9, pad=6)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=9, rotation=45, ha="right")
            g.fig.suptitle(f"Top {len(heat_data)} DEGs - Z-score normalized", fontsize=13, fontweight="bold", y=1.02)
            st.pyplot(g.fig)
            store_fig(figures, "heatmap", g.fig)
        else:
            st.info("No significant genes were found in the count matrix, so no heatmap was drawn.")
    except Exception as e:
        st.warning(f"Heatmap could not be generated: {e}")

# ============================================================
# 4. GO/KEGG ENRICHMENT (Enrichr API)
# ============================================================
if run_enrichment and "de_results" in results:
    progress.progress(25, text="Running GO/KEGG enrichment...")
    st.header("4. GO & KEGG Enrichment")

    import requests

    de = results["de_results"]
    up_genes = de[de["status"] == "Upregulated"].index.tolist()
    down_genes = de[de["status"] == "Downregulated"].index.tolist()

    enrichment_results = {}

    # Filter out numeric-only gene names (Entrez IDs that weren't converted)
    up_genes_clean = [g for g in up_genes if not str(g).isdigit()]
    down_genes_clean = [g for g in down_genes if not str(g).isdigit()]

    # If most genes are still numeric, warn the user
    if len(up_genes_clean) == 0 and len(down_genes_clean) == 0:
        st.warning("All gene IDs appear to be numeric (Entrez IDs). Enrichr requires gene symbols. "
                   "Please enable 'Convert Entrez IDs to gene symbols' and re-run.")
    else:
        st.write(f"Submitting {len(up_genes_clean)} upregulated and {len(down_genes_clean)} downregulated gene symbols to Enrichr")
        st.write(f"Example genes: {', '.join(up_genes_clean[:5])}")

    if len(up_genes_clean) > 0:
        with st.spinner("Querying Enrichr for upregulated genes..."):
            up_enrich = fetch_enrichr(tuple(up_genes_clean))
            if up_enrich is not None and len(up_enrich) > 0:
                up_enrich["Direction"] = "Upregulated"
                enrichment_results["up"] = up_enrich
            else:
                st.warning("Enrichr returned no results for upregulated genes.")

    if len(down_genes_clean) > 0:
        with st.spinner("Querying Enrichr for downregulated genes..."):
            down_enrich = fetch_enrichr(tuple(down_genes_clean))
            if down_enrich is not None and len(down_enrich) > 0:
                down_enrich["Direction"] = "Downregulated"
                enrichment_results["down"] = down_enrich
            else:
                st.warning("Enrichr returned no results for downregulated genes.")

    if enrichment_results:
        all_enrich = pd.concat(enrichment_results.values(), ignore_index=True)
        results["enrichment"] = all_enrich

        enrich_top_n = st.slider("Top pathways/terms to show (per direction)", 5, 40, 15, 1,
                                 key="enrich_top_n")

        # Plot top terms
        import matplotlib.pyplot as plt
        import math

        for direction, label, color in [("Upregulated", "Upregulated", "#e74c3c"),
                                         ("Downregulated", "Downregulated", "#3498db")]:
            sub = all_enrich[(all_enrich["Direction"] == direction) & (all_enrich["P-value"] < 0.05)]
            if len(sub) == 0:
                continue

            top = sub.nsmallest(enrich_top_n, "P-value").copy()
            top["Short_Term"] = top["Term"].apply(lambda t: t.split("(GO")[0].strip() if "(GO" in t else t[:50])
            top["neg_log10_p"] = -top["P-value"].apply(lambda x: math.log10(max(x, 1e-30)))
            top = top.sort_values("Combined Score", ascending=True)

            fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 2))
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=top["neg_log10_p"].min() - 0.2,
                                     vmax=top["neg_log10_p"].max() + 0.2)

            scatter = ax.scatter(
                top["Combined Score"], range(len(top)),
                s=top["Overlap_count"] * 40, c=top["neg_log10_p"],
                cmap=plt.cm.RdYlBu_r, norm=norm,
                alpha=0.85, edgecolors="black", linewidth=0.6, zorder=3,
            )
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["Short_Term"], fontsize=9)
            ax.set_xlabel("Combined Score", fontsize=11)
            ax.set_title(f"GO & KEGG Enrichment - {label} genes", fontsize=12, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)
            cbar.set_label("-log$_{10}$(P-value)", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            store_fig(figures, f"enrichment_{direction}", fig)

# ============================================================
# 4. GSEA
# ============================================================
if run_gsea and "de_results" in results:
    progress.progress(40, text="Running GSEA...")
    st.header("5. Gene Set Enrichment Analysis (GSEA)")

    try:
        import gseapy as gp

        de = results["de_results"].copy()
        de["rank_metric"] = -np.log10(de["pvalue"].clip(lower=1e-300)) * np.sign(de["log2FoldChange"])
        de = de.sort_values("rank_metric", ascending=False)

        ranked = de["rank_metric"].dropna()
        ranked = ranked[~ranked.index.duplicated()]

        # Hallmark only on the hosted app - KEGG GSEA is heavy and KEGG terms are already
        # covered by the Enrichr GO/KEGG step; this keeps GSEA within free-tier CPU limits.
        gsea_dbs = ["MSigDB_Hallmark_2020"]
        gsea_results = {}

        ranked_items = tuple(ranked.items())
        for db in gsea_dbs:
            with st.spinner(f"Running GSEA against {db}..."):
                try:
                    res_df = run_gsea_cached(ranked_items, db)
                    if len(res_df) > 0:
                        gsea_results[db] = res_df
                except Exception as e:
                    st.warning(f"GSEA failed for {db}: {e}")

        if gsea_results:
            all_gsea = pd.concat(gsea_results.values(), ignore_index=True)
            results["gsea"] = all_gsea

            st.caption(
                "A **gene set** is a predefined group of genes sharing a function or pathway "
                "(here: MSigDB Hallmark and KEGG). GSEA asks whether each set is shifted toward "
                "the top (up) or bottom (down) of your genes ranked by fold change. "
                "**NES** = Normalized Enrichment Score: positive (red) = enriched in "
                f"{gname_treatment}, negative (blue) = enriched in {gname_control}."
            )

            import matplotlib.pyplot as plt

            sig = all_gsea[all_gsea["FDR"] < 0.25].dropna(subset=["NES"]).copy()
            if len(sig) > 0:
                sig = sig.sort_values("NES")
                top_sig = pd.concat([sig.head(10), sig.tail(10)]).drop_duplicates()

                fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_sig) + 2))
                colors = ["#3498db" if x < 0 else "#e74c3c" for x in top_sig["NES"]]
                ax.barh(range(len(top_sig)), top_sig["NES"], color=colors, edgecolor="white")
                ax.set_yticks(range(len(top_sig)))
                ax.set_yticklabels(top_sig["Term"].astype(str).str[:50], fontsize=8)
                ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=11)
                ax.set_title("GSEA - Significant Gene Sets (FDR < 0.25)", fontsize=12, fontweight="bold")
                ax.axvline(0, color="black", linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                store_fig(figures, "gsea_nes", fig)
            else:
                st.info("No gene sets reached FDR < 0.25.")
        else:
            st.warning("GSEA returned no results.")

    except ImportError:
        st.warning("gseapy not installed. Skipping GSEA. Install with: `pip install gseapy`")

# ============================================================
# 5. REACTOME PATHWAYS
# ============================================================
if run_reactome and "de_results" in results:
    progress.progress(55, text="Querying Reactome...")
    st.header("6. Reactome Pathway Enrichment")

    import requests
    import math

    de = results["de_results"]
    sig_genes = de[de["status"] != "Not significant"].index.tolist()

    if len(sig_genes) > 0:
        with st.spinner("Querying Reactome API..."):
            reactome_df = fetch_reactome(tuple(sig_genes))

        if reactome_df is None:
            st.warning("Reactome query failed.")
        elif reactome_df.empty:
            st.warning("No pathways returned from Reactome.")
        else:
            reactome_sig = reactome_df[reactome_df["FDR"] < 0.05]
            results["reactome"] = reactome_df

            st.write(f"**{len(reactome_sig)}** significant pathways (FDR < 0.05)")

            if len(reactome_sig) > 0:
                import matplotlib.pyplot as plt

                top = reactome_sig.nsmallest(20, "FDR").copy()
                top["neg_log10_fdr"] = -top["FDR"].apply(lambda x: math.log10(max(x, 1e-30)))
                top = top.sort_values("Ratio", ascending=True)

                fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 2))
                import matplotlib.colors as mcolors
                norm = mcolors.Normalize(vmin=top["neg_log10_fdr"].min() - 0.2,
                                         vmax=top["neg_log10_fdr"].max() + 0.2)
                scatter = ax.scatter(
                    top["Ratio"], range(len(top)),
                    s=top["Found"] * 15, c=top["neg_log10_fdr"],
                    cmap=plt.cm.RdYlBu_r, norm=norm,
                    alpha=0.85, edgecolors="black", linewidth=0.6,
                )
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels(top["Pathway"].str[:55], fontsize=8)
                ax.set_xlabel("Gene Ratio", fontsize=11)
                ax.set_title("Reactome Pathway Enrichment (FDR < 0.05)", fontsize=12, fontweight="bold")
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)
                cbar.set_label("-log$_{10}$(FDR)", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                store_fig(figures, "reactome", fig)
            else:
                st.info("No pathways reached FDR < 0.05.")

# ============================================================
# 6. SPLICING FACTOR ANALYSIS
# ============================================================
if run_splicing and "de_results" in results:
    progress.progress(65, text="Analyzing splicing factors...")
    st.header("7. Splicing Factor Analysis")

    import matplotlib.pyplot as plt

    # Known splicing factors
    sr_proteins = ["SRSF1","SRSF2","SRSF3","SRSF4","SRSF5","SRSF6","SRSF7","SRSF8","SRSF9","SRSF10","SRSF11","SRSF12"]
    hnrnps = ["HNRNPA0","HNRNPA1","HNRNPA2B1","HNRNPA3","HNRNPC","HNRNPD","HNRNPDL","HNRNPF","HNRNPH1","HNRNPH2","HNRNPH3","HNRNPK","HNRNPL","HNRNPLL","HNRNPM","HNRNPR","HNRNPU","HNRNPUL1","HNRNPUL2"]
    core_spliceosome = ["SF3B1","SF3B2","SF3B3","SF3A1","SF3A2","SF3A3","U2AF1","U2AF2","PRPF8","PRPF3","PRPF4","PRPF6","PRPF31","PRPF40A","SNRNP200","SNRNP70","SNRPD1","SNRPD2","SNRPD3","SNRPE","SNRPF","SNRPG","SNRPA","SNRPB","SNRPB2"]
    regulators = ["RBFOX1","RBFOX2","RBFOX3","MBNL1","MBNL2","MBNL3","PTBP1","PTBP2","QKI","NOVA1","NOVA2","CELF1","CELF2","TRA2A","TRA2B","RBM3","RBM4","RBM5","RBM10","RBM15","RBM17","RBM22","RBM25","RBM39"]

    all_sf = list(set(sr_proteins + hnrnps + core_spliceosome + regulators))

    de = results["de_results"]
    sf_in_data = de[de.index.isin(all_sf)].copy()

    if len(sf_in_data) > 0:
        sf_in_data["Category"] = "Other"
        sf_in_data.loc[sf_in_data.index.isin(sr_proteins), "Category"] = "SR Protein"
        sf_in_data.loc[sf_in_data.index.isin(hnrnps), "Category"] = "hnRNP"
        sf_in_data.loc[sf_in_data.index.isin(core_spliceosome), "Category"] = "Core Spliceosome"
        sf_in_data.loc[sf_in_data.index.isin(regulators), "Category"] = "Regulator"

        sig_sf = sf_in_data[sf_in_data["pvalue"] < 0.05].sort_values("log2FoldChange")
        results["splicing_factors"] = sf_in_data

        st.write(f"**{len(sf_in_data)}** splicing factors found in dataset, **{len(sig_sf)}** nominally significant (p<0.05)")

        n_up = (sf_in_data["log2FoldChange"] > 0).sum()
        n_down = (sf_in_data["log2FoldChange"] < 0).sum()
        st.write(f"Direction bias: {n_up} trending up vs {n_down} trending down")

        if len(sig_sf) > 0:
            cat_colors = {"SR Protein": "#e74c3c", "hnRNP": "#3498db", "Core Spliceosome": "#2ecc71", "Regulator": "#9b59b6"}
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(sig_sf) + 1)))
            colors = [cat_colors.get(c, "gray") for c in sig_sf["Category"]]
            ax.barh(range(len(sig_sf)), sig_sf["log2FoldChange"], color=colors, edgecolor="white")
            ax.set_yticks(range(len(sig_sf)))
            ax.set_yticklabels(sig_sf.index, fontsize=9)
            ax.set_xlabel("log$_2$(Fold Change)", fontsize=11)
            ax.set_title("Differentially Expressed Splicing Factors (p < 0.05)", fontsize=12, fontweight="bold")
            ax.axvline(0, color="black", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for cat, color in cat_colors.items():
                ax.barh([], [], color=color, label=cat)
            ax.legend(fontsize=8, loc="lower right")

            plt.tight_layout()
            st.pyplot(fig)
            store_fig(figures, "splicing", fig)

# ============================================================
# 7. DRUG TARGET ANALYSIS
# ============================================================
if run_drug and "de_results" in results:
    progress.progress(75, text="Analyzing drug targets...")
    st.header("8. Drug Target Analysis")

    import matplotlib.pyplot as plt

    drug_targets = {
        "EGFR": ["Erlotinib", "Gefitinib", "Cetuximab"],
        "ERBB2": ["Trastuzumab", "Lapatinib", "Pertuzumab"],
        "ESR1": ["Tamoxifen", "Fulvestrant"],
        "PGR": ["Megestrol"],
        "AR": ["Enzalutamide", "Bicalutamide"],
        "BRCA1": ["Olaparib"],
        "BRCA2": ["Olaparib", "Rucaparib"],
        "PIK3CA": ["Alpelisib"],
        "MTOR": ["Everolimus", "Temsirolimus"],
        "CDK4": ["Palbociclib", "Ribociclib"],
        "CDK6": ["Palbociclib", "Abemaciclib"],
        "PARP1": ["Olaparib", "Niraparib"],
        "PARP2": ["Olaparib"],
        "VEGFA": ["Bevacizumab"],
        "TOP2A": ["Doxorubicin", "Etoposide"],
        "TUBB": ["Paclitaxel", "Docetaxel"],
        "TUBB3": ["Paclitaxel"],
        "TYMS": ["5-Fluorouracil", "Capecitabine"],
        "DHFR": ["Methotrexate"],
        "BCL2": ["Venetoclax"],
        "BRD4": ["JQ1", "OTX015"],
        "HDAC1": ["Vorinostat", "Panobinostat"],
        "HDAC2": ["Vorinostat"],
        "PLK1": ["Volasertib"],
        "AURKA": ["Alisertib"],
        "AURKB": ["Barasertib"],
        "SRC": ["Dasatinib"],
        "ABL1": ["Imatinib"],
        "JAK2": ["Ruxolitinib"],
        "TERT": ["Imetelstat"],
    }

    de = results["de_results"]
    drug_de = []
    for gene, drugs in drug_targets.items():
        if gene in de.index:
            row = de.loc[gene]
            drug_de.append({
                "Gene": gene,
                "Drugs": ", ".join(drugs),
                "log2FC": row["log2FoldChange"],
                "P-value": row["pvalue"],
                "FDR": row["padj"],
                "Status": row["status"],
            })

    if drug_de:
        drug_df = pd.DataFrame(drug_de).sort_values("log2FC")
        results["drug_targets"] = drug_df

        st.dataframe(drug_df, use_container_width=True)

        # Plot
        sig_drugs = drug_df[drug_df["P-value"] < 0.1].sort_values("log2FC")
        if len(sig_drugs) > 0:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(sig_drugs) + 1)))
            colors = ["#3498db" if x < 0 else "#e74c3c" for x in sig_drugs["log2FC"]]
            ax.barh(range(len(sig_drugs)), sig_drugs["log2FC"], color=colors, edgecolor="white")
            ax.set_yticks(range(len(sig_drugs)))
            labels = [f"{row['Gene']} ({row['Drugs'][:30]})" for _, row in sig_drugs.iterrows()]
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("log$_2$(Fold Change)", fontsize=11)
            ax.set_title("Drug Target Genes - Expression Changes", fontsize=12, fontweight="bold")
            ax.axvline(0, color="black", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            store_fig(figures, "drug_targets", fig)

# ============================================================
# 9. TRANSCRIPTION FACTOR ANALYSIS
# ============================================================
if run_tf and "de_results" in results:
    progress.progress(80, text="Analyzing transcription factors...")
    st.header("9. Transcription Factor Analysis")

    import matplotlib.pyplot as plt

    # Curated list of human transcription factors
    tf_list = [
        # General TFs
        "TP53","MYC","MYCN","JUN","JUNB","JUND","FOS","FOSB","FOSL1","FOSL2",
        "SP1","SP3","SP4","EGR1","EGR2","EGR3",
        # NF-kB family
        "NFKB1","NFKB2","RELA","RELB","REL",
        # STAT family
        "STAT1","STAT2","STAT3","STAT4","STAT5A","STAT5B","STAT6",
        # HOX family
        "HOXA1","HOXA5","HOXA9","HOXA10","HOXA11","HOXA13",
        "HOXB1","HOXB2","HOXB4","HOXB7","HOXB9","HOXB13",
        "HOXC4","HOXC6","HOXC8","HOXC9","HOXC10","HOXC11","HOXC13",
        "HOXD1","HOXD3","HOXD4","HOXD8","HOXD9","HOXD10","HOXD11","HOXD13",
        # FOX family
        "FOXA1","FOXA2","FOXC1","FOXC2","FOXD1","FOXF1","FOXF2",
        "FOXM1","FOXO1","FOXO3","FOXO4","FOXP1","FOXP2","FOXP3","FOXP4",
        # SOX family
        "SOX2","SOX4","SOX9","SOX10","SOX11","SOX17",
        # GATA family
        "GATA1","GATA2","GATA3","GATA4","GATA6",
        # Nuclear receptors
        "ESR1","ESR2","AR","PGR","NR3C1","NR4A1","NR4A2","NR4A3",
        "PPARA","PPARD","PPARG","RXRA","RARA","RARB","RARG",
        # EMT TFs
        "SNAI1","SNAI2","TWIST1","TWIST2","ZEB1","ZEB2",
        # Cell cycle / proliferation TFs
        "E2F1","E2F2","E2F3","E2F4","E2F5","E2F6","E2F7","E2F8",
        "RB1","TFDP1",
        # Signaling TFs
        "HIF1A","EPAS1","ARNT","NOTCH1","NOTCH2","NOTCH3",
        "SMAD1","SMAD2","SMAD3","SMAD4","SMAD5","SMAD7",
        # Cancer-related
        "ETS1","ETS2","ERG","ETV1","ETV4","ETV5","ELF1","ELF3",
        "IRF1","IRF3","IRF4","IRF5","IRF7","IRF8","IRF9",
        "RUNX1","RUNX2","RUNX3",
        "KLF2","KLF4","KLF5","KLF6",
        "YAP1","TEAD1","TEAD2","TEAD3","TEAD4",
        "CEBPA","CEBPB","CEBPD","CEBPG",
        "ATF1","ATF2","ATF3","ATF4","ATF6",
        "BACH1","BACH2","NFE2L2","KEAP1",
        "MYB","MYBL2","ELK1",
        "TFAP2A","TFAP2B","TFAP2C",
        "TCF7","TCF7L1","TCF7L2","LEF1",
        "ASCL1","GRHL2","XBP1","ID1","ID2","ID3","ID4",
        "SPDEF","CUX1","CUX2",
    ]

    de = results["de_results"]
    tf_in_data = de[de.index.isin(tf_list)].copy()

    if len(tf_in_data) > 0:
        sig_tf = tf_in_data[tf_in_data["pvalue"] < 0.05].sort_values("log2FoldChange")
        results["transcription_factors"] = tf_in_data

        n_up = (tf_in_data["log2FoldChange"] > 0).sum()
        n_down = (tf_in_data["log2FoldChange"] < 0).sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("TFs in dataset", len(tf_in_data))
        col2.metric("Significant (p<0.05)", len(sig_tf))
        col3.metric("Up / Down trend", f"{n_up} / {n_down}")

        if len(sig_tf) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(sig_tf) + 1)))
            colors = ["#3498db" if x < 0 else "#e74c3c" for x in sig_tf["log2FoldChange"]]
            bars = ax.barh(range(len(sig_tf)), sig_tf["log2FoldChange"], color=colors, edgecolor="white")
            ax.set_yticks(range(len(sig_tf)))
            ax.set_yticklabels(sig_tf.index, fontsize=9)
            ax.set_xlabel("log$_2$(Fold Change)", fontsize=11)
            ax.set_title("Differentially Expressed Transcription Factors (p < 0.05)", fontsize=12, fontweight="bold")
            ax.axvline(0, color="black", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add p-value annotations
            for i, (idx, row) in enumerate(sig_tf.iterrows()):
                stars = "***" if row["pvalue"] < 0.001 else "**" if row["pvalue"] < 0.01 else "*"
                xpos = row["log2FoldChange"]
                offset = 0.05 if xpos >= 0 else -0.05
                ha = "left" if xpos >= 0 else "right"
                ax.text(xpos + offset, i, stars, va="center", ha=ha, fontsize=8, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig)
            store_fig(figures, "transcription_factors", fig)

            st.dataframe(sig_tf[["log2FoldChange", "pvalue", "padj"]].round(4), use_container_width=True)

# ============================================================
# 10. PPI NETWORK (STRING) - slide-ready figure
# ============================================================
if run_ppi and "de_results" in results:
    progress.progress(88, text="Building PPI network...")
    st.header("10. Protein-Protein Interaction Network")

    import requests
    import matplotlib.pyplot as plt

    de = results["de_results"]
    sig_genes = de[de["status"] != "Not significant"].index.tolist()[:150]

    string_score = st.slider("STRING confidence cutoff", 0.4, 0.9, 0.7, 0.05, key="ppi_score",
                             help="Minimum STRING interaction confidence to keep an edge. Higher = stricter.")

    if len(sig_genes) > 5:
        with st.spinner("Querying STRING API..."):
            try:
                int_df_full = fetch_string(tuple(sig_genes))
                if int_df_full is not None:
                    if not int_df_full.empty:
                        int_df = int_df_full[int_df_full["Confidence"] >= string_score].sort_values("Confidence", ascending=False)
                        results["ppi"] = int_df

                        st.write(f"**{len(int_df)}** interactions (confidence ≥ {string_score})")

                        # Build networkx graph
                        try:
                            import networkx as nx
                        except ImportError:
                            import subprocess, sys
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx", "-q"])
                            import networkx as nx

                        G = nx.Graph()
                        for _, row in int_df.iterrows():
                            G.add_edge(row["Protein_A"], row["Protein_B"], weight=row["Confidence"])

                        # Keep only the largest connected component for clean visualization
                        if len(G.nodes) > 0:
                            components = list(nx.connected_components(G))
                            components.sort(key=len, reverse=True)

                            # Keep components with >= 3 nodes, or the largest if none qualify
                            kept_components = [c for c in components if len(c) >= 3]
                            if not kept_components:
                                kept_components = [components[0]] if components else []
                            kept_nodes = set()
                            for c in kept_components:
                                kept_nodes.update(c)
                            G = G.subgraph(kept_nodes).copy()

                        if len(G.nodes) > 0:
                            # Degree analysis - hubs get bigger
                            degrees = dict(G.degree())
                            max_deg = max(degrees.values()) if degrees else 1

                            # Identify top hubs (top 10 or top 20% of nodes)
                            n_hubs = max(5, int(len(G.nodes) * 0.15))
                            top_hubs = sorted(degrees.items(), key=lambda x: -x[1])[:n_hubs]
                            hub_names = set(h[0] for h in top_hubs)

                            # Color + size nodes
                            node_colors = []
                            node_sizes = []
                            node_edge_colors = []
                            node_edge_widths = []
                            for node in G.nodes:
                                if node in de.index:
                                    if de.loc[node, "status"] == "Upregulated":
                                        node_colors.append("#e74c3c")
                                    elif de.loc[node, "status"] == "Downregulated":
                                        node_colors.append("#3498db")
                                    else:
                                        node_colors.append("#bdc3c7")
                                else:
                                    node_colors.append("#bdc3c7")
                                # Size scaled by degree (hubs much bigger)
                                deg = degrees[node]
                                size = 400 + (deg / max_deg) * 2200
                                node_sizes.append(size)
                                # Hubs get a bold dark border
                                if node in hub_names:
                                    node_edge_colors.append("#2c3e50")
                                    node_edge_widths.append(2.5)
                                else:
                                    node_edge_colors.append("white")
                                    node_edge_widths.append(1.0)

                            # Edge width by confidence
                            edge_widths = [G[u][v]["weight"] * 2.5 for u, v in G.edges]
                            edge_alphas = [min(0.7, G[u][v]["weight"]) for u, v in G.edges]

                            # Layout - use better algorithm for cleaner positioning
                            fig, ax = plt.subplots(figsize=(14, 11), dpi=150)
                            fig.patch.set_facecolor("white")
                            ax.set_facecolor("white")

                            # Use kamada_kawai for connected components, spring as fallback
                            try:
                                pos = nx.kamada_kawai_layout(G)
                            except Exception:
                                pos = nx.spring_layout(G, k=3.5, iterations=100, seed=42)

                            # Draw edges first
                            nx.draw_networkx_edges(
                                G, pos, ax=ax,
                                width=edge_widths,
                                edge_color="#7f8c8d",
                                alpha=0.4,
                            )

                            # Draw nodes
                            nx.draw_networkx_nodes(
                                G, pos, ax=ax,
                                node_color=node_colors,
                                node_size=node_sizes,
                                edgecolors=node_edge_colors,
                                linewidths=node_edge_widths,
                                alpha=0.92,
                            )

                            # Only label hub genes with large font
                            hub_labels = {n: n for n in G.nodes if n in hub_names}
                            # Other nodes get small labels
                            other_labels = {n: n for n in G.nodes if n not in hub_names}

                            # Draw non-hub labels (smaller)
                            nx.draw_networkx_labels(
                                G, pos, labels=other_labels, ax=ax,
                                font_size=7, font_color="#2c3e50", font_weight="normal"
                            )
                            # Draw hub labels (larger, bold, white background)
                            for node, (x, y) in pos.items():
                                if node in hub_names:
                                    ax.text(
                                        x, y, node,
                                        fontsize=11, fontweight="bold",
                                        ha="center", va="center",
                                        color="white",
                                        bbox=dict(boxstyle="round,pad=0.3", fc="#2c3e50", ec="none", alpha=0.85),
                                        zorder=10,
                                    )

                            # Stats boxes
                            n_up_nodes = sum(1 for n in G.nodes if n in de.index and de.loc[n, "status"] == "Upregulated")
                            n_down_nodes = sum(1 for n in G.nodes if n in de.index and de.loc[n, "status"] == "Downregulated")

                            # Clean legend
                            from matplotlib.lines import Line2D
                            legend_elements = [
                                Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                                       markersize=14, markeredgecolor='white', markeredgewidth=1,
                                       label=f'Upregulated (n={n_up_nodes})'),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
                                       markersize=14, markeredgecolor='white', markeredgewidth=1,
                                       label=f'Downregulated (n={n_down_nodes})'),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor='#bdc3c7',
                                       markersize=14, markeredgecolor='#2c3e50', markeredgewidth=2,
                                       label='Hub gene (top connections)'),
                            ]
                            ax.legend(
                                handles=legend_elements,
                                loc="upper left",
                                fontsize=11,
                                framealpha=0.95,
                                edgecolor="#bdc3c7",
                                fancybox=True,
                                borderpad=0.8,
                                labelspacing=0.8,
                            )

                            # Title with stats
                            ax.set_title(
                                f"Protein-Protein Interaction Network\n"
                                f"{len(G.nodes)} proteins, {len(G.edges)} interactions "
                                f"(STRING confidence ≥ {string_score})",
                                fontsize=14, fontweight="bold", pad=20
                            )
                            ax.axis("off")

                            # Add padding around plot
                            x_vals = [p[0] for p in pos.values()]
                            y_vals = [p[1] for p in pos.values()]
                            x_margin = (max(x_vals) - min(x_vals)) * 0.15
                            y_margin = (max(y_vals) - min(y_vals)) * 0.15
                            ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
                            ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

                            plt.tight_layout()
                            st.pyplot(fig)
                            store_fig(figures, "ppi_network", fig)

                            # Hub analysis table
                            st.subheader("Top Hub Genes")
                            hub_df = pd.DataFrame([
                                {
                                    "Gene": h[0],
                                    "Connections": h[1],
                                    "log2FC": de.loc[h[0], "log2FoldChange"] if h[0] in de.index else None,
                                    "Direction": de.loc[h[0], "status"] if h[0] in de.index else "Unknown",
                                }
                                for h in top_hubs
                            ])
                            st.dataframe(hub_df, use_container_width=True)

                        else:
                            st.warning("No connected components found in PPI network.")

            except Exception as e:
                st.warning(f"STRING query failed: {e}")

# ============================================================
# EXPORT
# ============================================================
progress.progress(92, text="Assembling methods & citations...")
st.header("Methods & How to Cite")

active_keys = {
    "de": run_de,
    "enrichment": run_enrichment,
    "gsea": run_gsea,
    "reactome": run_reactome,
    "ppi": run_ppi,
}
active_keys = {k for k, on in active_keys.items() if on}
methods_text = build_methods_text(active_keys, de_method, correction, author_name, app_name, app_url)

st.markdown(
    "Copy the paragraph below into your **Methods** section. It cites only the tools "
    "you actually ran. The full reference list is included in your results ZIP as `methods.txt`."
)
st.text_area("Auto-generated Methods paragraph", methods_text, height=320)
st.download_button(
    "Download methods.txt",
    data=methods_text,
    file_name="methods.txt",
    mime="text/plain",
)

st.header("Download Results")

# Create zip file
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("methods.txt", methods_text)
    # CSVs
    for name, df in results.items():
        if isinstance(df, pd.DataFrame):
            csv_data = df.to_csv()
            zf.writestr(f"tables/{name}.csv", csv_data)

    # Figures (already rendered to bytes during the run)
    for name, (png_bytes, pdf_bytes) in figures.items():
        zf.writestr(f"figures/{name}.png", png_bytes)
        zf.writestr(f"figures/{name}.pdf", pdf_bytes)

zip_buffer.seek(0)

st.download_button(
    label="Download All Results (ZIP)",
    data=zip_buffer,
    file_name="rnaseq_analysis_results.zip",
    mime="application/zip",
    type="primary",
    use_container_width=True,
)

progress.progress(100, text="Analysis complete!")
st.success("Analysis complete! Download your results above.")
st.balloons()
