"""Reproduce the result tables from the stored run records.

Every run under ../results stores two evaluation traces logged at the same
epochs: ``val_eval_results`` (validation split) and ``eval_results`` (test
split). Models are selected on validation and reported on test: for each run we
take the epoch with the highest validation Recall@20 and read the test metrics
at that epoch. The training harness already writes this value to the
``test_at_val_best`` field of each record; where an older record lacks it we
recompute it from the two traces. These are the numbers that appear in the
report.

Running the script produces:

  results/summary.csv        one row per run, the selected test metrics for
                             recall / ndcg / arhr at cutoffs 3, 5, 10, 20, 50
  analysis/tables/rq1_kK.csv the headline accuracy comparison at K in {3,5,10},
  analysis/tables/rq1_kK.tex LightGCN vs the two LightCCN variants

The mapping from each thesis table to the records that feed it is in
../results/PROVENANCE.md.

Significance markers in the thesis come from paired per-user tests, which need
the per-user ranking files (``*.ranks.npz``). Those are large and are not
distributed here; they are available on request. Without them this script
reports point estimates, which is what the tables below contain.

    python analysis/reproduce_tables.py
"""
import csv
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "analysis", "tables")

FAMILIES = ("recall", "ndcg", "arhr")
CUTOFFS = (3, 5, 10, 20, 50)
METRICS = [f"{fam}@{k}" for k in CUTOFFS for fam in FAMILIES]

# Operating similarity threshold per dataset for the main LightCCN results,
# selected on validation Recall@20 in the threshold sweep (RQ4). L3 is the
# three-layer backbone shared with LightGCN. These are the settings behind the
# headline accuracy table.
OPERATING_TAU = {"ciaodvd": 5, "epinions": 3, "beidian": 4,
                 "foursquare-nyc": 2, "foursquare-tky": 55, "gowalla": 60}
RQ1_ORDER = ["ciaodvd", "epinions", "beidian",
             "foursquare-nyc", "foursquare-tky", "gowalla"]
PRETTY = {"ciaodvd": "CiaoDVD", "epinions": "Epinions", "beidian": "Beidian",
          "foursquare-nyc": "Foursquare-NYC", "foursquare-tky": "Foursquare-TKY",
          "gowalla": "Gowalla"}


def method_of(label, model_name):
    """Decode the method from a run label. LightCCN operator labels encode the
    active channels: 'sef.sm.B' is the full model (propagation plus the user
    readout), 'sef.sm' is propagation only. Everything else is a named
    baseline carried in model_name."""
    if "sef.sm.B" in label:
        return "LightCCN-full"
    if re.search(r"sef\.sm(?!\.)", label):
        return "LightCCN-prop"
    if label.startswith("lightgcn") or model_name == "lightgcn" or "__lightgcn__" in label:
        return "LightGCN"
    return model_name or (label.split("__")[0] if label else "")


def selected_metrics(d):
    """The reported test metrics for a run, and the rule that produced them.

    Runs with a validation split are selected on it: the epoch with the highest
    validation Recall@20, read on the test split. The harness stores this in
    ``test_at_val_best``; older records are recomputed from the two traces.
    Runs evaluated without a validation carve report their own ``final_metrics``
    instead. Returns (metrics, rule) or None."""
    stored = d.get("test_at_val_best")
    if isinstance(stored, dict) and "recall@20" in stored:
        return stored, "val-best-recall@20"
    val, test = d.get("val_eval_results"), d.get("eval_results")
    if val and test:
        epoch = max(val, key=lambda r: r["recall@20"])["epoch"]
        row = next((r for r in test if r["epoch"] == epoch), None)
        if row:
            return row, "val-best-recall@20"
    fm = d.get("final_metrics")
    if isinstance(fm, dict) and "recall@20" in fm:
        return fm, "test-frame"
    return None


def load_runs():
    runs = []
    for path in glob.glob(os.path.join(RESULTS, "**", "*.json"), recursive=True):
        base = os.path.basename(path)
        if base == "queue.json" or base.endswith("taucurve.json"):
            continue
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if not d.get("dataset"):
            continue
        sel = selected_metrics(d)
        if not sel:
            continue
        metrics, rule = sel
        label = d.get("label", "") or ""
        tau = re.search(r"__t(\d+)__", label)
        layers = re.search(r"__L(\d+)", label)
        runs.append({
            "dataset": d["dataset"],
            "method": method_of(label, d.get("model_name")),
            "tau": int(tau.group(1)) if tau else "",
            "layers": int(layers.group(1)) if layers else (d.get("n_layers") or ""),
            "selection": rule,
            "best_epoch": metrics.get("epoch", d.get("best_epoch")),
            "label": label,
            "path": os.path.relpath(path, ROOT),
            "metrics": metrics,
        })
    return runs


def write_summary(runs):
    def _n(x):
        return x if isinstance(x, int) else -1
    runs = sorted(runs, key=lambda r: (r["dataset"], r["method"],
                                       _n(r["tau"]), _n(r["layers"])))
    cols = ["dataset", "method", "tau", "layers", "selection",
            "best_epoch", "label"] + METRICS
    out = os.path.join(RESULTS, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            w.writerow([r["dataset"], r["method"], r["tau"], r["layers"],
                        r["selection"], r["best_epoch"], r["label"]]
                       + [r["metrics"].get(m) for m in METRICS])
    print(f"summary.csv: {len(runs)} runs -> {os.path.relpath(out, ROOT)}")


def rq1_pick(runs, dataset, method):
    """The main-results run for a (dataset, method): three layers, at the
    dataset's operating threshold for the LightCCN variants."""
    tau = OPERATING_TAU[dataset]
    for r in runs:
        if r["dataset"] != dataset or r["layers"] != 3:
            continue
        if r["method"] != method or r["selection"] != "val-best-recall@20":
            continue
        if method.startswith("LightCCN") and r["tau"] != tau:
            continue
        return r
    return None


def rq1_tables(runs):
    os.makedirs(TABLES, exist_ok=True)
    for K in (3, 5, 10):
        rows = []
        for ds in RQ1_ORDER:
            base = rq1_pick(runs, ds, "LightGCN")
            full = rq1_pick(runs, ds, "LightCCN-full")
            prop = rq1_pick(runs, ds, "LightCCN-prop")
            if not (base and full):
                continue
            cell = {"dataset": ds}
            for tag, r in (("lightgcn", base), ("full", full), ("prop", prop)):
                if not r:
                    continue
                for fam in FAMILIES:
                    cell[f"{tag}_{fam}"] = r["metrics"].get(f"{fam}@{K}")
            for fam in FAMILIES:
                lg = cell.get(f"lightgcn_{fam}")
                cc = cell.get(f"full_{fam}")
                cell[f"delta_{fam}"] = 100 * (cc - lg) / lg if lg else None
            rows.append(cell)
        _write_rq1_csv(rows, K)
        _write_rq1_tex(rows, K)
        print(f"rq1_k{K}: {len(rows)} datasets -> "
              f"{os.path.relpath(os.path.join(TABLES, f'rq1_k{K}.tex'), ROOT)}")
    return rows  # last K (=10); only used by callers that ignore it


def _write_rq1_csv(rows, K):
    cols = ["dataset"]
    for tag in ("lightgcn", "full", "prop"):
        cols += [f"{tag}_{fam}" for fam in FAMILIES]
    cols += [f"delta_{fam}_pct" for fam in FAMILIES]
    with open(os.path.join(TABLES, f"rq1_k{K}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            out = [r["dataset"]]
            for tag in ("lightgcn", "full", "prop"):
                out += [r.get(f"{tag}_{fam}") for fam in FAMILIES]
            out += [None if r.get(f"delta_{fam}") is None
                    else round(r[f"delta_{fam}"], 1) for fam in FAMILIES]
            w.writerow(out)


def _write_rq1_tex(rows, K):
    up = {"recall": f"Recall@{K}", "ndcg": f"NDCG@{K}", "arhr": f"ARHR@{K}"}
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Model & " +
        " & ".join(up[f] for f in ("ndcg", "recall", "arhr")) + r" \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        def fmt(tag):
            return " & ".join(
                f"{r.get(f'{tag}_{fam}'):.4f}" if r.get(f"{tag}_{fam}") is not None
                else "--" for fam in ("ndcg", "recall", "arhr"))

        def dfmt():
            return " & ".join(
                f"\\textbf{{{r[f'delta_{fam}']:+.1f}\\%}}"
                if r.get(f"delta_{fam}") is not None else "--"
                for fam in ("ndcg", "recall", "arhr"))
        lines.append(f"\\textbf{{{PRETTY[r['dataset']]}}} & LightGCN & {fmt('lightgcn')} \\\\")
        lines.append(f" & LightCCN-full & {fmt('full')} \\\\")
        lines.append(f" & $\\Delta$\\% & {dfmt()} \\\\")
        if i < len(rows) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLES, f"rq1_k{K}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


# Values transcribed from the thesis RQ1 table, used to prove the pipeline
# reproduces the reported numbers. (dataset, model): (ndcg@5, recall@5, arhr@5).
ANCHORS = {
    ("ciaodvd", "LightGCN"): (0.0245, 0.0370, 0.0179),
    ("ciaodvd", "LightCCN-full"): (0.0284, 0.0402, 0.0216),
    ("epinions", "LightCCN-full"): (0.0223, 0.0236, 0.0125),
    ("beidian", "LightCCN-full"): (0.0390, 0.0607, 0.0319),
    ("foursquare-nyc", "LightCCN-full"): (0.0771, 0.0224, 0.0125),
    ("foursquare-tky", "LightCCN-full"): (0.1982, 0.0525, 0.0305),
}


def validate(runs):
    ok = True
    for (ds, method), (n5, r5, a5) in ANCHORS.items():
        r = rq1_pick(runs, ds, method)
        got = None if not r else (r["metrics"]["ndcg@5"],
                                  r["metrics"]["recall@5"], r["metrics"]["arhr@5"])
        good = got is not None and all(abs(g - e) < 5e-4
                                       for g, e in zip(got, (n5, r5, a5)))
        ok = ok and good
        mark = "ok " if good else "MISMATCH"
        shown = "missing" if got is None else \
            f"ndcg={got[0]:.4f} recall={got[1]:.4f} arhr={got[2]:.4f}"
        print(f"  [{mark}] {ds:16} {method:14} {shown}")
    return ok


def main():
    runs = load_runs()
    write_summary(runs)
    rq1_tables(runs)
    print("validation against thesis RQ1 anchors:")
    if validate(runs):
        print("all anchors reproduced.")
    else:
        raise SystemExit("one or more anchors did not reproduce; see above.")


if __name__ == "__main__":
    main()
