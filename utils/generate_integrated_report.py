"""Generate a detailed integrated development report (HTML + LaTeX/PDF).

Usage from project root:
    python utils/generate_integrated_report.py --project-root .

The script reads outputs from notebooks 05/06/07 and, when available, 07b.
It is intentionally narrative: every major figure/table is followed by an
interpretation and a concrete answer to "what does this mean operationally?".
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from jinja2 import Template
except ImportError as exc:
    raise ImportError("Install jinja2: python -m pip install jinja2") from exc


def read_csv_if(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def latex_escape(s: object) -> str:
    text = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text


def fmt(x, digits=1, suffix=""):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}{suffix}"
    except Exception:
        return str(x)


def image_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:image/png;base64," + data


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def dataframe_html(df: pd.DataFrame, digits=2, max_rows=40) -> str:
    z = df.head(max_rows).copy()
    for c in z.select_dtypes(include=[np.number]).columns:
        z[c] = z[c].round(digits)
    return z.to_html(index=False, border=0, classes="data-table")


def dataframe_latex_rows(df: pd.DataFrame, columns: List[str], formatters: Optional[Dict[str, callable]] = None, max_rows=30) -> str:
    rows = []
    formatters = formatters or {}
    for _, r in df.head(max_rows).iterrows():
        vals = []
        for c in columns:
            v = r[c]
            if c in formatters:
                vals.append(latex_escape(formatters[c](v)))
            elif isinstance(v, (float, np.floating)):
                vals.append("-" if pd.isna(v) else f"{v:.2f}")
            else:
                vals.append(latex_escape(v))
        rows.append(" & ".join(vals) + r" \\")
    return "\n".join(rows)


def load_inputs(root: Path):
    outputs = root / "outputs"
    tables = outputs / "tables"
    predictions = outputs / "predictions"

    point = read_csv_if(predictions / "point_models_2024_120min_v5.csv")
    if point is None:
        raise FileNotFoundError(
            "Nao encontrei outputs/predictions/point_models_2024_120min_v5.csv. "
            "Rode os notebooks 06 e 07 v5 antes do relatorio integrado."
        )
    if "prediction_time" in point.columns:
        point["prediction_time"] = pd.to_datetime(point["prediction_time"])

    data = {
        "point_predictions": point,
        "point_metrics": read_csv_if(tables / "operational_point_models_2024.csv"),
        "point_severity": read_csv_if(tables / "operational_point_models_by_severity_2024.csv"),
        "alert_heads": read_csv_if(tables / "operational_alert_heads_2024.csv"),
        "episode_summary": read_csv_if(tables / "radar_episode_summary_2024.csv"),
        "episode_detection": read_csv_if(tables / "radar_episode_detection_2024.csv"),
        "radar_regression": read_csv_if(tables / "radar_regression_comparison_2024.csv"),
        "radar_classifier": read_csv_if(tables / "radar_classifier_comparison_2024.csv"),
        "uncertainty": read_csv_if(tables / "uncertainty_calibration_2024.csv"),
        "neural_point": read_csv_if(tables / "neural_point_metrics_2024_v6.csv"),
        "neural_severity": read_csv_if(tables / "neural_severity_metrics_2024_v6.csv"),
        "neural_alert": read_csv_if(tables / "neural_alert_metrics_2024_v6.csv"),
        "neural_episode": read_csv_if(tables / "neural_episode_metrics_2024_v6.csv"),
        "neural_conformal": read_csv_if(tables / "neural_conformal_coverage_2024_v6.csv"),
    }
    return outputs, data


def compute_extreme_diagnostics(point: pd.DataFrame):
    target = "target_delta_120m"
    model_cols = {
        "XGBoost normal": "pred_xgboost_normal",
        "XGBoost + radar": "pred_xgboost_plus_radar",
        "XGBoost ponderado": "pred_xgboost_ponderado",
        "LightGBM": "pred_lightgbm",
        "Duas etapas": "pred_duas_etapas",
    }
    model_cols = {k: v for k, v in model_cols.items() if v in point.columns}

    z = point[point[target] > 3].copy()
    dist = {
        "n": len(z),
        "actual_mean": z[target].mean(),
        "actual_median": z[target].median(),
        "actual_min": z[target].min(),
        "actual_p75": z[target].quantile(.75),
        "actual_p90": z[target].quantile(.90),
        "actual_max": z[target].max(),
    }
    bins = [3, 3.5, 4, 5, 6, np.inf]
    labels = ["3-3.5", "3.5-4", "4-5", "5-6", ">6"]
    z["actual_band_gt3"] = pd.cut(
        z[target], bins=bins, labels=labels, right=True)

    rows = []
    for name, c in model_cols.items():
        e = z[c]-z[target]
        ae = e.abs()
        rows.append({
            "model": name, "n": len(z),
            "real_mean_m": z[target].mean(), "pred_mean_m": z[c].mean(),
            "MAE_m": ae.mean(), "median_AE_m": ae.median(), "RMSE_m": np.sqrt(np.mean(e**2)),
            "p90_AE_m": ae.quantile(.90), "max_AE_m": ae.max(),
            "bias_m": e.mean(), "underprediction_pct": 100*np.mean(e < 0),
        })
    model_diag = pd.DataFrame(rows).sort_values("MAE_m")

    # Error decomposition for the best continuous model, XGBoost + radar when present.
    best_col = model_cols.get(
        "XGBoost + radar", next(iter(model_cols.values())))
    zz = z.copy()
    zz["AE"] = (zz[best_col]-zz[target]).abs()
    zz["SE"] = (zz[best_col]-zz[target])**2
    decomp = (zz.groupby("actual_band_gt3", observed=True)
              .agg(n=(target, "size"), real_mean_m=(target, "mean"), pred_mean_m=(best_col, "mean"), MAE_m=("AE", "mean"), sum_AE=("AE", "sum"), MSE=("SE", "mean"), sum_SE=("SE", "sum"))
              .reset_index())
    decomp["RMSE_m"] = np.sqrt(decomp["MSE"])
    decomp["share_abs_error_pct"] = 100*decomp["sum_AE"]/decomp["sum_AE"].sum()
    decomp["share_squared_error_pct"] = 100 * \
        decomp["sum_SE"]/decomp["sum_SE"].sum()

    # Concrete examples: severe misses, near-3m good/bad, largest event.
    z["xgb_radar_error_m"] = z[best_col]-z[target]
    examples = []
    # Largest event
    examples.append(z.loc[z[target].idxmax()])
    # Worst abs error in 3-3.5m (demonstrates not only 6-7m events)
    moderate = z[(z[target] > 3) & (z[target] <= 3.5)].copy()
    if len(moderate):
        examples.append(
            moderate.loc[moderate["xgb_radar_error_m"].abs().idxmax()])
        examples.append(
            moderate.loc[moderate["xgb_radar_error_m"].abs().idxmin()])
    # 4-5 and 5-6 examples
    for lo, hi in [(4, 5), (5, 6)]:
        q = z[(z[target] > lo) & (z[target] <= hi)]
        if len(q):
            examples.append(q.loc[q["xgb_radar_error_m"].abs().idxmax()])
    ex = pd.DataFrame(examples).drop_duplicates(
        subset=["prediction_time"]).copy()
    keep = ["prediction_time", target]+list(model_cols.values())
    ex = ex[[c for c in keep if c in ex.columns]
            ].sort_values("prediction_time")

    # Summary for the 3-3.5m subset specifically.
    moderate = z[(z[target] > 3) & (z[target] <= 3.5)]
    moderate_summary = {
        "n": len(moderate),
        "real_mean_m": moderate[target].mean(),
        "pred_mean_m": moderate[best_col].mean(),
        "MAE_m": (moderate[best_col]-moderate[target]).abs().mean(),
        "underprediction_pct": 100*np.mean(moderate[best_col] < moderate[target]),
        "min_pred_m": moderate[best_col].min(),
        "max_pred_m": moderate[best_col].max(),
    }
    return dist, model_diag, decomp, ex, moderate_summary, model_cols


def create_figures(outdir: Path, data: dict, extreme: tuple) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    figs = {}
    point = data["point_predictions"]
    target = "target_delta_120m"
    dist, model_diag, decomp, examples, moderate_summary, model_cols = extreme

    # 1 global RMSE: tabular + neural if available
    rows = []
    pm = data.get("point_metrics")
    if pm is not None:
        for _, r in pm.iterrows():
            rows.append((str(r["model"]), float(r["RMSE_cm"]), "Tabular"))
    nm = data.get("neural_point")
    if nm is not None:
        for _, r in nm.iterrows():
            rows.append((str(r["model"]), float(
                r["RMSE_cm"]), "Rede temporal"))
    if rows:
        rdf = pd.DataFrame(
            rows, columns=["model", "RMSE_cm", "family"]).sort_values("RMSE_cm")
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.barh(rdf["model"], rdf["RMSE_cm"])
        ax.invert_yaxis()
        ax.set_xlabel("RMSE em 2024 (cm)")
        ax.set_title("Erro global da previsao de 2 h")
        for i, v in enumerate(rdf["RMSE_cm"]):
            ax.text(v+0.3, i, f"{v:.1f}", va="center", fontsize=8)
        fig.tight_layout()
        p = outdir/"01_rmse_global.png"
        save_fig(fig, p)
        figs["rmse_global"] = p

    # 2 severity MAE tabular + neural
    sev = data.get("point_severity")
    frames = []
    if sev is not None:
        s = sev[["severity_band", "model", "MAE_cm"]].copy()
        frames.append(s)
    nsev = data.get("neural_severity")
    if nsev is not None:
        s = nsev[["severity_band", "model", "MAE_cm"]].copy()
        frames.append(s)
    if frames:
        allsev = pd.concat(frames, ignore_index=True)
        order = ["<=0", "0-0.25", "0.25-0.5", "0.5-1", "1-2", "2-3", ">3"]
        pv = allsev.pivot_table(index="severity_band", columns="model",
                                values="MAE_cm", aggfunc="first").reindex(order)
        fig, ax = plt.subplots(figsize=(12, 5.7))
        pv.plot(kind="bar", ax=ax)
        ax.set_ylabel("MAE (cm)")
        ax.set_xlabel("Variacao real em 2 h (m)")
        ax.set_title("O erro cresce fortemente com a severidade")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = outdir/"02_mae_por_severidade.png"
        save_fig(fig, p)
        figs["severity"] = p

    # 3 Distribution actual >3m
    z = point[point[target] > 3]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(z[target], bins=np.arange(3, 7.26, .25), edgecolor="white")
    ax.axvline(z[target].median(), linestyle="--",
               label=f"mediana {z[target].median():.2f} m")
    ax.axvline(z[target].mean(), linestyle=":",
               label=f"media {z[target].mean():.2f} m")
    ax.set_xlabel("Subida real nas 2 h seguintes (m)")
    ax.set_ylabel("Numero de timestamps")
    ax.set_title("Distribuicao dos 115 timestamps com subida > 3 m")
    ax.legend()
    fig.tight_layout()
    p = outdir/"03_distribuicao_gt3m.png"
    save_fig(fig, p)
    figs["gt3_dist"] = p

    # 4 XGB+radar scatter >3m
    best_col = model_cols.get(
        "XGBoost + radar", next(iter(model_cols.values())))
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.scatter(z[target], z[best_col], s=22, alpha=.65)
    lo = min(3, z[best_col].min())
    hi = max(z[target].max(), z[best_col].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("Subida real (m)")
    ax.set_ylabel("Subida prevista - XGBoost + radar (m)")
    ax.set_title("Casos reais > 3 m: previsto x real")
    fig.tight_layout()
    p = outdir/"04_scatter_gt3m.png"
    save_fig(fig, p)
    figs["gt3_scatter"] = p

    # 5 contribution to error
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(decomp))
    w = .38
    ax.bar(x-w/2, decomp["share_abs_error_pct"], width=w,
           label="Participacao no erro absoluto")
    ax.bar(x+w/2, decomp["share_squared_error_pct"],
           width=w, label="Participacao no erro quadratico")
    ax.set_xticks(x, decomp["actual_band_gt3"].astype(str))
    ax.set_ylabel("Participacao no erro total dos casos >3 m (%)")
    ax.set_xlabel("Faixa da subida real (m)")
    ax.set_title("Os poucos casos >6 m nao explicam sozinhos o erro alto")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir/"05_decomposicao_erro_gt3m.png"
    save_fig(fig, p)
    figs["decomp"] = p

    # 6 Case studies around two dates
    for key, center, label in [("feb14", pd.Timestamp("2024-02-14 01:20"), "14/02/2024"), ("mar08", pd.Timestamp("2024-03-08 18:00"), "08/03/2024")]:
        q = point[(point["prediction_time"] >= center-pd.Timedelta(hours=4))
                  & (point["prediction_time"] <= center+pd.Timedelta(hours=4))].copy()
        if len(q):
            fig, ax = plt.subplots(figsize=(11, 4.8))
            ax.plot(q["prediction_time"]+pd.Timedelta(hours=2),
                    100*q[target], label="Real", linewidth=2.2)
            for name, c in model_cols.items():
                if name in ["XGBoost + radar", "XGBoost normal", "Duas etapas"]:
                    ax.plot(q["prediction_time"]+pd.Timedelta(hours=2),
                            100*q[c], label=name, alpha=.9)
            ax.axhline(100, linestyle="--", linewidth=1, label="1 m")
            ax.set_ylabel("Variacao nas 2 h seguintes (cm)")
            ax.set_xlabel("Horario alvo")
            ax.set_title(f"Estudo de caso - {label}")
            ax.legend(fontsize=8, ncol=2)
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            p = outdir/f"06_case_{key}.png"
            save_fig(fig, p)
            figs[key] = p

    # 7 alert precision/recall
    ah = data.get("alert_heads")
    if ah is not None:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for _, r in ah.iterrows():
            ax.scatter(r["recall_pct"], r["precision_pct"],
                       s=90, label=str(r["model"]))
        ax.set_xlabel("Recall (%) - quanto dos eventos >=1 m foi detectado")
        ax.set_ylabel("Precision (%) - quanto dos alertas estava correto")
        ax.set_title("Trade-off do classificador de alerta")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir/"07_alert_precision_recall.png"
        save_fig(fig, p)
        figs["alert_pr"] = p

    # 8 episode recall
    ep = data.get("episode_summary")
    if ep is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, g in ep.groupby("method"):
            ax.plot(g["severity_threshold_m"],
                    g["episode_recall_pct"], marker="o", label=method)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Pico real do episodio (limiar em m)")
        ax.set_ylabel("Episodios com algum alerta (%)")
        ax.set_title(
            "Deteccao por episodio - cuidado: classificadores atuais alertam apenas >=1 m")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = outdir/"08_episode_recall.png"
        save_fig(fig, p)
        figs["episode"] = p

    # 9 uncertainty
    unc = data.get("uncertainty")
    if unc is not None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for method, g in unc.groupby("method"):
            ax.plot(g["nominal_coverage_pct"],
                    g["empirical_coverage_2024_pct"], marker="o", label=method)
        ax.plot([75, 100], [75, 100], linestyle="--",
                linewidth=1, label="calibracao perfeita")
        ax.set_xlabel("Cobertura nominal (%)")
        ax.set_ylabel("Cobertura empirica em 2024 (%)")
        ax.set_title("Calibracao global dos intervalos")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir/"09_uncertainty.png"
        save_fig(fig, p)
        figs["uncertainty"] = p

    # 10 neural alert if available
    na = data.get("neural_alert")
    if na is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, g in na.groupby("model"):
            ax.plot(g["event_threshold_m"], g["recall_pct"],
                    marker="o", label=model)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Severidade do alerta (m em 2 h)")
        ax.set_ylabel("Recall (%)")
        ax.set_title("Redes temporais: heads especificas para 1, 2 e 3 m")
        ax.legend()
        fig.tight_layout()
        p = outdir/"10_neural_alerts.png"
        save_fig(fig, p)
        figs["neural_alerts"] = p

    return figs


def compile_latex(tex_path: Path, pdf_path: Path) -> Tuple[bool, str]:
    # Prefer latexmk, then pdflatex. No shell=True to keep paths safe.
    engine = shutil.which("latexmk")
    work = tex_path.parent
    if engine:
        cmd = [engine, "-pdf", "-interaction=nonstopmode",
               "-halt-on-error", tex_path.name]
        p = subprocess.run(cmd, cwd=work, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    else:
        engine = shutil.which("pdflatex")
        if not engine:
            return False, "Nenhum latexmk/pdflatex encontrado. O .tex foi gerado, mas nao compilado."
        cmd = [engine, "-interaction=nonstopmode",
               "-halt-on-error", tex_path.name]
        p = subprocess.run(cmd, cwd=work, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        if p.returncode == 0:
            p = subprocess.run(cmd, cwd=work, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    generated = tex_path.with_suffix(".pdf")
    if p.returncode == 0 and generated.exists():
        if generated.resolve() != pdf_path.resolve():
            shutil.copy2(generated, pdf_path)
        return True, p.stdout[-2000:]
    return False, p.stdout[-5000:]


def build_context(data: dict, extreme: tuple):
    point = data["point_predictions"]
    dist, model_diag, decomp, examples, moderate_summary, model_cols = extreme
    pm = data.get("point_metrics")
    best_point = pm.sort_values("RMSE_cm").iloc[0] if pm is not None else None
    radar = data.get("radar_regression")
    radar_test = radar[radar["period"] ==
                       "test_2024"] if radar is not None else None
    radar_gain = float(radar_test[radar_test["model"] == "com_radar"]["rmse_gain_vs_base_pct"].iloc[0]
                       ) if radar_test is not None and len(radar_test[radar_test["model"] == "com_radar"]) else np.nan
    ah = data.get("alert_heads")
    ep = data.get("episode_summary")
    unc = data.get("uncertainty")
    neural = data.get("neural_point")
    best_neural = neural.sort_values(
        "RMSE_cm").iloc[0] if neural is not None and len(neural) else None

    return {
        "dist": dist, "model_diag": model_diag, "decomp": decomp, "examples": examples, "moderate": moderate_summary,
        "best_point": best_point, "radar_gain": radar_gain, "alert_heads": ah, "episodes": ep, "uncertainty": unc,
        "neural": neural, "best_neural": best_neural, "model_cols": model_cols,
    }


def make_html(report_dir: Path, figs: dict, ctx: dict, data: dict) -> Path:
    dist = ctx["dist"]
    moderate = ctx["moderate"]
    best = ctx["best_point"]
    neural_available = ctx["neural"] is not None and len(ctx["neural"])

    best_text = f"{best['model']} - RMSE {best['RMSE_cm']:.1f} cm" if best is not None else "-"
    neural_text = f"{ctx['best_neural']['model']} - RMSE {ctx['best_neural']['RMSE_cm']:.1f} cm" if neural_available else "Aguardando notebook 07b"

    # Concrete extreme examples table with friendly names.
    ex = ctx["examples"].copy()
    rename = {"target_delta_120m": "Real (m)", "pred_xgboost_normal": "XGB (m)", "pred_xgboost_plus_radar": "XGB+radar (m)", "pred_xgboost_ponderado": "XGB pond. (m)",
              "pred_lightgbm": "LightGBM (m)", "pred_duas_etapas": "Duas etapas (m)", "prediction_time": "Horario da previsao"}
    ex = ex.rename(columns=rename)

    tmpl = Template(r'''<!doctype html>
<html lang="pt-br"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relatorio integrado - previsao hidrologica 2024</title>
<style>
:root{--bg:#f4f6f8;--paper:#fff;--text:#1f2833;--muted:#687383;--line:#dde3e9;--accent:#244f76;--soft:#eef4f8;--warn:#fff5df;--good:#edf7f1}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;line-height:1.55}
nav{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);padding:10px 22px;display:flex;gap:14px;flex-wrap:wrap;font-size:13px}
nav a{color:var(--accent);text-decoration:none}main{max-width:1180px;margin:0 auto;padding:34px 26px 70px}.hero{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:30px;margin-bottom:22px}.hero h1{font-family:Georgia,serif;font-size:34px;margin:0 0 8px}.subtitle{color:var(--muted);margin:0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:20px 0}.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:17px}.card .k{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}.card .v{font-size:25px;font-weight:700;margin-top:5px}.section{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:25px;margin:20px 0}.section h2{font-family:Georgia,serif;font-size:25px;margin-top:0;border-bottom:1px solid var(--line);padding-bottom:9px}.section h3{margin-top:25px}.callout{padding:15px 17px;border-left:5px solid var(--accent);background:var(--soft);border-radius:8px;margin:16px 0}.warning{background:var(--warn);border-left-color:#b7832d}.good{background:var(--good);border-left-color:#3b8058}.figure{margin:18px 0;text-align:center}.figure img{max-width:100%;height:auto;border-radius:8px}.caption{font-size:13px;color:var(--muted);text-align:left;margin-top:8px}.data-table{width:100%;border-collapse:collapse;font-size:13px}.data-table th{background:#edf1f4;text-align:left;padding:8px;position:sticky;top:42px}.data-table td{border-top:1px solid #e6e9ed;padding:7px 8px}.table-wrap{overflow:auto;margin:15px 0}details{border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:12px 0}summary{cursor:pointer;font-weight:600;color:var(--accent)}code{background:#f0f2f4;padding:2px 5px;border-radius:4px}@media(max-width:700px){main{padding:18px 12px}.hero h1{font-size:28px}.section{padding:18px}}
</style></head><body>
<nav><a href="#resumo">Resumo</a><a href="#metricas">Metricas</a><a href="#extremos">Extremos &gt;3 m</a><a href="#alertas">Alertas</a><a href="#casos">Casos reais</a><a href="#incerteza">Incerteza</a><a href="#redes">Redes neurais</a><a href="#conclusao">Conclusao</a></nav>
<main>
<section class="hero" id="resumo"><h1>Previsao do nivel do Tamanduatei - relatorio de desenvolvimento</h1><p class="subtitle">Avaliacao em 2024, horizonte principal de 2 h. 2025 permanece reservado para o teste cego final.</p>
<div class="cards"><div class="card"><div class="k">Melhor modelo tabular</div><div class="v">{{best_text}}</div></div><div class="card"><div class="k">Ganho do radar no RMSE</div><div class="v">{{radar_gain}}%</div></div><div class="card"><div class="k">Casos com subida &gt; 3 m</div><div class="v">{{n_gt3}}</div></div><div class="card"><div class="k">Rede temporal</div><div class="v">{{neural_text}}</div></div></div>
<div class="callout good"><b>Conclusao atual:</b> o XGBoost + radar e o melhor regressor tabular para a variacao de 2 h, mas a magnitude dos extremos continua sendo o principal gargalo. O classificador de risco deve ser tratado como uma tarefa separada da regressao.</div></section>

<section class="section" id="metricas"><h2>1. Como ler MAE e RMSE sem ficar preso a siglas</h2>
<p>Para cada horario, existe um erro <code>previsto - real</code>. O <b>MAE</b> faz a media dos valores absolutos desses erros. O <b>RMSE</b> eleva os erros ao quadrado antes de tirar a media e, por isso, pune muito mais uma falha grande.</p>
<div class="callout"><b>Exemplo:</b> se quatro erros forem 5, 5, 5 e 50 cm, o MAE e 16,25 cm, enquanto o RMSE e 25,4 cm. O RMSE sobe mais porque o erro de 50 cm recebe peso quadratico. Ele nao quer dizer que "o modelo erra exatamente isso em cada previsao".</div>
<div class="figure"><img src="{{fig_rmse}}"><div class="caption">Quanto menor, melhor. Este grafico resume milhares de previsoes; ele nao substitui a analise dos eventos graves.</div></div></section>

<section class="section" id="extremos"><h2>2. O que realmente significa "MAE de cerca de 2,3 m quando a subida passa de 3 m"?</h2>
<p>Esta e uma das perguntas mais importantes do projeto. Em 2024 houve <b>{{n_gt3}}</b> timestamps em que a subida real nas duas horas seguintes foi maior que 3 m. A subida real media nesses timestamps foi <b>{{actual_mean}} m</b>, a mediana foi <b>{{actual_median}} m</b> e o maximo foi <b>{{actual_max}} m</b>.</p>
<div class="callout warning"><b>Nao e apenas um efeito dos poucos eventos de 6-7 m.</b> Mais da metade dos timestamps &gt;3 m ({{moderate_n}} de {{n_gt3}}) esta entre 3,0 e 3,5 m. Nessa faixa relativamente proxima de 3 m, o XGBoost + radar previu em media {{moderate_pred}} m para uma subida real media de {{moderate_real}} m, com MAE de {{moderate_mae}} m.</div>
<div class="figure"><img src="{{fig_dist}}"><div class="caption">A maioria dos casos &gt;3 m nao esta perto de 7 m; ha uma concentracao forte logo acima de 3 m.</div></div>
<div class="figure"><img src="{{fig_scatter}}"><div class="caption">Pontos abaixo da diagonal representam subestimacao. A concentracao abaixo da linha mostra por que o bias dos extremos e fortemente negativo.</div></div>
<div class="figure"><img src="{{fig_decomp}}"><div class="caption">Esta decomposicao responde diretamente se os poucos maiores eventos "inflam" a media. No XGBoost + radar, os casos de 3-3,5 m respondem por uma fracao maior do erro absoluto total do que os poucos casos &gt;6 m.</div></div>
<h3>Numeros por modelo, considerando somente subidas reais &gt;3 m</h3><div class="table-wrap">{{extreme_table}}</div>
<h3>Exemplos concretos</h3><div class="table-wrap">{{examples_table}}</div>
<p>Esses exemplos mostram que existem os dois comportamentos: alguns casos perto de 3-3,5 m sao previstos razoavelmente, enquanto outros sao quase completamente perdidos. Portanto o MAE alto vem de uma falha sistematica em muitos extremos, somada - mas nao causada exclusivamente - pelos eventos de 5-7 m.</p>
<div class="figure"><img src="{{fig_severity}}"><div class="caption">A curva de dificuldade e monotona: quanto mais severa a subida, maior o erro. O problema e estrutural, nao apenas um outlier isolado.</div></div></section>

<section class="section" id="alertas"><h2>3. Prever magnitude e emitir alerta sao tarefas diferentes</h2>
<p>Um regressor tenta dizer <i>quanto</i> o nivel vai variar. Um classificador tenta responder se uma condicao perigosa sera atingida. Um sistema operacional pode usar os dois em paralelo.</p>
<div class="figure"><img src="{{fig_pr}}"><div class="caption">Recall alto significa perder menos eventos reais; precision alta significa menos falsos alertas.</div></div>
<div class="table-wrap">{{alert_table}}</div>
<div class="figure"><img src="{{fig_episode}}"><div class="caption"><b>Cuidado com a leitura:</b> o classificador atual foi treinado para >=1 m. Quando ele "detecta" um episodio cujo pico real passou de 3 m, isso significa que emitiu pelo menos um alerta de >=1 m durante o episodio - nao que previu corretamente 3 m.</div></div>
<p>Essa limitacao motivou as novas redes temporais multi-head: elas terao saidas separadas para >=1 m, >=2 m e >=3 m.</p></section>

<section class="section" id="casos"><h2>4. Dois eventos que tornam as metricas palpaveis</h2>
<h3>14/02/2024</h3><p>O maior pico de variacao em 2 h chegou a aproximadamente 7,05 m. Os modelos reconheceram que havia uma subida enorme, mas ainda subestimaram fortemente sua magnitude.</p><div class="figure"><img src="{{fig_feb}}"></div>
<h3>08/03/2024</h3><p>Este episodio e ainda mais interessante para alerta precoce: houve momentos em que a subida real futura ja era de 4-5 m e o regressor ainda previa proximo de zero. O radar ajudou, mas nao resolveu a defasagem. E exatamente o tipo de padrao temporal que TCN/GRU podem tentar aprender.</p><div class="figure"><img src="{{fig_mar}}"></div></section>

<section class="section" id="incerteza"><h2>5. Incerteza: um intervalo pode ser mais honesto que um unico numero</h2>
<p>O split conformal calibra uma margem a partir dos erros de 2023 e pergunta com que frequencia o valor de 2024 cai dentro do intervalo. Globalmente, o metodo ficou um pouco conservador.</p><div class="figure"><img src="{{fig_unc}}"></div><div class="table-wrap">{{unc_table}}</div>
<div class="callout warning">Cobertura global boa nao garante cobertura boa durante enchentes. O novo pipeline tambem calcula cobertura condicional para subidas >=1, >=2 e >=3 m.</div></section>

<section class="section" id="redes"><h2>6. Redes neurais temporais</h2>
{% if neural_available %}<p>O notebook 07b ja foi executado. A tabela abaixo compara as redes temporais.</p><div class="table-wrap">{{neural_table}}</div>{% if fig_neural %}<div class="figure"><img src="{{fig_neural}}"></div>{% endif %}
{% else %}<div class="callout"><b>Aguardando execucao do notebook 07b.</b> Ele treina TCN e GRU usando uma sequencia das ultimas 6 h e quatro saidas: regressao de Delta H e probabilidades de >=1, >=2 e >=3 m. Depois de roda-lo, execute novamente o 07c e esta secao sera preenchida automaticamente.</div>{% endif %}
<h3>Por que esta etapa faz sentido?</h3><p>Os modelos tabulares recebem resumos construidos por nos. TCN e GRU recebem a ordem temporal dos dados e podem aprender padroes como chuva crescente -> subida a montante -> aceleracao na 413. A literatura de previsao hidrologica usa amplamente GRU/LSTM para dependencias temporais, e ha estudos em que TCN supera LSTM em previsao de cheias; isso justifica o teste, mas nao garante que a rede sera melhor nesta bacia.</p></section>

<section class="section" id="conclusao"><h2>7. Conclusao e decisao antes de 2025</h2>
<ol><li><b>Regressao tabular:</b> XGBoost + radar e a referencia atual.</li><li><b>Alerta:</b> manter tarefa dedicada; o ganho do radar no classificador e misto e depende do custo de falsos alertas.</li><li><b>Extremos:</b> a subestimacao acima de 3 m nao e explicada apenas pelos poucos casos de 6-7 m; ela ja aparece forte entre 3 e 3,5 m.</li><li><b>Proxima comparacao:</b> TCN e GRU multi-head contra XGBoost + radar, usando 2024.</li><li><b>2025:</b> continua cego. O notebook 08 so deve ser executado depois de congelar a arquitetura.</li></ol>
<h3>Limite sobre "evitar perdas"</h3><p>Sem dados de danos, evacuacoes, vitimas e decisoes reais, nao e cientificamente defensavel estimar quantos reais ou quantas perdas humanas teriam sido evitados. Podemos medir proxies operacionais: eventos antecipados, eventos perdidos, falsos alertas, antecedencia e erro de magnitude.</p></section>

<section class="section"><h2>Referencias que motivam a etapa temporal</h2><ul><li>Application of temporal convolutional network for flood forecasting, Hydrology Research (2021), DOI: 10.2166/nh.2021.021.</li><li>Water Level Prediction Model Applying an LSTM-GRU Method for Flood Prediction, Water (2022), DOI: 10.3390/w14142221.</li><li>A Comprehensive Review of Methods for Hydrological Forecasting Based on Deep Learning, Water (2024), DOI: 10.3390/w16101407.</li><li>Review and Intercomparison of Machine Learning Applications for Short-term Flood Forecasting, Water Resources Management (2025), DOI: 10.1007/s11269-025-04093-x.</li></ul></section>
</main></body></html>''')

    html = tmpl.render(
        best_text=best_text, radar_gain=fmt(ctx["radar_gain"], 1), n_gt3=dist["n"], neural_text=neural_text,
        actual_mean=fmt(dist["actual_mean"], 2), actual_median=fmt(dist["actual_median"], 2), actual_max=fmt(dist["actual_max"], 2),
        moderate_n=moderate["n"], moderate_pred=fmt(moderate["pred_mean_m"], 2), moderate_real=fmt(moderate["real_mean_m"], 2), moderate_mae=fmt(moderate["MAE_m"], 2),
        fig_rmse=image_uri(figs["rmse_global"]), fig_dist=image_uri(figs["gt3_dist"]), fig_scatter=image_uri(figs["gt3_scatter"]), fig_decomp=image_uri(figs["decomp"]), fig_severity=image_uri(figs["severity"]),
        fig_pr=image_uri(figs["alert_pr"]), fig_episode=image_uri(figs["episode"]), fig_feb=image_uri(figs["feb14"]), fig_mar=image_uri(figs["mar08"]), fig_unc=image_uri(figs["uncertainty"]),
        extreme_table=dataframe_html(ctx["model_diag"], 2), examples_table=dataframe_html(ex, 3),
        alert_table=dataframe_html(ctx["alert_heads"], 2) if ctx["alert_heads"] is not None else "-", unc_table=dataframe_html(ctx["uncertainty"], 2) if ctx["uncertainty"] is not None else "-",
        neural_available=neural_available, neural_table=dataframe_html(ctx["neural"], 2) if neural_available else "", fig_neural=image_uri(figs["neural_alerts"]) if "neural_alerts" in figs else None,
    )
    path = report_dir/"relatorio_integrado_2024.html"
    path.write_text(html, encoding="utf-8")
    return path


def make_latex(report_dir: Path, figs: dict, ctx: dict, data: dict) -> Tuple[Path, Path, bool, str]:
    dist = ctx["dist"]
    moderate = ctx["moderate"]
    best = ctx["best_point"]
    best_name = latex_escape(best["model"] if best is not None else "-")
    best_rmse = fmt(best["RMSE_cm"] if best is not None else np.nan, 1)
    neural_available = ctx["neural"] is not None and len(ctx["neural"])

    # Tables
    diag = ctx["model_diag"].copy()
    extreme_rows = dataframe_latex_rows(diag, ["model", "MAE_m", "median_AE_m", "RMSE_m", "pred_mean_m", "underprediction_pct"], {
        "MAE_m": lambda x: f"{x:.2f}", "median_AE_m": lambda x: f"{x:.2f}", "RMSE_m": lambda x: f"{x:.2f}", "pred_mean_m": lambda x: f"{x:.2f}", "underprediction_pct": lambda x: f"{x:.1f}\\%"
    })
    decomp = ctx["decomp"].copy()
    decomp_rows = dataframe_latex_rows(decomp, ["actual_band_gt3", "n", "real_mean_m", "pred_mean_m", "MAE_m", "share_abs_error_pct", "share_squared_error_pct"], {
        "share_abs_error_pct": lambda x: f"{x:.1f}\\%", "share_squared_error_pct": lambda x: f"{x:.1f}\\%"
    })

    ex = ctx["examples"].copy()
    ex_cols = [c for c in ["prediction_time", "target_delta_120m", "pred_xgboost_plus_radar",
                           "pred_xgboost_normal", "pred_duas_etapas"] if c in ex.columns]
    ex = ex[ex_cols].copy()
    if "prediction_time" in ex.columns:
        ex["prediction_time"] = pd.to_datetime(
            ex["prediction_time"]).dt.strftime("%d/%m/%Y %H:%M")
    example_rows = dataframe_latex_rows(ex, ex_cols, {
        "target_delta_120m": lambda x: f"{x:.2f}",
        "pred_xgboost_plus_radar": lambda x: f"{x:.2f}",
        "pred_xgboost_normal": lambda x: f"{x:.2f}",
        "pred_duas_etapas": lambda x: f"{x:.2f}",
    })

    ah = ctx["alert_heads"]
    alert_rows = ""
    if ah is not None:
        alert_rows = dataframe_latex_rows(ah, ["model", "precision_pct", "recall_pct", "F2", "false_alerts"], {
            "precision_pct": lambda x: f"{x:.1f}\\%", "recall_pct": lambda x: f"{x:.1f}\\%", "F2": lambda x: f"{x:.3f}"
        })

    unc = ctx["uncertainty"]
    unc_rows = ""
    if unc is not None:
        unc_rows = dataframe_latex_rows(unc, ["method", "nominal_coverage_pct", "empirical_coverage_2024_pct", "mean_width_cm"], {
            "nominal_coverage_pct": lambda x: f"{x:.0f}\\%", "empirical_coverage_2024_pct": lambda x: f"{x:.1f}\\%", "mean_width_cm": lambda x: f"{x:.1f}"
        })

    neural_tex = ""
    if neural_available:
        nr = dataframe_latex_rows(
            ctx["neural"], ["model", "MAE_cm", "RMSE_cm", "max_abs_error_cm"])
        neural_tex = r"""
\section{Redes neurais temporais}
O notebook 07b foi executado. A TCN e a GRU recebem a sequencia causal das ultimas horas e possuem heads separadas para regressao e para os limiares de 1, 2 e 3 m.
\begin{table}[H]\centering\small
\begin{tabular}{lrrr}\toprule Modelo & MAE (cm) & RMSE (cm) & Maior erro (cm)\\\midrule
"""+nr+r"""\bottomrule\end{tabular}\caption{Comparacao das redes temporais em 2024.}\end{table}
"""
        if "neural_alerts" in figs:
            neural_tex += rf"\begin{{figure}}[H]\centering\includegraphics[width=.86\textwidth]{{{figs['neural_alerts'].as_posix()}}}\caption{{Recall das heads de severidade das redes temporais.}}\end{{figure}}"
    else:
        neural_tex = r"""
\section{Redes neurais temporais: etapa pendente}
O notebook 07b ainda nao foi executado neste conjunto de outputs. Ele foi desenhado para testar TCN e GRU usando uma janela causal de 6 h e quatro saidas simultaneas: $\widehat{\Delta H}_{2h}$, $P(\Delta H\ge1m)$, $P(\Delta H\ge2m)$ e $P(\Delta H\ge3m)$. A comparacao sera feita integralmente em 2024 antes de abrir 2025.
"""

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern,microtype}}
\usepackage[margin=2.35cm]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx,float,booktabs,tabularx,array}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{caption}}
\usepackage{{enumitem}}
\usepackage{{fancyhdr}}
\definecolor{{navy}}{{HTML}}{{244F76}}
\definecolor{{soft}}{{HTML}}{{EEF4F8}}
\definecolor{{warn}}{{HTML}}{{FFF4DA}}
\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=navy}}
\pagestyle{{fancy}}\fancyhf{{}}\lhead{{Previsao hidrologica - Tamanduatei}}\rhead{{Desenvolvimento 2024}}\cfoot{{\thepage}}\renewcommand{{\headrulewidth}}{{0.4pt}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{4pt}}\raggedbottom
\newcommand{{\callout}}[1]{{\begin{{center}}\fcolorbox{{navy}}{{soft}}{{\parbox{{.91\textwidth}}{{#1}}}}\end{{center}}}}
\newcommand{{\warningbox}}[1]{{\begin{{center}}\fcolorbox{{orange!70!black}}{{warn}}{{\parbox{{.91\textwidth}}{{#1}}}}\end{{center}}}}
\title{{\Huge\bfseries Relatorio integrado de previsao hidrologica\\[5pt]\Large Bacia do rio Tamanduatei}}
\author{{Pipeline TTI-HydroMet / iFAST}}
\date{{Avaliacao de desenvolvimento em 2024 - horizonte principal de 2 h}}
\begin{{document}}\maketitle
\begin{{abstract}}
Este relatorio reune a avaliacao dos modelos tabulares, radar, alertas, extremos e incerteza. O objetivo e tornar as metricas interpretaveis em termos de previsoes concretas. O periodo de 2025 permanece reservado para o teste cego final.
\end{{abstract}}
\tableofcontents\newpage

\section{{Resumo executivo}}
O melhor regressor tabular atual e \textbf{{{best_name}}}, com RMSE de \textbf{{{best_rmse} cm}} em 2024. A inclusao do radar reduziu o RMSE do XGBoost em aproximadamente \textbf{{{fmt(ctx['radar_gain'],1)}\%}}. Apesar disso, a magnitude das subidas extremas continua sendo o principal gargalo.
\callout{{A regressao e o alerta devem ser tratados como tarefas complementares. O regressor responde ``quanto deve subir?''; o classificador responde ``ha risco de ultrapassar um limiar perigoso?''.}}

\section{{Como interpretar MAE e RMSE}}
Para cada horario, o erro e a diferenca entre a variacao prevista e a variacao real. O MAE e a media dos erros absolutos. O RMSE eleva os erros ao quadrado antes de fazer a media e, portanto, pune mais fortemente falhas grandes. Assim, um RMSE de 20 cm nao significa que toda previsao erra 20 cm; significa que, no conjunto, os erros tem essa escala quadratica.

Formalmente, se $e_i=\widehat{{\Delta H}}_i-\Delta H_i$, entao
\[
\mathrm{{MAE}}=\frac1n\sum_i |e_i|,\qquad
\mathrm{{RMSE}}=\sqrt{{\frac1n\sum_i e_i^2}}.
\]

Como exemplo, para erros de 5, 5, 5 e 50 cm, o MAE vale 16,25 cm e o RMSE vale 25,4 cm. O unico erro de 50 cm aumenta muito mais o RMSE. Isso explica por que o RMSE e especialmente util neste projeto: uma previsao muito ruim durante uma enchente deve pesar mais do que uma pequena imprecisao em um periodo tranquilo.

\callout{{Outra consequencia importante: um RMSE global baixo pode coexistir com desempenho ruim nos extremos se a maioria dos timestamps for tranquila. Por isso este relatorio nunca usa apenas uma metrica global.}}
\begin{{figure}}[H]\centering\includegraphics[width=.88\textwidth]{{{figs['rmse_global'].as_posix()}}}\caption{{RMSE global dos modelos disponiveis em 2024.}}\end{{figure}}

\section{{O que o radar realmente acrescentou}}
A inclusao de resumos do radar reduziu o RMSE do XGBoost em aproximadamente \textbf{{{fmt(ctx['radar_gain'],1)}\%}} em 2024. O ganho e modesto em termos absolutos, mas aparece tambem na analise dos extremos: para subidas reais de pelo menos 1, 2 e 3 m, a versao com radar aumentou o recall da propria regressao. O radar, portanto, nao serve apenas para ``embelezar'' o erro medio; ele acrescenta alguma informacao antecipada sobre a precipitacao espacial.

Ao mesmo tempo, o classificador dedicado com radar nao domina o classificador sem radar: ele ganha recall por timestamp, mas perde precision e gera mais falsos alertas. Isso sugere que a utilidade do radar depende da tarefa. Na regressao continua, o ganho e claro; na cabeca de alerta, ainda existe um trade-off.

\section{{A dificuldade cresce com a severidade}}
\begin{{figure}}[H]\centering\includegraphics[width=.95\textwidth]{{{figs['severity'].as_posix()}}}\caption{{MAE por faixa da variacao real em duas horas.}}\end{{figure}}
A figura mostra que o bom desempenho global e dominado pelos periodos tranquilos, que sao muito mais frequentes. A qualidade cai rapidamente conforme a subida real se torna mais extrema.

\section{{O que significa MAE de cerca de 2,3 m nos casos acima de 3 m?}}
Em 2024 houve \textbf{{{dist['n']}}} timestamps com $\Delta H_{{2h}}>3$ m. Nesses casos, a subida real media foi \textbf{{{dist['actual_mean']:.2f} m}}, a mediana foi \textbf{{{dist['actual_median']:.2f} m}} e o maximo foi \textbf{{{dist['actual_max']:.2f} m}}.

\warningbox{{O erro alto nao e explicado apenas pelos poucos casos de 6--7 m. Entre os {dist['n']} timestamps acima de 3 m, {moderate['n']} estao entre 3,0 e 3,5 m. Nessa faixa, a subida real media foi {moderate['real_mean_m']:.2f} m, o XGBoost + radar previu em media {moderate['pred_mean_m']:.2f} m e o MAE ja foi {moderate['MAE_m']:.2f} m.}}

\begin{{figure}}[H]\centering\includegraphics[width=.80\textwidth]{{{figs['gt3_dist'].as_posix()}}}\caption{{Distribuicao das subidas reais acima de 3 m.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.68\textwidth]{{{figs['gt3_scatter'].as_posix()}}}\caption{{XGBoost + radar: subida prevista versus real para os casos acima de 3 m. A diagonal representa previsao perfeita.}}\end{{figure}}

\begin{{table}}[H]\centering\small
\begin{{tabular}}{{lrrrrr}}\toprule Modelo & MAE (m) & Mediana AE & RMSE & Prev. media (m) & Subestima (\%)\\\midrule
{extreme_rows}
\bottomrule\end{{tabular}}\caption{{Desempenho condicionado a subidas reais maiores que 3 m.}}\end{{table}}

No XGBoost + radar, o MAE e a mediana do erro absoluto sao ambos da ordem de 2,2 m. Isso e uma pista forte de que o resultado nao e produzido por um ou dois outliers: um caso ``tipico'' dentro do conjunto $\Delta H>3$ m ja apresenta erro muito grande.

\subsection{{Exemplos concretos: ha casos perto de 3 m previstos como quase zero?}}
Sim. A tabela abaixo foi escolhida para mostrar comportamentos diferentes, e nao apenas os maiores picos.
\begin{{table}}[H]\centering\scriptsize
\begin{{tabular}}{{lrrrr}}\toprule Horario & Real (m) & XGB+radar & XGB & Duas etapas\\\midrule
{example_rows}
\bottomrule\end{{tabular}}\caption{{Exemplos concretos entre os casos severos. A tabela inclui um grande evento, um erro grave perto de 3 m e um caso perto de 3 m previsto razoavelmente.}}\end{{table}}
Assim, interpretar ``MAE de 2,3 m'' como ``o modelo sempre previu 0,7 m quando subiu 3 m'' seria incorreto. Ha uma distribuicao: alguns casos de 3--3,5 m foram previstos perto do real, enquanto outros foram previstos perto de zero. A media resume essa mistura.

\subsection{{Os casos enormes estao inflando a media?}}
A tabela seguinte decompoe o erro do XGBoost + radar por subfaixa. Se os poucos eventos maiores que 6 m fossem a causa principal, eles dominariam a soma do erro. Isso nao acontece.
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{lrrrrrr}}\toprule Real (m) & n & Real media & Prev. media & MAE & \% erro abs. & \% erro quad.\\\midrule
{decomp_rows}
\bottomrule\end{{tabular}}\caption{{Contribuicao das subfaixas para o erro total dentro do conjunto $\Delta H>3$ m.}}\end{{table}}
\begin{{figure}}[H]\centering\includegraphics[width=.87\textwidth]{{{figs['decomp'].as_posix()}}}\caption{{Participacao de cada subfaixa no erro absoluto e quadratico.}}\end{{figure}}

\section{{Dois estudos de caso}}
\subsection{{14/02/2024}}
O maior pico de variacao em duas horas chegou a aproximadamente 7,05 m. Os modelos identificaram uma subida muito grande, mas subestimaram a magnitude.
\begin{{figure}}[H]\centering\includegraphics[width=.96\textwidth]{{{figs['feb14'].as_posix()}}}\caption{{Evolucao real e prevista em torno do episodio de 14/02/2024.}}\end{{figure}}
\subsection{{08/03/2024}}
Este episodio evidencia atraso de resposta: em alguns horarios, o futuro ja continha uma subida de 4--5 m e o regressor ainda estava proximo de zero. O radar melhorou a resposta, mas nao resolveu completamente o problema.
\begin{{figure}}[H]\centering\includegraphics[width=.96\textwidth]{{{figs['mar08'].as_posix()}}}\caption{{Evolucao real e prevista em torno do episodio de 08/03/2024.}}\end{{figure}}

\section{{Alerta: recall, precision e episodios}}
\begin{{figure}}[H]\centering\includegraphics[width=.72\textwidth]{{{figs['alert_pr'].as_posix()}}}\caption{{Trade-off entre recall e precision do classificador de evento $\Delta H\ge1$ m em duas horas.}}\end{{figure}}
\begin{{table}}[H]\centering\small\begin{{tabular}}{{lrrrr}}\toprule Modelo & Precision & Recall & F2 & Falsos alertas\\\midrule
{alert_rows}
\bottomrule\end{{tabular}}\caption{{Avaliacao por timestamp em 2024.}}\end{{table}}
\begin{{figure}}[H]\centering\includegraphics[width=.90\textwidth]{{{figs['episode'].as_posix()}}}\caption{{Avaliacao por episodio. Os classificadores atuais continuam sendo classificadores de $\ge1$ m, mesmo quando o pico real do episodio excede 2 ou 3 m.}}\end{{figure}}
\warningbox{{Interpretacao correta: dizer que um classificador de $\ge1$ m detectou 17 de 18 episodios cujo pico real passou de 3 m nao significa que ele previu 3 m. Significa apenas que emitiu algum alerta de pelo menos 1 m durante esses episodios. Essa lacuna motiva heads especificas para 2 m e 3 m.}}

\section{{Incerteza}}
\begin{{figure}}[H]\centering\includegraphics[width=.75\textwidth]{{{figs['uncertainty'].as_posix()}}}\caption{{Cobertura nominal versus empirica dos intervalos.}}\end{{figure}}
\begin{{table}}[H]\centering\small\begin{{tabular}}{{lrrr}}\toprule Metodo & Nominal & Empirica & Largura media (cm)\\\midrule
{unc_rows}
\bottomrule\end{{tabular}}\caption{{Calibracao global em 2024.}}\end{{table}}
Uma cobertura global adequada pode esconder baixa cobertura durante enchentes. O pipeline neural passa a relatar tambem cobertura condicional para subidas de 1, 2 e 3 m.

{neural_tex}

\section{{Conclusao antes do teste cego}}
\begin{{enumerate}}[leftmargin=*]
\item O XGBoost + radar permanece a referencia tabular para regressao continua.
\item O sistema de alerta deve ter uma tarefa dedicada e nao depender apenas da magnitude prevista pelo regressor.
\item O erro nos extremos acima de 3 m e sistematico e ja aparece fortemente na faixa 3--3,5 m; nao e apenas um artefato dos poucos eventos de 6--7 m.
\item TCN e GRU serao comparadas em 2024 com heads separadas para $\ge1$, $\ge2$ e $\ge3$ m.
\item O periodo de 2025 nao deve ser aberto ate que a arquitetura seja congelada.
\end{{enumerate}}

\section{{Limites sobre ``evitar perdas''}}
O dataset nao contem danos monetarios, vitimas, evacuacoes ou decisoes reais de defesa civil. Portanto, nao e defensavel estimar diretamente perdas evitadas. A utilidade operacional pode ser expressa por proxies: episodios antecipados, eventos perdidos, falsos alertas, antecedencia, erro de magnitude e cobertura de incerteza.

\section{{Referencias para a etapa temporal}}
\begin{{itemize}}[leftmargin=*]
\item \textit{{Application of temporal convolutional network for flood forecasting}}, Hydrology Research (2021), DOI 10.2166/nh.2021.021.
\item \textit{{Water Level Prediction Model Applying an LSTM--GRU Method for Flood Prediction}}, Water (2022), DOI 10.3390/w14142221.
\item \textit{{A Comprehensive Review of Methods for Hydrological Forecasting Based on Deep Learning}}, Water (2024), DOI 10.3390/w16101407.
\end{{itemize}}
\end{{document}}
"""
    tex_path = report_dir/"relatorio_integrado_2024.tex"
    tex_path.write_text(tex, encoding="utf-8")
    pdf_path = report_dir/"relatorio_integrado_2024.pdf"
    ok, log = compile_latex(tex_path, pdf_path)
    (report_dir/"latex_build.log").write_text(log, encoding="utf-8")
    return tex_path, pdf_path, ok, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--report-dir", default=None)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    outputs, data = load_inputs(root)
    report_dir = Path(args.report_dir).resolve(
    ) if args.report_dir else outputs/"reports"/"integrated_v7"
    fig_dir = report_dir/"figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    extreme = compute_extreme_diagnostics(data["point_predictions"])
    dist, diag, decomp, examples, moderate, model_cols = extreme
    # Save diagnostics as CSV for auditability.
    diag.to_csv(report_dir/"extremos_gt3_por_modelo.csv", index=False)
    decomp.to_csv(
        report_dir/"decomposicao_erro_gt3_xgb_radar.csv", index=False)
    examples.to_csv(report_dir/"exemplos_extremos_concretos.csv", index=False)
    (report_dir/"diagnostico_gt3_resumo.json").write_text(json.dumps(
        {**dist, "moderate_3_3_5": moderate}, indent=2, default=float), encoding="utf-8")

    figs = create_figures(fig_dir, data, extreme)
    ctx = build_context(data, extreme)
    html = make_html(report_dir, figs, ctx, data)
    tex, pdf, ok, log = make_latex(report_dir, figs, ctx, data)
    print("HTML:", html)
    print("TeX:", tex)
    print("PDF:", pdf if ok else "nao compilado - veja latex_build.log")
    print("Neural results included:", data.get("neural_point") is not None)


if __name__ == "__main__":
    main()
