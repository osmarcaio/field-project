from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def img_uri(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def fmt(v, digits=1):
    if pd.isna(v):
        return "-"
    return f"{float(v):.{digits}f}"


def html_table(df: pd.DataFrame, cols=None) -> str:
    z = df.copy()
    if cols:
        z = z[cols]
    return z.to_html(index=False, classes="data-table", border=0, escape=False)


def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def build(project_root: Path) -> Path:
    root = project_root.resolve()
    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    models = root / "models"
    out = root / "outputs" / "reports" / "operational_v7"
    out.mkdir(parents=True, exist_ok=True)

    ep = pd.read_csv(tables / "operational_episode_metrics_2024_v7.csv")
    pt = pd.read_csv(tables / "operational_point_alert_metrics_2024_v7.csv")
    hybrid = pd.read_csv(tables / "hybrid_point_metrics_2024_v7.csv")
    catalog = pd.read_csv(tables / "episode_catalog_2024_v7.csv")
    arch = json.loads((models / "architecture_candidate_v7.json").read_text(encoding="utf-8"))

    persistence = int(arch["severity_signals"]["persistence_steps"])
    tcn_raw = pt[(pt.model == "TCN") & (pt.min_consecutive == 1)].copy()
    tcn_p = pt[(pt.model == "TCN") & (pt.min_consecutive == persistence)].copy()
    ep_raw = ep[(ep.model == "TCN") & (ep.min_consecutive == 1)].copy()
    ep_p = ep[(ep.model == "TCN") & (ep.min_consecutive == persistence)].copy()

    quick = ep_p[[
        "severity_threshold_m", "episodes", "episodes_detected", "episode_recall_pct",
        "episode_precision_pct", "median_detection_delay_min", "median_lead_to_reference_peak_min",
        "episode_recall_with_lead_ge_120min_pct",
    ]].copy()
    quick.columns = [
        "Faixa", "Episodios", "Detectados", "Recall episodio (%)", "Precision blocos (%)",
        "Atraso mediano (min)", "Antecedencia mediana ao pico (min)", "Recall com >=120 min (%)",
    ]
    quick["Faixa"] = quick["Faixa"].map(lambda x: f">= {x:g} m")
    for col in quick.columns[3:]:
        quick[col] = quick[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    point_quick = tcn_raw[["severity_threshold_m", "recall_pct", "precision_pct", "real_events", "false_alerts"]].copy()
    point_quick.columns = ["Faixa", "Recall timestamp (%)", "Precision timestamp (%)", "Janelas reais", "Falsos sinais"]
    point_quick["Faixa"] = point_quick["Faixa"].map(lambda x: f">= {x:g} m")
    point_quick["Recall timestamp (%)"] = point_quick["Recall timestamp (%)"].map(lambda x: f"{x:.1f}")
    point_quick["Precision timestamp (%)"] = point_quick["Precision timestamp (%)"].map(lambda x: f"{x:.1f}")

    h = hybrid.set_index("model")
    hybrid_mae = float(h.loc["Hibrido por regime v7", "MAE_cm"])
    hybrid_rmse = float(h.loc["Hibrido por regime v7", "RMSE_cm"])
    xgb_rmse = float(h.loc["XGBoost + radar", "RMSE_cm"])
    switch_pct = float(h.loc["Hibrido por regime v7", "risk_regime_fraction_pct"])

    html = f"""<!doctype html>
<html lang='pt-BR'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>TTI-HydroMet — avaliacao operacional v7</title>
<style>
:root{{--ink:#172033;--muted:#5d6678;--line:#dfe3ea;--panel:#f7f8fb;--accent:#1f5f8b}}
*{{box-sizing:border-box}} body{{margin:0;font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:#fff}}
main{{max-width:1080px;margin:auto;padding:38px 28px 72px}} h1{{font-size:34px;line-height:1.15;margin:0 0 10px}} h2{{margin-top:42px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.lead{{font-size:18px;color:var(--muted);max-width:900px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:24px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}} .value{{font-size:26px;font-weight:700}} .label{{color:var(--muted);font-size:13px}}
.note{{border-left:4px solid var(--accent);background:#f2f7fb;padding:13px 16px;margin:18px 0}} .warn{{border-left-color:#a55b15;background:#fff7ed}}
.data-table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0 26px}} .data-table th,.data-table td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}} .data-table th:first-child,.data-table td:first-child{{text-align:left}}
.figure{{margin:25px 0}} .figure img{{max-width:100%;border:1px solid var(--line);border-radius:8px}} code{{background:#f1f3f6;padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<h1>Avaliacao operacional por episodio — 2024</h1>
<p class='lead'>Estacao 413, horizonte de 2 h. Esta versao separa qualidade numerica, deteccao por episodio e antecedencia. Os limiares de 1/2/3 m representam <b>subida futura</b>; nao sao cotas oficiais.</p>
<div class='cards'>
<div class='card'><div class='value'>{hybrid_mae:.2f} cm</div><div class='label'>MAE do hibrido</div></div>
<div class='card'><div class='value'>{hybrid_rmse:.2f} cm</div><div class='label'>RMSE do hibrido</div></div>
<div class='card'><div class='value'>{switch_pct:.2f}%</div><div class='label'>timestamps roteados para TCN+GRU</div></div>
<div class='card'><div class='value'>{len(catalog)}</div><div class='label'>episodios >=1 m pela nova segmentacao</div></div>
</div>
<div class='note'><b>Arquitetura candidata.</b> XGBoost+radar fornece a magnitude no regime normal. Se a TCN identifica padrao de subida >=1 m, a magnitude passa a ser a media TCN+GRU. As heads da TCN fornecem sinais experimentais >=1/2/3 m.</div>
<h2>1. Por que episodio?</h2>
<p>Uma tempestade severa pode produzir dezenas de previsoes consecutivas de 10 em 10 minutos. Essas previsoes continuam importantes, mas nao devem ser interpretadas como dezenas de enchentes independentes. A v7 cria um catalogo comum de episodios a partir de <code>DeltaH_2h >= 1 m</code> e separa episodios quando existe mais de 60 min sem essa condicao.</p>
<p>Com essa regra, 2024 possui <b>{len(catalog)}</b> episodios >=1 m; <b>{int((catalog.peak_delta_m>=2).sum())}</b> chegaram a >=2 m e <b>{int((catalog.peak_delta_m>=3).sum())}</b> a >=3 m.</p>
<h2>2. Metricas tradicionais por timestamp</h2>
{html_table(point_quick)}
<p>Precision baixa significa que muitos timestamps sinalizados nao correspondem a uma janela realmente acima daquela faixa. Isso nao deve ser escondido pelo bom recall por episodio.</p>
<h2>3. Metricas por episodio com persistencia de {persistence} timestamps</h2>
{html_table(quick)}
<div class='note warn'><b>Interpretacao do tempo.</b> Sem cotas oficiais, a antecedencia e medida ate o fim da janela de 2 h com a maior subida observada no episodio (“pico de referencia”). Ela e uma proxy de oportunidade operacional, nao tempo ate uma cota de emergencia.</div>
<div class='figure'><img src='{img_uri(figures / "v7_episode_recall_vs_persistence.png")}'><p>Exigir persistencia reduz flicker e falsos blocos, mas tambem perde alguns episodios. A escolha foi feita em 2024, antes do teste cego.</p></div>
<div class='figure'><img src='{img_uri(figures / "v7_episode_lead_time_distribution.png")}'><p>Distribuicao da antecedencia do primeiro sinal correto em relacao ao pico de referencia.</p></div>
<h2>4. O que a v7 penaliza melhor</h2>
<p>Um modelo que so reage perto do pico ja nao recebe exatamente o mesmo credito de um modelo que reconhece o episodio cedo. O relatorio mostra recall com pelo menos 30, 60, 90 e 120 min de antecedencia, alem do atraso desde o inicio da faixa severa.</p>
<h2>5. Limites atuais</h2>
<p>O projeto ainda nao conhece as cotas oficiais de atencao/alerta/emergencia da 413. Por isso nao e correto dizer que um sinal de 3 m equivale a “emergencia”. Alem disso, o radar usado pela arquitetura principal continua espacialmente agregado; o notebook 06b testa se os 434 pixels contem ganho adicional.</p>
<h2>6. Conclusao</h2>
<p>A avaliacao v7 preserva as previsoes de 10 min, mas passa a responder uma pergunta operacional mais apropriada: <b>quantos episodios foram percebidos, com que antecedencia e com quantos falsos blocos de sinal?</b> O holdout de 2025 continua reservado para o notebook 08.</p>
</main></body></html>"""
    html_path = out / "relatorio_operacional_episodios_2024_v7.html"
    html_path.write_text(html, encoding="utf-8")

    # Compact LaTeX source. It can be compiled locally if pdflatex/latexmk exists.
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}\usepackage{booktabs}\usepackage{graphicx}\usepackage{float}\usepackage{microtype}\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage[brazil]{babel}\usepackage{hyperref}\usepackage{xcolor}
\title{Avaliacao operacional por episodio -- TTI-HydroMet v7}\author{Field Project}\date{2024 development test}
\begin{document}\maketitle
\section{Escopo}
O alvo e exclusivamente o nivel da estacao 413. Os limiares de 1, 2 e 3 m sao faixas de subida futura em duas horas e nao cotas oficiais.
\section{Arquitetura candidata}
No regime normal, a magnitude vem do XGBoost com radar. Quando a TCN ultrapassa o limiar de subida de 1 m, usa-se a media das magnitudes TCN e GRU. As heads da TCN fornecem sinais de severidade.
\section{Resumo numerico}
"""
    tex += f"MAE do hibrido: {hybrid_mae:.2f} cm. RMSE: {hybrid_rmse:.2f} cm. O roteamento neural ocorre em {switch_pct:.2f}\\% dos timestamps.\\\n"
    tex += r"\section{Avaliacao por episodio}" + "\n"
    tex += "O catalogo v7 contem %d episodios com subida de pelo menos 1 m; %d chegam a 2 m e %d a 3 m.\\\n" % (len(catalog), int((catalog.peak_delta_m>=2).sum()), int((catalog.peak_delta_m>=3).sum()))
    tex += r"\begin{table}[H]\centering\small\begin{tabular}{lrrrr}\toprule Faixa & Episodios & Recall (\%) & Precision blocos (\%) & Lead mediano (min)\\\midrule" + "\n"
    for _, r in ep_p.sort_values("severity_threshold_m").iterrows():
        tex += f">= {r.severity_threshold_m:g} m & {int(r.episodes)} & {r.episode_recall_pct:.1f} & {r.episode_precision_pct:.1f} & {r.median_lead_to_reference_peak_min:.0f} \\\\\n"
    tex += r"\bottomrule\end{tabular}\caption{Metricas por episodio com persistencia operacional.}\end{table}" + "\n"
    for fn, cap in [("v7_episode_recall_vs_persistence.png","Recall por episodio versus persistencia."),("v7_episode_lead_time_distribution.png","Antecedencia do primeiro sinal ao pico de referencia.")]:
        p = figures / fn
        if p.exists():
            tex += f"\\begin{{figure}}[H]\\centering\\includegraphics[width=.9\\textwidth]{{{p.as_posix()}}}\\caption{{{cap}}}\\end{{figure}}\n"
    tex += r"\section{Limites}Sem cotas oficiais, a antecedencia e medida em relacao a um pico de referencia do episodio, nao a uma cota de emergencia.\end{document}"
    tex_path = out / "relatorio_operacional_episodios_2024_v7.tex"
    tex_path.write_text(tex, encoding="utf-8")

    log_lines = []
    compiler = shutil.which("latexmk") or shutil.which("pdflatex")
    if compiler:
        try:
            if Path(compiler).name.lower().startswith("latexmk"):
                cmd = [compiler, "-pdf", "-interaction=nonstopmode", tex_path.name]
            else:
                cmd = [compiler, "-interaction=nonstopmode", tex_path.name]
            result = subprocess.run(cmd, cwd=out, capture_output=True, text=True, timeout=120)
            log_lines.append(result.stdout)
            log_lines.append(result.stderr)
        except Exception as exc:
            log_lines.append(str(exc))
    else:
        log_lines.append("Nenhum latexmk/pdflatex encontrado. O .tex foi gerado normalmente.")
    (out / "latex_build_v7.log").write_text("\n".join(log_lines), encoding="utf-8")
    return html_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args()
    path = build(Path(args.project_root))
    print("Relatorio v7:", path)


if __name__ == "__main__":
    main()
