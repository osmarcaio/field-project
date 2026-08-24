from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeConfig:
    base_threshold_m: float = 1.0
    gap_minutes: int = 60
    alert_gap_minutes: int = 30
    horizon_minutes: int = 120
    expected_step_minutes: int = 10


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes: {missing}")


def _as_time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z["prediction_time"] = pd.to_datetime(z["prediction_time"], errors="coerce")
    z = z.dropna(subset=["prediction_time"]).sort_values("prediction_time").reset_index(drop=True)
    return z


def build_base_episode_catalog(
    pred_df: pd.DataFrame,
    config: EpisodeConfig = EpisodeConfig(),
    target_col: str = "target_delta_120m",
) -> pd.DataFrame:
    """Build one common catalog of severe-rise episodes from the >=1 m label.

    Episodes are groups of timestamps whose *true* future rise is at least
    `base_threshold_m`. A temporal gap larger than `gap_minutes` starts a new
    episode. The catalog is retrospective and is meant for evaluation, not for
    online detection.
    """
    _require_columns(pred_df, ["prediction_time", target_col])
    df = _as_time_sorted(pred_df)
    active = df[pd.to_numeric(df[target_col], errors="coerce") >= config.base_threshold_m].copy()

    cols = [
        "episode_id", "base_start", "base_end", "n_base_timestamps",
        "peak_delta_m", "peak_signal_time", "reference_peak_time",
    ]
    if active.empty:
        return pd.DataFrame(columns=cols)

    new_ep = active["prediction_time"].diff().gt(pd.Timedelta(minutes=config.gap_minutes)).fillna(True)
    active["episode_id"] = new_ep.cumsum().astype(int)

    rows: List[Dict] = []
    for eid, g in active.groupby("episode_id", sort=True):
        g = g.sort_values("prediction_time")
        peak_idx = pd.to_numeric(g[target_col], errors="coerce").idxmax()
        peak_signal_time = g.loc[peak_idx, "prediction_time"]
        rows.append({
            "episode_id": int(eid),
            "base_start": g["prediction_time"].min(),
            "base_end": g["prediction_time"].max(),
            "n_base_timestamps": int(len(g)),
            "peak_delta_m": float(g.loc[peak_idx, target_col]),
            "peak_signal_time": peak_signal_time,
            # If the strongest 2 h rise is forecast from t, the end of that
            # strongest two-hour window is t + horizon.
            "reference_peak_time": peak_signal_time + pd.Timedelta(minutes=config.horizon_minutes),
        })
    return pd.DataFrame(rows)


def threshold_episode_windows(
    pred_df: pd.DataFrame,
    base_catalog: pd.DataFrame,
    threshold_m: float,
    target_col: str = "target_delta_120m",
) -> pd.DataFrame:
    """Return threshold-specific active intervals inside the common episodes."""
    _require_columns(pred_df, ["prediction_time", target_col])
    df = _as_time_sorted(pred_df)
    rows: List[Dict] = []

    for _, ep in base_catalog.iterrows():
        if float(ep["peak_delta_m"]) < float(threshold_m):
            continue
        g = df[
            (df["prediction_time"] >= ep["base_start"])
            & (df["prediction_time"] <= ep["base_end"])
            & (pd.to_numeric(df[target_col], errors="coerce") >= threshold_m)
        ]
        if g.empty:
            continue
        row = ep.to_dict()
        row.update({
            "threshold_m": float(threshold_m),
            "active_start": g["prediction_time"].min(),
            "active_end": g["prediction_time"].max(),
            "n_active_timestamps": int(len(g)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def persistent_alert_mask(
    pred_df: pd.DataFrame,
    score_col: str,
    score_threshold: float,
    min_consecutive: int = 1,
    expected_step_minutes: int = 10,
) -> np.ndarray:
    """Return an alert mask with an optional persistence requirement.

    With `min_consecutive=2`, an alert only becomes active at the second
    consecutive 10-minute exceedance. Gaps in timestamps reset the streak.
    """
    _require_columns(pred_df, ["prediction_time", score_col])
    df = _as_time_sorted(pred_df)
    raw = pd.to_numeric(df[score_col], errors="coerce").ge(float(score_threshold)).to_numpy()
    if min_consecutive <= 1:
        return raw

    times = df["prediction_time"].to_numpy(dtype="datetime64[ns]")
    mask = np.zeros(len(df), dtype=bool)
    streak = 0
    expected = np.timedelta64(int(expected_step_minutes), "m")
    for i in range(len(df)):
        if i > 0 and times[i] - times[i - 1] != expected:
            streak = 0
        if raw[i]:
            streak += 1
        else:
            streak = 0
        mask[i] = streak >= int(min_consecutive)
    return mask


def cluster_alert_episodes(
    pred_df: pd.DataFrame,
    alert_mask: np.ndarray,
    alert_gap_minutes: int = 30,
) -> pd.DataFrame:
    """Collapse repeated 10-minute alerts into alert episodes."""
    df = _as_time_sorted(pred_df)
    if len(alert_mask) != len(df):
        raise ValueError("alert_mask precisa ter o mesmo numero de linhas do dataframe ordenado.")
    z = df.loc[np.asarray(alert_mask, dtype=bool), ["prediction_time"]].copy()
    cols = ["alert_episode_id", "alert_start", "alert_end", "n_alert_timestamps"]
    if z.empty:
        return pd.DataFrame(columns=cols)

    new_group = z["prediction_time"].diff().gt(pd.Timedelta(minutes=alert_gap_minutes)).fillna(True)
    z["alert_episode_id"] = new_group.cumsum().astype(int)
    out = z.groupby("alert_episode_id", as_index=False).agg(
        alert_start=("prediction_time", "min"),
        alert_end=("prediction_time", "max"),
        n_alert_timestamps=("prediction_time", "size"),
    )
    return out


def _timing_class(lead_to_peak_min: float) -> str:
    if not np.isfinite(lead_to_peak_min):
        return "nao_detectado"
    if lead_to_peak_min >= 120:
        return ">=120min_antes_pico"
    if lead_to_peak_min >= 60:
        return "60-119min_antes_pico"
    if lead_to_peak_min >= 30:
        return "30-59min_antes_pico"
    if lead_to_peak_min >= 0:
        return "0-29min_antes_pico"
    return "apos_pico_referencia"


def evaluate_episode_alerts(
    pred_df: pd.DataFrame,
    score_col: str,
    score_threshold: float,
    severity_threshold_m: float,
    config: EpisodeConfig = EpisodeConfig(),
    target_col: str = "target_delta_120m",
    min_consecutive: int = 1,
    base_catalog: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Evaluate alert usefulness at episode scale.

    Important: the thresholds 1/2/3 m are research thresholds for *future rise*,
    not official flood-stage thresholds. Lead time is therefore measured to a
    reference peak of the severe-rise episode, not to an official critical stage.
    """
    df = _as_time_sorted(pred_df)
    _require_columns(df, ["prediction_time", target_col, score_col])

    if base_catalog is None:
        base_catalog = build_base_episode_catalog(df, config=config, target_col=target_col)
    events = threshold_episode_windows(df, base_catalog, severity_threshold_m, target_col=target_col)

    alert_mask = persistent_alert_mask(
        df,
        score_col=score_col,
        score_threshold=score_threshold,
        min_consecutive=min_consecutive,
        expected_step_minutes=config.expected_step_minutes,
    )
    df = df.copy()
    df["alert_active"] = alert_mask
    alert_eps = cluster_alert_episodes(df, alert_mask, alert_gap_minutes=config.alert_gap_minutes)

    event_rows: List[Dict] = []
    for _, ev in events.iterrows():
        hits = df[
            (df["prediction_time"] >= ev["active_start"])
            & (df["prediction_time"] <= ev["active_end"])
            & df["alert_active"]
        ]
        detected = not hits.empty
        first_alert = hits["prediction_time"].min() if detected else pd.NaT
        delay = (
            (first_alert - ev["active_start"]).total_seconds() / 60.0
            if detected else np.nan
        )
        lead = (
            (ev["reference_peak_time"] - first_alert).total_seconds() / 60.0
            if detected else np.nan
        )
        row = ev.to_dict()
        row.update({
            "score_col": score_col,
            "score_threshold": float(score_threshold),
            "min_consecutive": int(min_consecutive),
            "detected": bool(detected),
            "first_alert_time": first_alert,
            "detection_delay_from_active_start_min": delay,
            "lead_to_reference_peak_min": lead,
            "timing_class": _timing_class(lead),
            "alert_timestamps_inside_event": int(len(hits)),
        })
        event_rows.append(row)
    event_eval = pd.DataFrame(event_rows)

    # Episode-level alert precision: a predicted alert cluster is considered a
    # true alert episode if it overlaps at least one true threshold-active window.
    alert_match_rows: List[Dict] = []
    for _, a in alert_eps.iterrows():
        overlaps = events[
            (events["active_end"] >= a["alert_start"])
            & (events["active_start"] <= a["alert_end"])
        ] if not events.empty else pd.DataFrame()
        alert_match_rows.append({
            **a.to_dict(),
            "matched_true_episode": bool(len(overlaps)),
            "matched_episode_ids": ",".join(map(str, overlaps["episode_id"].astype(int).tolist())) if len(overlaps) else "",
        })
    alert_eval = pd.DataFrame(alert_match_rows)

    n_events = int(len(events))
    n_detected = int(event_eval["detected"].sum()) if len(event_eval) else 0
    n_alert_eps = int(len(alert_eval))
    n_true_alert_eps = int(alert_eval["matched_true_episode"].sum()) if len(alert_eval) else 0

    detected_leads = event_eval.loc[event_eval["detected"], "lead_to_reference_peak_min"].astype(float) if len(event_eval) else pd.Series(dtype=float)
    delays = event_eval.loc[event_eval["detected"], "detection_delay_from_active_start_min"].astype(float) if len(event_eval) else pd.Series(dtype=float)

    metrics: Dict[str, float] = {
        "severity_threshold_m": float(severity_threshold_m),
        "score_threshold": float(score_threshold),
        "min_consecutive": int(min_consecutive),
        "episodes": n_events,
        "episodes_detected": n_detected,
        "episode_recall_pct": 100.0 * n_detected / n_events if n_events else np.nan,
        "alert_episodes": n_alert_eps,
        "true_alert_episodes": n_true_alert_eps,
        "false_alert_episodes": n_alert_eps - n_true_alert_eps,
        "episode_precision_pct": 100.0 * n_true_alert_eps / n_alert_eps if n_alert_eps else np.nan,
        "median_detection_delay_min": float(delays.median()) if len(delays) else np.nan,
        "median_lead_to_reference_peak_min": float(detected_leads.median()) if len(detected_leads) else np.nan,
    }
    for lead_min in (30, 60, 90, 120):
        metrics[f"episode_recall_with_lead_ge_{lead_min}min_pct"] = (
            100.0 * float((detected_leads >= lead_min).sum()) / n_events if n_events else np.nan
        )
    metrics["post_peak_detection_pct_of_all_episodes"] = (
        100.0 * float((detected_leads < 0).sum()) / n_events if n_events else np.nan
    )
    return event_eval, alert_eval, metrics


def point_classification_metrics(
    pred_df: pd.DataFrame,
    score_col: str,
    score_threshold: float,
    severity_threshold_m: float,
    target_col: str = "target_delta_120m",
    min_consecutive: int = 1,
    expected_step_minutes: int = 10,
) -> Dict[str, float]:
    df = _as_time_sorted(pred_df)
    _require_columns(df, [target_col, score_col])
    alert = persistent_alert_mask(
        df, score_col, score_threshold,
        min_consecutive=min_consecutive,
        expected_step_minutes=expected_step_minutes,
    )
    real = pd.to_numeric(df[target_col], errors="coerce").ge(severity_threshold_m).to_numpy()
    tp = int(np.sum(real & alert))
    fn = int(np.sum(real & ~alert))
    fp = int(np.sum(~real & alert))
    tn = int(np.sum(~real & ~alert))
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan
    beta = 2.0
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if np.isfinite(precision) and np.isfinite(recall) and beta**2 * precision + recall else np.nan
    return {
        "severity_threshold_m": float(severity_threshold_m),
        "score_threshold": float(score_threshold),
        "min_consecutive": int(min_consecutive),
        "real_events": int(real.sum()),
        "predicted_alerts": int(alert.sum()),
        "true_positives": tp,
        "missed_events": fn,
        "false_alerts": fp,
        "true_negatives": tn,
        "recall_pct": 100 * recall if np.isfinite(recall) else np.nan,
        "precision_pct": 100 * precision if np.isfinite(precision) else np.nan,
        "F1": f1,
        "F2": f2,
    }
