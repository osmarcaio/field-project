"""Utilities for causal temporal neural models in the TTI-HydroMet project.

Designed for notebooks 07b and 08. The module deliberately keeps all time
alignment explicit to reduce leakage risk.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import random

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the temporal models. Install it in the project venv with: "
        "python -m pip install torch"
    ) from exc


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # We favor reproducibility over peak speed here.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

DEFAULT_STAGE_COLS = ["stage_413", "stage_1000490", "stage_143", "stage_283"]


def _deduplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.has_duplicates:
        return df.sort_index()
    return df.groupby(level=0).mean(numeric_only=True).sort_index()


def select_instantaneous_radar_columns(radar: pd.DataFrame) -> List[str]:
    """Select non-rolling radar summaries for a sequence model.

    The temporal neural network gets the chronology itself, so feeding only
    instantaneous radar summaries avoids duplicating the hand-crafted rolling
    windows already used by the tabular models.
    """
    preferred = [
        "radar_mean",
        "radar_max",
        "radar_p75",
        "radar_p90",
        "radar_p95",
        "radar_wet_fraction",
        "radar_cells_available",
        "radar_failure",
    ]
    cols = [c for c in preferred if c in radar.columns]
    if cols:
        return cols

    # Robust fallback: keep columns that do not visibly encode rolling windows.
    forbidden = ("_1h", "_3h", "_6h", "_12h", "rolling", "sum_", "mean_")
    return [
        c for c in radar.columns
        if c.startswith("radar_") and not any(token in c for token in forbidden)
    ]


def build_temporal_frame(
    master: pd.DataFrame,
    radar: Optional[pd.DataFrame] = None,
    include_time_features: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Build raw causal channels for temporal models.

    Channels are *raw/instantaneous* rather than the 166 engineered features:
    levels, 10-minute rainfall observations, a few instantaneous radar summaries,
    and cyclical time-of-day/year features. Missingness masks are added later by
    the preprocessor.
    """
    master = master.sort_index().copy()

    stage_cols = [c for c in DEFAULT_STAGE_COLS if c in master.columns]
    rain_cols = [c for c in master.columns if c.startswith("rain_")]
    cols = stage_cols + rain_cols
    out = master[cols].copy()

    radar_cols: List[str] = []
    if radar is not None:
        radar = _deduplicate_index(radar)
        radar_cols = select_instantaneous_radar_columns(radar)
        if radar_cols:
            out = out.join(radar[radar_cols], how="left")

    time_cols: List[str] = []
    if include_time_features:
        idx = out.index
        minute_of_day = idx.hour * 60 + idx.minute
        day_of_year = idx.dayofyear
        out["time_sin_day"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        out["time_cos_day"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
        out["time_sin_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
        out["time_cos_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
        time_cols = ["time_sin_day", "time_cos_day", "time_sin_year", "time_cos_year"]

    groups = {
        "stage": stage_cols,
        "rain": rain_cols,
        "radar": radar_cols,
        "time": time_cols,
    }
    return out.astype("float32"), groups


@dataclass
class TemporalPreprocessorState:
    input_columns: List[str]
    output_columns: List[str]
    stage_columns: List[str]
    rain_columns: List[str]
    radar_columns: List[str]
    time_columns: List[str]
    medians: Dict[str, float]
    means: Dict[str, float]
    stds: Dict[str, float]
    target_mean: float
    target_std: float
    stage_ffill_limit: int = 6

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "TemporalPreprocessorState":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class TemporalPreprocessor:
    """Causal missing-value handling + scaling fitted only on training data.

    - Stage: causal forward-fill for at most one hour (6 x 10 min), then training median.
    - Rain/radar: do *not* forward-fill; use training median and append missingness masks.
      This keeps "missing" distinct from "observed zero", because the mask is a channel.
    - Time features: already finite and simply standardized.
    """

    def __init__(self, groups: Dict[str, List[str]], stage_ffill_limit: int = 6):
        self.groups = groups
        self.stage_ffill_limit = stage_ffill_limit
        self.state: Optional[TemporalPreprocessorState] = None

    def fit(self, raw: pd.DataFrame, train_mask: pd.Series, target: pd.Series) -> "TemporalPreprocessor":
        input_columns = list(raw.columns)
        train_raw = raw.loc[train_mask]

        medians: Dict[str, float] = {}
        for c in input_columns:
            val = float(train_raw[c].median(skipna=True))
            if not np.isfinite(val):
                val = 0.0
            medians[c] = val

        # Transform once to estimate scaling stats without leaking validation/test.
        transformed = self._impute_and_mask(raw, medians)
        train_transformed = transformed.loc[train_mask]

        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for c in transformed.columns:
            mu = float(train_transformed[c].mean())
            sd = float(train_transformed[c].std())
            if not np.isfinite(mu):
                mu = 0.0
            if not np.isfinite(sd) or sd < 1e-6:
                sd = 1.0
            means[c] = mu
            stds[c] = sd

        ytrain = pd.to_numeric(target.loc[train_mask], errors="coerce").dropna()
        target_mean = float(ytrain.mean())
        target_std = float(ytrain.std())
        if not np.isfinite(target_std) or target_std < 1e-6:
            target_std = 1.0

        self.state = TemporalPreprocessorState(
            input_columns=input_columns,
            output_columns=list(transformed.columns),
            stage_columns=list(self.groups.get("stage", [])),
            rain_columns=list(self.groups.get("rain", [])),
            radar_columns=list(self.groups.get("radar", [])),
            time_columns=list(self.groups.get("time", [])),
            medians=medians,
            means=means,
            stds=stds,
            target_mean=target_mean,
            target_std=target_std,
            stage_ffill_limit=self.stage_ffill_limit,
        )
        return self

    def _impute_and_mask(self, raw: pd.DataFrame, medians: Dict[str, float]) -> pd.DataFrame:
        z = raw.copy()
        mask = z.isna().astype("float32")

        # Causal short ffill for water levels only.
        stage_cols = [c for c in self.groups.get("stage", []) if c in z.columns]
        if stage_cols:
            z[stage_cols] = z[stage_cols].ffill(limit=self.stage_ffill_limit)

        for c in z.columns:
            z[c] = z[c].fillna(medians.get(c, 0.0))

        # Missingness channels are useful for sensors, not deterministic time features.
        mask_source = [
            c for c in z.columns
            if c not in self.groups.get("time", [])
        ]
        mask = mask[mask_source].rename(columns={c: f"missing__{c}" for c in mask_source})
        return pd.concat([z, mask], axis=1).astype("float32")

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if self.state is None:
            raise RuntimeError("TemporalPreprocessor must be fitted before transform().")
        z = self._impute_and_mask(raw[self.state.input_columns], self.state.medians)
        z = z[self.state.output_columns]
        for c in z.columns:
            z[c] = (z[c] - self.state.means[c]) / self.state.stds[c]
        return z.astype("float32")

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Preprocessor not fitted.")
        return ((y - self.state.target_mean) / self.state.target_std).astype("float32")

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Preprocessor not fitted.")
        return y_scaled * self.state.target_std + self.state.target_mean


class SequenceDataset(Dataset):
    """Lazy overlapping sequence dataset; avoids materializing N x L x C arrays."""

    def __init__(
        self,
        matrix: np.ndarray,
        y_reg_scaled: np.ndarray,
        y_reg_raw: np.ndarray,
        indices: np.ndarray,
        lookback_steps: int,
        event_thresholds: Sequence[float] = (1.0, 2.0, 3.0),
    ):
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.y_reg_scaled = np.asarray(y_reg_scaled, dtype=np.float32)
        self.y_reg_raw = np.asarray(y_reg_raw, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.lookback_steps = int(lookback_steps)
        self.thresholds = np.asarray(event_thresholds, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        end = int(self.indices[i])
        start = end - self.lookback_steps + 1
        x = self.matrix[start:end + 1]
        y_scaled = self.y_reg_scaled[end]
        y_raw = self.y_reg_raw[end]
        y_cls = (y_raw >= self.thresholds).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.tensor(y_scaled, dtype=torch.float32),
            torch.from_numpy(y_cls),
            torch.tensor(y_raw, dtype=torch.float32),
            torch.tensor(end, dtype=torch.long),
        )


def valid_sequence_indices(
    index: pd.DatetimeIndex,
    split: pd.Series,
    target: pd.Series,
    split_name: str,
    lookback_steps: int,
    stride: int = 1,
) -> np.ndarray:
    split_arr = split.reindex(index).to_numpy()
    y = target.reindex(index).to_numpy()
    positions = np.arange(len(index))
    mask = (split_arr == split_name) & np.isfinite(y) & (positions >= lookback_steps - 1)
    idx = positions[mask]
    return idx[::max(1, int(stride))]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.net(x) + self.downsample(x))


class MultiTaskHeads(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg = nn.Linear(hidden_dim, 1)
        self.cls = nn.Linear(hidden_dim, 3)

    def forward(self, h):
        h = self.shared(h)
        return self.reg(h).squeeze(-1), self.cls(h)


class TCNMultiTask(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: Sequence[int] = (64, 64, 96, 96),
        kernel_size: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        blocks = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        self.heads = MultiTaskHeads(channels[-1], dropout=dropout)

    def forward(self, x):
        # x: batch x time x features -> batch x features x time
        h = self.tcn(x.transpose(1, 2))
        h_last = h[:, :, -1]
        return self.heads(h_last)


class GRUMultiTask(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.heads = MultiTaskHeads(hidden_dim, dropout=dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        h_last = self.norm(out[:, -1, :])
        return self.heads(h_last)


# -----------------------------------------------------------------------------
# Training and metrics
# -----------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    model_name: str
    lookback_steps: int = 36
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    regression_event_weights: Tuple[float, float, float, float] = (1.0, 1.5, 2.5, 4.0)
    classification_loss_weight: float = 0.35
    seed: int = 42


def compute_pos_weights(y_train_raw: np.ndarray, thresholds=(1.0, 2.0, 3.0), cap: float = 60.0) -> torch.Tensor:
    vals = []
    for th in thresholds:
        pos = np.sum(y_train_raw >= th)
        neg = np.sum(y_train_raw < th)
        ratio = neg / max(pos, 1)
        vals.append(float(np.clip(ratio, 1.0, cap)))
    return torch.tensor(vals, dtype=torch.float32)


def regression_sample_weights(y_raw: torch.Tensor, weights=(1.0, 1.5, 2.5, 4.0)) -> torch.Tensor:
    w = torch.full_like(y_raw, float(weights[0]))
    w = torch.where(y_raw >= 1.0, torch.tensor(float(weights[1]), device=y_raw.device), w)
    w = torch.where(y_raw >= 2.0, torch.tensor(float(weights[2]), device=y_raw.device), w)
    w = torch.where(y_raw >= 3.0, torch.tensor(float(weights[3]), device=y_raw.device), w)
    return w


def _loss_fn(
    reg_pred_scaled: torch.Tensor,
    cls_logits: torch.Tensor,
    y_reg_scaled: torch.Tensor,
    y_cls: torch.Tensor,
    y_raw: torch.Tensor,
    pos_weight: torch.Tensor,
    cfg: TrainingConfig,
):
    huber = nn.functional.huber_loss(reg_pred_scaled, y_reg_scaled, reduction="none", delta=1.0)
    rw = regression_sample_weights(y_raw, cfg.regression_event_weights)
    reg_loss = (huber * rw).sum() / rw.sum().clamp_min(1.0)

    cls_loss = nn.functional.binary_cross_entropy_with_logits(
        cls_logits,
        y_cls,
        pos_weight=pos_weight.to(cls_logits.device),
    )
    total = reg_loss + cfg.classification_loss_weight * cls_loss
    return total, reg_loss.detach(), cls_loss.detach()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    y_train_raw: np.ndarray,
    cfg: TrainingConfig,
    model_path: Path,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    seed_everything(cfg.seed)
    device = device or choose_device()
    model = model.to(device)
    pos_weight = compute_pos_weights(y_train_raw).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = math.inf
    patience_left = cfg.patience
    rows = []
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_total = train_reg = train_cls = 0.0
        n_train = 0

        for xb, yb_scaled, yb_cls, yb_raw, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb_scaled = yb_scaled.to(device, non_blocking=True)
            yb_cls = yb_cls.to(device, non_blocking=True)
            yb_raw = yb_raw.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preg, pcl = model(xb)
                loss, rloss, closs = _loss_fn(preg, pcl, yb_scaled, yb_cls, yb_raw, pos_weight, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            b = len(xb)
            train_total += float(loss.detach()) * b
            train_reg += float(rloss) * b
            train_cls += float(closs) * b
            n_train += b

        model.eval()
        val_total = val_reg = val_cls = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb_scaled, yb_cls, yb_raw, _ in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb_scaled = yb_scaled.to(device, non_blocking=True)
                yb_cls = yb_cls.to(device, non_blocking=True)
                yb_raw = yb_raw.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    preg, pcl = model(xb)
                    loss, rloss, closs = _loss_fn(preg, pcl, yb_scaled, yb_cls, yb_raw, pos_weight, cfg)
                b = len(xb)
                val_total += float(loss) * b
                val_reg += float(rloss) * b
                val_cls += float(closs) * b
                n_val += b

        tr = train_total / max(n_train, 1)
        va = val_total / max(n_val, 1)
        scheduler.step(va)
        rows.append({
            "epoch": epoch,
            "train_loss": tr,
            "train_reg_loss": train_reg / max(n_train, 1),
            "train_cls_loss": train_cls / max(n_train, 1),
            "valid_loss": va,
            "valid_reg_loss": val_reg / max(n_val, 1),
            "valid_cls_loss": val_cls / max(n_val, 1),
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"[{cfg.model_name}] epoch {epoch:02d} | train={tr:.4f} | valid={va:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if va < best_val - 1e-4:
            best_val = va
            patience_left = cfg.patience
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, model_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}; best validation loss={best_val:.4f}")
                break

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return pd.DataFrame(rows)


def predict_model(
    model: nn.Module,
    loader: DataLoader,
    preprocessor: TemporalPreprocessor,
    index: pd.DatetimeIndex,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    device = device or choose_device()
    model = model.to(device)
    model.eval()
    reg_scaled, logits_all, y_raw_all, positions = [], [], [], []

    with torch.no_grad():
        for xb, _, _, yb_raw, pos in loader:
            xb = xb.to(device, non_blocking=True)
            preg, pcl = model(xb)
            reg_scaled.append(preg.detach().cpu().numpy())
            logits_all.append(pcl.detach().cpu().numpy())
            y_raw_all.append(yb_raw.numpy())
            positions.append(pos.numpy())

    reg_scaled = np.concatenate(reg_scaled)
    logits = np.concatenate(logits_all)
    y_raw = np.concatenate(y_raw_all)
    positions = np.concatenate(positions).astype(int)
    pred = preprocessor.inverse_target(reg_scaled)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    # Enforce nested probabilities at reporting/inference time:
    # P(>=3m) <= P(>=2m) <= P(>=1m).
    probs[:, 1] = np.minimum(probs[:, 1], probs[:, 0])
    probs[:, 2] = np.minimum(probs[:, 2], probs[:, 1])

    return pd.DataFrame({
        "prediction_time": index[positions],
        "position": positions,
        "target_delta_120m": y_raw,
        "pred_delta_120m": pred,
        "p_ge_1m": probs[:, 0],
        "p_ge_2m": probs[:, 1],
        "p_ge_3m": probs[:, 2],
    })


def fbeta_from_pr(precision: np.ndarray, recall: np.ndarray, beta: float = 2.0) -> np.ndarray:
    b2 = beta**2
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)


def choose_probability_threshold(y_true: np.ndarray, probability: np.ndarray, beta: float = 2.0) -> Dict[str, float]:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), probability)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision": np.nan, "recall": np.nan, "fbeta": np.nan}
    f = fbeta_from_pr(precision[:-1], recall[:-1], beta=beta)
    i = int(np.nanargmax(f))
    return {
        "threshold": float(thresholds[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "fbeta": float(f[i]),
    }


def point_regression_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    e = np.asarray(pred) - np.asarray(y)
    ae = np.abs(e)
    return {
        "n": int(len(y)),
        "MAE_cm": float(100 * ae.mean()),
        "RMSE_cm": float(100 * np.sqrt(np.mean(e**2))),
        "bias_cm": float(100 * e.mean()),
        "median_abs_error_cm": float(100 * np.median(ae)),
        "max_abs_error_cm": float(100 * ae.max()),
        "within_20cm_pct": float(100 * np.mean(ae <= 0.2)),
        "within_50cm_pct": float(100 * np.mean(ae <= 0.5)),
    }


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(probability) >= threshold
    tp = int(np.sum(y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and precision + recall else np.nan
    f2 = 5 * precision * recall / (4 * precision + recall) if precision == precision and recall == recall and (4 * precision + recall) else np.nan
    return {
        "real_events": int(y_true.sum()),
        "predicted_alerts": int(y_pred.sum()),
        "true_positives": tp,
        "missed_events": fn,
        "false_alerts": fp,
        "true_negatives": tn,
        "recall_pct": float(100 * recall) if recall == recall else np.nan,
        "precision_pct": float(100 * precision) if precision == precision else np.nan,
        "F1": float(f1) if f1 == f1 else np.nan,
        "F2": float(f2) if f2 == f2 else np.nan,
    }


def severity_metrics(y: np.ndarray, pred: np.ndarray, model_name: str) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "pred": pred})
    bins = [-np.inf, 0, 0.25, 0.5, 1.0, 2.0, 3.0, np.inf]
    labels = ["<=0", "0-0.25", "0.25-0.5", "0.5-1", "1-2", "2-3", ">3"]
    df["band"] = pd.cut(df["y"], bins=bins, labels=labels)
    rows = []
    for band in labels:
        z = df[df["band"] == band]
        if z.empty:
            continue
        e = z["pred"] - z["y"]
        rows.append({
            "model": model_name,
            "severity_band": band,
            "n": len(z),
            "real_mean_m": z["y"].mean(),
            "pred_mean_m": z["pred"].mean(),
            "MAE_cm": 100 * e.abs().mean(),
            "RMSE_cm": 100 * np.sqrt(np.mean(e**2)),
            "bias_cm": 100 * e.mean(),
            "underprediction_pct": 100 * np.mean(e < 0),
            "max_abs_error_cm": 100 * e.abs().max(),
        })
    return pd.DataFrame(rows)


def conformal_intervals(calibration_y: np.ndarray, calibration_pred: np.ndarray, test_pred: np.ndarray, coverages=(0.80, 0.90, 0.95)):
    residual = np.abs(np.asarray(calibration_y) - np.asarray(calibration_pred))
    rows = []
    intervals = {}
    n = len(residual)
    for coverage in coverages:
        # finite-sample split conformal quantile
        q_level = min(1.0, math.ceil((n + 1) * coverage) / n)
        q = float(np.quantile(residual, q_level, method="higher"))
        intervals[coverage] = (test_pred - q, test_pred + q, q)
        rows.append({"nominal_coverage": coverage, "half_width_m": q, "full_width_cm": 200 * q})
    return intervals, pd.DataFrame(rows)


def save_model_definition(path: Path, model_kind: str, input_dim: int, **kwargs) -> None:
    payload = {"model_kind": model_kind, "input_dim": input_dim, **kwargs}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model_from_definition(definition_path: Path, checkpoint_path: Path, device: Optional[torch.device] = None):
    d = json.loads(definition_path.read_text(encoding="utf-8"))
    kind = d["model_kind"]
    input_dim = int(d["input_dim"])
    if kind == "TCN":
        model = TCNMultiTask(
            input_dim=input_dim,
            channels=tuple(d.get("channels", [64, 64, 96, 96])),
            kernel_size=int(d.get("kernel_size", 3)),
            dropout=float(d.get("dropout", 0.15)),
        )
    elif kind == "GRU":
        model = GRUMultiTask(
            input_dim=input_dim,
            hidden_dim=int(d.get("hidden_dim", 96)),
            num_layers=int(d.get("num_layers", 2)),
            dropout=float(d.get("dropout", 0.15)),
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    device = device or choose_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


def fit_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    y_train_raw: np.ndarray,
    cfg: TrainingConfig,
    epochs: int,
    model_path: Path,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Fit a final model for a fixed number of epochs after hyperparameters are frozen.

    This is intended for the blind-test stage: the epoch count must come from
    pre-2025 development (e.g. the best validation epoch from notebook 07b).
    """
    seed_everything(cfg.seed)
    device = device or choose_device()
    model = model.to(device)
    pos_weight = compute_pos_weights(y_train_raw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rows=[]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(epochs)+1):
        model.train(); total=reg_total=cls_total=0.0; n=0
        for xb,yb_scaled,yb_cls,yb_raw,_ in train_loader:
            xb=xb.to(device,non_blocking=True); yb_scaled=yb_scaled.to(device,non_blocking=True)
            yb_cls=yb_cls.to(device,non_blocking=True); yb_raw=yb_raw.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preg,pcl=model(xb)
                loss,rloss,closs=_loss_fn(preg,pcl,yb_scaled,yb_cls,yb_raw,pos_weight,cfg)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
            b=len(xb); total+=float(loss.detach())*b; reg_total+=float(rloss)*b; cls_total+=float(closs)*b; n+=b
        row={"epoch":epoch,"train_loss":total/max(n,1),"train_reg_loss":reg_total/max(n,1),"train_cls_loss":cls_total/max(n,1)}
        rows.append(row)
        print(f"[{cfg.model_name} final] epoch {epoch:02d}/{epochs} | loss={row['train_loss']:.4f}")

    torch.save({"state_dict":model.state_dict(),"config":asdict(cfg),"fixed_epochs":int(epochs)},model_path)
    return pd.DataFrame(rows)
