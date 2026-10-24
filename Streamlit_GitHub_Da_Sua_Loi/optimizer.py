"""Experimentation engine; independent of Streamlit so it can be tested offline.

Never use Channel/Region as training inputs by default. Reference labels may be
used for *diagnostics only*, not for tuning or model selection.
"""
from dataclasses import asdict, dataclass
from itertools import product
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SEED = 42
WHOLESALE = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]


@dataclass(frozen=True)
class Config:
    features: tuple[str, ...]
    transform: str = "sqrt"
    scaler: str = "MinMax"
    pca: int = 2  # 0 = off
    algorithm: str = "KMeans"
    k: int = 2
    clip_percent: int = 100

    def key(self):
        return (self.features, self.transform, self.scaler, self.pca,
                self.algorithm, self.k, self.clip_percent)


def prep_fit(X: pd.DataFrame, cfg: Config):
    x = X.loc[:, list(cfg.features)].astype(float).to_numpy(copy=True)
    if not np.isfinite(x).all():
        raise ValueError("Đặc trưng cần là số hữu hạn, không chứa giá trị thiếu.")
    if cfg.transform in ("sqrt", "log1p") and (x < 0).any():
        raise ValueError("Dữ liệu có số âm: không thể dùng sqrt/log1p.")
    upper = np.quantile(x, cfg.clip_percent / 100, axis=0) if cfg.clip_percent < 100 else None
    if upper is not None:
        x = np.minimum(x, upper)
    if cfg.transform == "sqrt":
        x = np.sqrt(x)
    elif cfg.transform == "log1p":
        x = np.log1p(x)
    elif cfg.transform != "raw":
        raise ValueError("Biến đổi không hợp lệ")
    scaler = {"Standard": StandardScaler, "Robust": RobustScaler,
              "MinMax": MinMaxScaler}[cfg.scaler]()
    xx = scaler.fit_transform(x)
    pca = None
    if cfg.pca:
        if cfg.pca >= len(cfg.features):
            raise ValueError("PCA phải có ít chiều hơn số đặc trưng.")
        pca = PCA(n_components=cfg.pca, random_state=0)
        xx = pca.fit_transform(xx)
    return xx, (upper, scaler, pca)


def prep_apply(X: pd.DataFrame, cfg: Config, fitted):
    upper, scaler, pca = fitted
    x = X.loc[:, list(cfg.features)].astype(float).to_numpy(copy=True)
    if upper is not None:
        x = np.minimum(x, upper)
    if cfg.transform == "sqrt":
        x = np.sqrt(x)
    elif cfg.transform == "log1p":
        x = np.log1p(x)
    x = scaler.transform(x)
    return pca.transform(x) if pca is not None else x


def make_model(cfg: Config):
    if cfg.algorithm == "KMeans":
        return KMeans(n_clusters=cfg.k, n_init=10, random_state=SEED)
    if cfg.algorithm == "Ward":
        return AgglomerativeClustering(n_clusters=cfg.k, linkage="ward")
    if cfg.algorithm == "GMM":
        return GaussianMixture(n_components=cfg.k, covariance_type="full",
                               n_init=2, random_state=SEED)
    raise ValueError("Thuật toán không hợp lệ")


def reference_space(df: pd.DataFrame, features: list[str]):
    """Fixed, full-feature space for a common, secondary silhouette diagnostic."""
    x = df[features].to_numpy(dtype=float)
    if (x >= 0).all():
        x = np.log1p(x)
    return StandardScaler().fit_transform(x)


def evaluate(df: pd.DataFrame, cfg: Config, reference=None, validation=None):
    x, transformers = prep_fit(df, cfg)
    model = make_model(cfg)
    labels = model.fit_predict(x)
    count = np.bincount(labels, minlength=cfg.k)
    if len(np.unique(labels)) < 2 or (count == 0).any():
        raise ValueError("Mô hình không tạo đủ cụm.")
    row = {
        "algorithm": cfg.algorithm, "transform": cfg.transform,
        "scaler": cfg.scaler, "pca": cfg.pca,
        "k": cfg.k, "clip_percent": cfg.clip_percent,
        "feature_set": ", ".join(cfg.features),
        "n_features": len(cfg.features),
        "silhouette": silhouette_score(x, labels),
        "davies_bouldin": davies_bouldin_score(x, labels),
        "calinski_harabasz": calinski_harabasz_score(x, labels),
        "smallest_cluster_pct": 100 * count.min() / len(labels),
        "largest_cluster_pct": 100 * count.max() / len(labels),
        "cluster_counts": "/".join(map(str, count)),
        "pca_variance_pct": 100 * transformers[2].explained_variance_ratio_.sum()
        if transformers[2] is not None else 100.0,
        "reference_silhouette": silhouette_score(reference, labels) if reference is not None else np.nan,
        "ari_vs_channel": adjusted_rand_score(validation, labels) if validation is not None else np.nan,
    }
    return row, labels, model, x, transformers


def experiments(df: pd.DataFrame, feature_sets: dict[str, list[str]],
                transforms: list[str], scalers: list[str], algorithms: list[str],
                ks: list[int], pca_dims: list[int], clips: list[int],
                min_share_pct: float = 20., validation=None):
    if len(df) < 6:
        raise ValueError("Cần tối thiểu 6 dòng dữ liệu hợp lệ.")
    main_features = list(next(iter(feature_sets.values())))
    reference = reference_space(df, main_features)
    rows = []
    for (set_name, cols), trans, scaler_name, pca, algo, k, clip in product(
        feature_sets.items(), transforms, scalers, pca_dims, algorithms, ks, clips
    ):
        if pca and (pca >= len(cols) or pca >= len(df)):
            continue
        if trans in ("sqrt", "log1p") and (df.loc[:, list(cols)].lt(0)).to_numpy().any():
            continue
        cfg = Config(tuple(cols), trans, scaler_name, pca, algo, k, clip)
        try:
            row, *_ = evaluate(df, cfg, reference, validation)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        row["feature_group"] = set_name
        row["eligible"] = row["smallest_cluster_pct"] >= min_share_pct
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(["eligible", "silhouette"], ascending=[False, False]).reset_index(drop=True)


def config_from_row(row: pd.Series):
    return Config(tuple(str(row["feature_set"]).split(", ")),
                  str(row["transform"]), str(row["scaler"]), int(row["pca"]),
                  str(row["algorithm"]), int(row["k"]), int(row["clip_percent"]))


def baseline_config(features: list[str]):
    return Config(tuple(features), "log1p", "Standard", 0, "KMeans", 2, 100)


def recommended_config(features: list[str]):
    return Config(tuple(features), "sqrt", "MinMax", 2 if len(features) > 2 else 0,
                  "KMeans", 2, 100)


def segment_profile(df, features, labels):
    sub = df[list(features)].copy()
    sub["Segment"] = labels
    profile = sub.groupby("Segment")[list(features)].median()
    profile.insert(0, "Số khách", sub.groupby("Segment").size())
    profile.insert(1, "Tỷ trọng (%)", (profile["Số khách"] / len(df) * 100).round(1))
    return profile


def segment_actions(df, features, labels):
    med = df[list(features)].median().replace(0, np.nan)
    profile = segment_profile(df, features, labels)
    results = []
    for idx, row in profile.iterrows():
        ratios = (row[list(features)] / med).replace([np.inf, -np.inf], np.nan)
        higher = ratios[ratios >= 1.35].sort_values(ascending=False).index.tolist()[:2]
        lower = ratios[ratios <= .7].sort_values().index.tolist()[:2]
        high_txt = ", ".join(higher) if higher else "không có nhóm nổi bật rõ"
        low_txt = ", ".join(lower) if lower else "không có nhóm rất thấp"
        if higher:
            action = "Thử gói ưu đãi/cross-sell quanh " + ", ".join(higher) + "."
        else:
            action = "Khảo sát nhu cầu và thử chương trình chăm sóc riêng trước khi cá nhân hóa ưu đãi."
        if lower:
            action += " Kiểm tra cơ hội giới thiệu thêm " + ", ".join(lower) + "."
        results.append({"Segment": int(idx), "Quy mô": int(row["Số khách"]),
                        "Tỷ trọng (%)": float(row["Tỷ trọng (%)"]),
                        "Nhóm chi tiêu nổi bật": high_txt,
                        "Nhóm chi tiêu thấp": low_txt, "Action đề xuất": action})
    return pd.DataFrame(results)
