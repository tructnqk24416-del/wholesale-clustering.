"""
Lab 5 – Pipeline phân khúc end-to-end trên Wholesale (slide 40–42)
Mục tiêu: tiền xử lý + PCA + k-Means trong một Pipeline, profile, benchmark, xuất kết quả.

Chạy:  python src/lab5_pipeline_end_to_end.py
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score)

from common import load_wholesale, wholesale_scaled, FEATURES, save_csv, RESULTS_DIR

ws = load_wholesale()
X_ws = ws[FEATURES]
y_channel = ws["Channel"]
X_ws_scaled = wholesale_scaled(ws)          # kết quả Lab 1

# ----- Code như trên slide
segmenter = make_pipeline(
    FunctionTransformer(np.log1p), StandardScaler(),
    PCA(n_components=2, random_state=0),
    KMeans(n_clusters=2, n_init=10, random_state=42))
ws["segment"] = segmenter.fit_predict(X_ws)

# Profile ở đơn vị gốc (median vì dữ liệu lệch)
print(ws.groupby("segment")[FEATURES].median().round(0))
print(pd.crosstab(ws["segment"], ws["Channel"]))   # kiểm chứng

# Benchmark trên cùng dữ liệu đã scale
models = {"KMeans": KMeans(2, n_init=10, random_state=42),
          "Agglomerative": AgglomerativeClustering(2),
          "GMM": GaussianMixture(2, random_state=42)}
rows = []
for name, m in models.items():
    lab = m.fit_predict(X_ws_scaled)
    print(name, round(silhouette_score(X_ws_scaled, lab), 3),
          round(davies_bouldin_score(X_ws_scaled, lab), 3),
          round(adjusted_rand_score(y_channel, lab), 3))
    rows.append([name, "/".join(map(str, np.bincount(lab))),
                 silhouette_score(X_ws_scaled, lab), davies_bouldin_score(X_ws_scaled, lab),
                 adjusted_rand_score(y_channel, lab), normalized_mutual_info_score(y_channel, lab)])

ws.to_csv(RESULTS_DIR / "wholesale_segments.csv", index=False)
print("[lab5] Đã lưu results/wholesale_segments.csv")

# ----- Lưu bảng kết quả (dùng cho slide 40 và báo cáo)
bench = pd.DataFrame(rows, columns=["model", "cluster_sizes", "silhouette",
                                    "davies_bouldin", "ARI_vs_Channel", "NMI_vs_Channel"])
print(bench.round(3))
save_csv(bench.round(4), "benchmark_wholesale.csv")
save_csv(ws.groupby("segment")[FEATURES].median().round(0), "wholesale_segment_profile.csv", index=True)

# Gán segment cho khách hàng mới bằng chính Pipeline đã fit
new_customers = pd.DataFrame([[3000, 9000, 15000, 800, 6000, 1200],
                              [20000, 1500, 2000, 5000, 300, 900]], columns=FEATURES)
print("Segment khách mới:", segmenter.predict(new_customers))
