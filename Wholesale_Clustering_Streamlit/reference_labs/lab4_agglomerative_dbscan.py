"""
Lab 4 – Agglomerative và DBSCAN (slide 33–37)
Mục tiêu: so sánh với k-Means, chọn eps có căn cứ, đọc điểm bất thường.

Câu hỏi nộp bài: tăng min_samples từ 5 lên 10 thì số điểm nhiễu thay đổi thế nào?
Chạy:  python src/lab4_agglomerative_dbscan.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

from common import (load_mall, load_wholesale, wholesale_scaled,
                    FEATURES, save_csv, save_fig)

mall = load_mall()
ws = load_wholesale()
X_mall = mall[["Income", "Spending"]]
X_ws_scaled = wholesale_scaled(ws)                       # kết quả Lab 1
mall["cluster"] = make_pipeline(                         # kết quả Lab 3
    StandardScaler(), KMeans(n_clusters=5, n_init=10, random_state=42)
).fit_predict(X_mall)

# ----- Code như trên slide
# 1) Agglomerative trên Mall – so với k-Means (Lab 3)
Xm = StandardScaler().fit_transform(X_mall)
agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
mall["agg"] = agg.fit_predict(Xm)
print(adjusted_rand_score(mall["cluster"], mall["agg"]))  # 0.94

# 2) k-distance để chọn eps cho Wholesale
nn = NearestNeighbors(n_neighbors=5).fit(X_ws_scaled)
dist, _ = nn.kneighbors(X_ws_scaled)
fig = plt.figure(figsize=(6, 4))
plt.plot(np.sort(dist[:, -1])); plt.axhline(2.0, ls="--", color="tab:orange")
plt.title("k-distance (k = 5)"); plt.ylabel("Khoảng cách tới láng giềng thứ 5")
save_fig(fig, "lab4_k_distance.png"); plt.show()

# 3) DBSCAN: nhãn -1 là nhiễu
db = DBSCAN(eps=2.0, min_samples=5).fit(X_ws_scaled)
ws["dbscan"] = db.labels_
print(ws["dbscan"].value_counts())     # 0: 428, -1: 12
outliers = ws[ws["dbscan"] == -1][FEATURES]
print(outliers)

# ----- Dendrogram Mall (slide 34)
fig, ax = plt.subplots(figsize=(8, 4.5))
dendrogram(linkage(Xm, method="ward"), truncate_mode="lastp", p=30, ax=ax)
ax.set_title("Dendrogram – Mall Customers (ward)")
save_fig(fig, "lab4_dendrogram.png"); plt.show()

# ----- Thử nhiều eps / min_samples (bài nộp)
rows = []
for eps in [1.5, 2.0, 2.5]:
    for ms in [5, 10]:
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_ws_scaled)
        rows.append([eps, ms, int((lab == -1).sum()), len(set(lab) - {-1})])
sweep = pd.DataFrame(rows, columns=["eps", "min_samples", "n_noise", "n_clusters"])
print(sweep)

save_csv(ws[ws["dbscan"] == -1], "wholesale_dbscan_outliers.csv", index=True)
save_csv(sweep, "dbscan_eps_sweep.csv")
