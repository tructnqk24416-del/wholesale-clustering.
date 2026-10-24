"""
Lab 2 – PCA trên Wholesale (slide 23)
Mục tiêu: chọn số thành phần, đặt tên PC, trực quan dữ liệu 6 chiều trên mặt phẳng.

Câu hỏi nộp bài: bỏ bước log1p thì PC1 giải thích bao nhiêu % và loadings thay đổi ra sao?
Chạy:  python src/lab2_pca.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import load_wholesale, wholesale_scaled, FEATURES, save_csv, save_fig

ws = load_wholesale()
X_ws = ws[FEATURES]
y_channel = ws["Channel"]
X_ws_scaled = wholesale_scaled(ws)          # kết quả Lab 1

# ----- Code như trên slide
pca = PCA(random_state=0).fit(X_ws_scaled)
print(pca.explained_variance_ratio_.round(3))
print(pca.explained_variance_ratio_.cumsum().round(3))

loadings = pd.DataFrame(pca.components_[:2], columns=FEATURES,
                        index=["PC1", "PC2"]).round(2)
print(loadings)

Z = pca.transform(X_ws_scaled)[:, :2]
fig = plt.figure(figsize=(6, 4.5))
plt.scatter(Z[:, 0], Z[:, 1], c=y_channel, cmap="coolwarm", s=15)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA Wholesale – màu theo Channel")
save_fig(fig, "lab2_pca_scatter.png"); plt.show()

pca90 = PCA(n_components=0.9).fit(X_ws_scaled)
print(pca90.n_components_)     # 4 thành phần giữ >= 90%

# ----- Biểu đồ explained variance (bài nộp)
ratio = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(6, 4))
pcs = [f"PC{i + 1}" for i in range(len(ratio))]
ax.bar(pcs, ratio * 100, label="Từng PC")
ax.plot(pcs, ratio.cumsum() * 100, "o-", color="tab:orange", label="Tích lũy")
ax.set_ylabel("% phương sai"); ax.legend(); ax.set_title("Explained variance – Wholesale")
save_fig(fig, "lab2_explained_variance.png"); plt.show()

save_csv(pd.DataFrame({"PC": pcs, "ratio": ratio.round(4),
                       "cumulative": ratio.cumsum().round(4)}), "pca_explained_variance.csv")
save_csv(pd.DataFrame(pca.components_, columns=FEATURES, index=pcs).round(3),
         "pca_loadings.csv", index=True)

# ----- Gợi ý cho câu hỏi nộp bài: PCA khi bỏ log1p
pca_raw = PCA().fit(StandardScaler().fit_transform(X_ws))
print("Không log1p – explained variance:", pca_raw.explained_variance_ratio_.round(3))
print(pd.DataFrame(pca_raw.components_[:2], columns=FEATURES, index=["PC1", "PC2"]).round(2))
