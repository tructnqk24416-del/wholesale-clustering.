"""
Lab 3 – k-Means và chọn k trên Mall Customers (slide 27–30)
Mục tiêu: quét k, chọn k có căn cứ, gắn nhãn cụm và profile.

Câu hỏi nộp bài: thêm Age vào X thì k tốt nhất và persona thay đổi ra sao?
Chạy:  python src/lab3_kmeans_mall.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import load_mall, save_csv, save_fig

mall = load_mall()

# ----- Code như trên slide
X_mall = mall[["Income", "Spending"]]
rows = []
for k in range(2, 11):
    pipe = make_pipeline(StandardScaler(),
               KMeans(n_clusters=k, n_init=10, random_state=42))
    labels = pipe.fit_predict(X_mall)
    Xs = pipe[:-1].transform(X_mall)
    rows.append([k, pipe[-1].inertia_, silhouette_score(Xs, labels)])
scores = pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])
print(scores.round(3))

best = make_pipeline(StandardScaler(),
           KMeans(n_clusters=5, n_init=10, random_state=42))
mall["cluster"] = best.fit_predict(X_mall)
profile = mall.groupby("cluster").agg(n=("Age", "size"), age=("Age", "mean"),
    income=("Income", "mean"), spending=("Spending", "mean")).round(1)
profile["female_ratio"] = mall.groupby("cluster")["Gender"].apply(
    lambda s: (s == "Female").mean()).round(2)
print(profile)

# Tâm cụm ở đơn vị gốc (k$, điểm)
centers = best[0].inverse_transform(best[-1].cluster_centers_)
print(pd.DataFrame(centers, columns=["Income", "Spending"]).round(1))

# ----- Biểu đồ elbow + silhouette (bài nộp)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(scores["k"], scores["inertia"], "o-"); axs[0].set_title("Elbow: inertia theo k")
axs[1].plot(scores["k"], scores["silhouette"], "o-", color="tab:orange")
axs[1].set_title("Silhouette theo k")
for ax in axs:
    ax.set_xlabel("Số cụm k")
save_fig(fig, "lab3_elbow_silhouette.png"); plt.show()

fig = plt.figure(figsize=(6, 4.5))
plt.scatter(mall["Income"], mall["Spending"], c=mall["cluster"], cmap="tab10", s=25)
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=200, c="black")
plt.xlabel("Income (k$)"); plt.ylabel("Spending"); plt.title("k-Means k = 5")
save_fig(fig, "lab3_kmeans5.png"); plt.show()

save_csv(scores.round(4), "mall_k_scores.csv")
save_csv(mall, "mall_clusters.csv")
save_csv(profile, "mall_cluster_profile.csv", index=True)

# ----- Gợi ý cho câu hỏi nộp bài: thêm Age vào X
X3 = mall[["Age", "Income", "Spending"]]
for k in range(2, 11):
    p = make_pipeline(StandardScaler(), KMeans(n_clusters=k, n_init=10, random_state=42))
    lab = p.fit_predict(X3)
    print("Age+Income+Spending | k =", k,
          "silhouette =", round(silhouette_score(p[:-1].transform(X3), lab), 3))
