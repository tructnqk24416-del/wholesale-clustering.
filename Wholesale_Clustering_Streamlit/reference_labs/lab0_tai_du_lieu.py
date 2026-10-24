"""
Lab 0 – Tải dữ liệu trực tiếp từ GitHub (slide 12)
Mục tiêu: đọc CSV bằng URL raw, đổi tên cột, kiểm tra nhanh chất lượng dữ liệu.

Chạy:  python src/lab0_tai_du_lieu.py
"""
import numpy as np
import pandas as pd

from common import DATA_DIR, MALL_FILE, WS_FILE

GH = "https://raw.githubusercontent.com"
MALL_URL = (GH + "/sharmaroshan/Clustering-of-Mall-Customers"
            "/master/Mall_Customers.csv")
WS_URL = (GH + "/udacity/machine-learning/master"
          "/projects/customer_segments/customers.csv")


def read_github_or_local(url, local_name):
    """Ưu tiên đọc từ GitHub như slide; nếu mạng bị chặn thì đọc file trong data/."""
    try:
        df = pd.read_csv(url)
        print(f"Đọc từ GitHub: {url}")
    except Exception as err:  # mất mạng, bị chặn proxy...
        print(f"Không đọc được GitHub ({type(err).__name__}) – dùng data/{local_name}")
        df = pd.read_csv(DATA_DIR / local_name)
    return df


mall = read_github_or_local(MALL_URL, MALL_FILE).rename(columns={
    "Genre": "Gender",
    "Annual Income (k$)": "Income",
    "Spending Score (1-100)": "Spending"})
ws = read_github_or_local(WS_URL, WS_FILE)

print(mall.shape, ws.shape)    # (200, 5) (440, 8)
print(mall.isna().sum().sum(), ws.isna().sum().sum())   # 0 0
print(mall.dtypes)
print(mall.describe().round(1))
print(ws.describe().round(0))
print(ws["Channel"].value_counts())   # 1 = Horeca (298), 2 = Retail (142)
