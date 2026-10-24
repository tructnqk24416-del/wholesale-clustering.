"""
Hàm dùng chung cho các bài Lab – Chương 3: Unsupervised Learning
Machine Learning · PGS.TS. Lê Hoành Sử · Khoa Hệ thống thông tin · UEL

Dữ liệu được đọc từ thư mục data/ đi kèm. Nếu không tìm thấy file cục bộ,
chương trình tự tải từ GitHub (đúng URL như trên slide Lab 0).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GH = "https://raw.githubusercontent.com"
MALL_URL = (GH + "/sharmaroshan/Clustering-of-Mall-Customers"
            "/master/Mall_Customers.csv")
WS_URL = (GH + "/udacity/machine-learning/master"
          "/projects/customer_segments/customers.csv")

MALL_FILE = "Mall_Customers.csv"
WS_FILE = "wholesale_customers.csv"

FEATURES = ["Fresh", "Milk", "Grocery", "Frozen",
            "Detergents_Paper", "Delicatessen"]


def _read_csv(local_name: str, url: str) -> pd.DataFrame:
    path = DATA_DIR / local_name
    if path.exists():
        return pd.read_csv(path)
    print(f"[common] Không thấy {path} – tải từ GitHub: {url}")
    return pd.read_csv(url)


def load_mall() -> pd.DataFrame:
    """Mall Customers (200 × 5), đã đổi tên cột như slide Lab 0."""
    return _read_csv(MALL_FILE, MALL_URL).rename(columns={
        "Genre": "Gender",
        "Annual Income (k$)": "Income",
        "Spending Score (1-100)": "Spending"})


def load_wholesale() -> pd.DataFrame:
    """Wholesale Customers (440 × 8): Channel, Region + 6 nhóm chi tiêu năm."""
    return _read_csv(WS_FILE, WS_URL)


def make_prep():
    """Pipeline tiền xử lý của Lab 1: log1p rồi StandardScaler."""
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler())


def wholesale_scaled(ws: pd.DataFrame) -> np.ndarray:
    """Trả về X_ws_scaled (dùng lại ở Lab 2, 4, 5)."""
    return make_prep().fit_transform(ws[FEATURES])


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = RESULTS_DIR / name
    df.to_csv(path, index=index)
    print(f"[common] Đã lưu {path.relative_to(ROOT)}")
    return path


def save_fig(fig, name: str) -> Path:
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[common] Đã lưu {path.relative_to(ROOT)}")
    return path
