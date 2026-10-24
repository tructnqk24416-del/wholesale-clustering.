"""
Lab 1 – Tiền xử lý Wholesale bằng Pipeline (slide 17)
Mục tiêu: log-transform + chuẩn hóa trong một bước, tái sử dụng được.

Câu hỏi nộp bài: thay StandardScaler bằng RobustScaler thì phân phối thay đổi thế nào?
Chạy:  python src/lab1_tien_xu_ly.py
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler

from common import load_wholesale, FEATURES

ws = load_wholesale()

X_ws = ws[FEATURES]
y_channel = ws["Channel"]          # chỉ dùng để kiểm chứng

prep = make_pipeline(
    FunctionTransformer(np.log1p),   # nén đuôi phải
    StandardScaler())                # mean 0, std 1
X_ws_scaled = prep.fit_transform(X_ws)

print(X_ws.skew().round(2).values)
# [ 2.56  4.05  3.59  5.91  3.63 11.15]
print(pd.DataFrame(X_ws_scaled).skew().round(2).values)
# [-1.58 -0.22 -0.67 -0.35 -0.24 -1.09]
print(X_ws_scaled.mean(0).round(2), X_ws_scaled.std(0).round(2))

# --- Gợi ý cho câu hỏi nộp bài: so sánh với RobustScaler
robust = make_pipeline(FunctionTransformer(np.log1p), RobustScaler())
X_robust = robust.fit_transform(X_ws)
compare = pd.DataFrame({
    "std_StandardScaler": X_ws_scaled.std(0),
    "std_RobustScaler": X_robust.std(0),
    "median_RobustScaler": np.median(X_robust, axis=0),
}, index=FEATURES).round(2)
print(compare)
