# UEL – Customer Segmentation Optimizer (Streamlit)

Ứng dụng phân khúc khách hàng thực tế, phát triển từ **bộ source bài Lab Chương Unsupervised Learning – PGS.TS. Lê Hoành Sử, UEL**.

## Đã bổ sung gì so với sample code?

- Dashboard Streamlit tương tác: xem bảng, outlier, biểu đồ phân khúc 2D, phân bố số lượng cụm, chân dung cụm.
- **Auto optimization:** thử `KMeans`, `Agglomerative Ward`, `GaussianMixture`; `k=2..8`; `raw`, `sqrt`, `log1p`; `Standard`, `Robust`, `MinMax`; PCA / không PCA; có hoặc không cắt outlier theo phân vị.
- Lọc cấu hình có **cụm quá nhỏ** để tránh lựa chọn chỉ vì Silhouette rất cao khi tách outliers.
- Đo `Silhouette (train space)`, `Silhouette (fixed reference space)`, Davies–Bouldin, Calinski–Harabasz, tỷ trọng cụm, % explained variance PCA; ARI tùy chọn với Channel (chỉ đánh giá, **không** làm target huấn luyện/tối ưu).
- Đọc profile median mỗi cụm và gợi ý action kinh doanh cần kiểm chứng.
- Gán nhóm cho khách hàng mới bằng K-Means hoặc GMM dùng **đúng** scaler/PCA đã fit; không hỗ trợ predict Ward (không có predict trực tiếp).
- Nạp sẵn bộ dữ liệu Wholesale & Mall Customer, có thể upload CSV riêng (tối đa 2000 dòng trong bản demo).
- Tải xuống CSV phân cụm, bảng thử nghiệm, action, và JSON cấu hình.

## Kết quả đo lại trên Wholesale – 440 khách hàng, 6 đặc trưng

| Cấu hình | Silhouette trên không gian train | Silhouette tham chiếu cố định | ARI vs Channel | Cụm nhỏ nhất |
|---|---:|---:|---:|---:|
| Gốc: log1p + Standard + KMeans(2), **không PCA** | 0.2903 | 0.2903 | 0.5135 | 42.7% |
| √ + MinMax + KMeans(2), **không PCA** | 0.3747 | phụ thuộc nhãn | 0.5959 | 30.7% |
| **Gợi ý: √ + MinMax + PCA(2) + KMeans(2)** | **0.4685** | **0.2844** | **0.6260** | **32.0%** |
| raw + Standard + PCA(2) + KMeans(2) | 0.6362 | tùy nhãn | 0.2308 | 10.2% |

**Quan trọng:** Điểm Silhouette trên 2 chiều PCA và trên 6 chiều ban đầu KHÔNG phải thước đo giống hệt. Score tăng không bảo đảm mô hình tốt hơn dưới cùng định nghĩa khoảng cách. Đó là lý do ứng dụng cung cấp thêm điểm tham chiếu cố định, bảng phân bố cụm và ARI với Channel. Một cấu hình raw có thể có score >0.6 nhưng nhóm nhỏ chủ yếu là người chi tiêu cực cao, ARI kém: không nên mặc nhiên coi đó là phân khúc tốt nhất.

Phương án gợi ý: PCA giữ lại **78.35%** phương sai đã chuẩn hóa; 141 khách trong cụm 0 và 299 khách trong cụm 1. Các số và số hiệu nhãn cụm phụ thuộc sklearn/seed.

## Chạy trên Mac / máy tính cá nhân

Tại thư mục này:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Trên Windows, kích hoạt môi trường ảo bằng `.venv\Scripts\activate`.

## Triển khai lên Streamlit Community Cloud (miễn phí)

1. Tạo repository GitHub (public hoặc private tùy tài khoản/quyền), upload **toàn bộ nội dung thư mục này** vào root của repo. Cần có `streamlit_app.py`, `optimizer.py`, `requirements.txt`, `data/`.
2. Mở <https://share.streamlit.io/> và đăng nhập qua GitHub.
3. Chọn **Create app**, repo và branch. Trong **Main file path** điền `streamlit_app.py`.
4. Nếu cần chọn Python, ưu tiên **Python 3.12**. Sau đó chọn **Deploy**.
5. Khi muốn cập nhật mô hình, chỉnh sửa file trong repo rồi commit để Streamlit tự triển khai lại.

Không cần API key và không cần liên kết database. **File ZIP chưa tự publish lên cloud**; chủ repo phải thực hiện deploy trong tài khoản của mình.

## Các thư mục

```text
streamlit_app.py       Dashboard
optimizer.py           Tối ưu, train/predict, đánh giá và action
requirements.txt       Các thư viện cần cho deployment
.streamlit/config.toml Theme cơ bản
data/                  Wholesale + Mall
reference_labs/        6 sample Lab gốc, giữ lại để đối chiếu
reports/               Kết quả kiểm chứng và full grid 2.103 cấu hình
tests/                 Kiểm thử engine
```

## Giới hạn và thực hành tốt

- Đây là **phân cụm không giám sát**; Channel chỉ để đánh giá ngoài, không được dùng để xây cluster nếu mục đích là unsupervised.
- Lựa chọn nhiều cấu hình theo điểm Silhouette trên cùng bộ dữ liệu có nguy cơ selection bias; cần đánh giá độ ổn định, thời gian và tính phù hợp kinh doanh trước khi ra quyết định.
- Hệ thống gợi ý **action thử nghiệm** chứ không cam kết hiệu quả marketing.
- Nếu dữ liệu doanh nghiệp có thông tin nhạy cảm, không nên đưa CSV lên app công khai khi chưa có kiểm soát truy cập.
- Thuật toán Agglomerative Ward không tự suy luận khách hàng mới. Với yêu cầu dự đoán, hãy dùng KMeans/GMM.
- Source giảng dạy và dữ liệu trong `reference_labs/`, `data/` giữ mục đích giáo dục của gói ban đầu. Cần kiểm tra giấy phép nếu ứng dụng thương mại.
