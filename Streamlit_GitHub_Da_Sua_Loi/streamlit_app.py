"""Customer segmentation explorer based on the UEL unsupervised-learning lab.

Run: streamlit run streamlit_app.py
Deploy entrypoint: streamlit_app.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from optimizer import (
    WHOLESALE, Config, baseline_config, config_from_row, evaluate,
    experiments, prep_apply, recommended_config, reference_space,
    segment_actions, segment_profile,
)

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="UEL | Clustering Optimizer", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1350px; padding-top: 1.6rem;}
[data-testid="stMetric"] {background: #f5f7fa; border-radius: 11px; padding: 12px 15px;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Segmentation · Clustering Optimizer")
st.caption("Phát triển từ sample code Unsupervised Learning UEL · Khám phá dữ liệu, tìm cấu hình tốt và đề xuất action")


@st.cache_data(show_spinner=False)
def builtin(which):
    p = BASE / "data" / ("wholesale_customers.csv" if which == "Wholesale" else "Mall_Customers.csv")
    df = pd.read_csv(p)
    return df.rename(columns={"Genre": "Gender", "Annual Income (k$)": "Income", "Spending Score (1-100)": "Spending"})


@st.cache_data(show_spinner=False)
def compute_sweep(data, feature_groups, transforms, scalers, algos, ks, pcas, clips, share, validation_col):
    y = data[validation_col] if validation_col else None
    return experiments(data, dict(feature_groups), list(transforms), list(scalers),
                       list(algos), list(ks), list(pcas), list(clips), share, validation=y)


with st.sidebar:
    st.header("1 · Dữ liệu")
    source = st.radio("Nguồn", ["Wholesale", "Mall Customers", "Tải CSV riêng"], index=0)
    if source == "Tải CSV riêng":
        uploaded = st.file_uploader("Tệp CSV (UTF-8)", type="csv")
        if not uploaded:
            st.info("Hãy tải CSV để bắt đầu.")
            st.stop()
        try:
            original_df = pd.read_csv(uploaded)
        except Exception as ex:
            st.error(f"Không đọc được CSV: {ex}")
            st.stop()
    else:
        original_df = builtin("Wholesale" if source == "Wholesale" else "Mall")
    numeric_cols = original_df.select_dtypes(include="number").columns.tolist()
    ignore_patterns = ("id", "code", "segment", "cluster", "label", "target")
    defaults = [c for c in numeric_cols
                if c not in ("Channel", "Region") and not any(p in c.lower() for p in ignore_patterns)]
    if source == "Mall Customers":
        defaults = ["Income", "Spending"]
    st.header("2 · Đặc trưng")
    features = st.multiselect("Cột đưa vào phân cụm", numeric_cols, default=defaults)
    validation_cols = [c for c in numeric_cols if c not in features]
    validation = st.selectbox("Nhãn kiểm chứng (không dùng để train)",
                               ["Không dùng"] + validation_cols,
                               index=(["Không dùng"] + validation_cols).index("Channel")
                               if "Channel" in validation_cols else 0)
    validation = None if validation == "Không dùng" else validation
    st.caption("Không đưa CustomerID, Channel, Region vào mô hình mặc định. Channel chỉ để kiểm chứng.")
    if st.button("Đặt lại kết quả tìm kiếm", use_container_width=True):
        st.session_state.pop("sweep", None)
        st.session_state.pop("manual_config", None)
        st.rerun()

if len(features) < 2:
    st.warning("Chọn tối thiểu 2 cột số để phân cụm.")
    st.stop()
if len(original_df) > 2000:
    st.error("Bản demo giới hạn 2.000 dòng để tránh quá tải Streamlit Community Cloud. Hãy dùng dữ liệu mẫu đại diện.")
    st.stop()
if len(original_df) < 6:
    st.error("Cần tối thiểu 6 dòng dữ liệu.")
    st.stop()
valid_mask = original_df[features].notna().all(axis=1) & np.isfinite(original_df[features].to_numpy(dtype=float)).all(axis=1)
df = original_df.loc[valid_mask].copy().reset_index(drop=True)
if len(df) < 6:
    st.error("Không đủ dòng dữ liệu hợp lệ sau khi loại dòng thiếu giá trị số.")
    st.stop()
if len(df) != len(original_df):
    st.warning(f"Đã loại {len(original_df)-len(df)} dòng thiếu/không hợp lệ; còn {len(df)} dòng.")
if len(df[features].drop_duplicates()) < 3:
    st.error("Dữ liệu quá ít giá trị phân biệt để phân cụm.")
    st.stop()

all_nonnegative = bool((df[features] >= 0).all().all())
starting_cfg = recommended_config(features) if all_nonnegative else Config(tuple(features), "raw", "Standard", 0, "KMeans", 2, 100)
ref = reference_space(df, features)
try:
    start, start_labels, start_model, start_x, start_pre = evaluate(
        df, starting_cfg, reference=ref, validation=df[validation] if validation else None)
    if all_nonnegative:
        base, _, _, _, _ = evaluate(df, baseline_config(features), reference=ref,
                                     validation=df[validation] if validation else None)
    else:
        base = None
except Exception as exc:
    st.error(f"Chưa thể phân cụm bộ dữ liệu: {exc}")
    st.stop()

selected_cfg = starting_cfg
search_table = st.session_state.get("sweep")
context = (source, tuple(features), len(df), tuple(df[features].sum().round(5)), validation)
if st.session_state.get("sweep_context") != context:
    search_table = None
    st.session_state.pop("sweep", None)
    st.session_state.pop("manual_config", None)

st.info("**Đọc điểm đúng cách:** Silhouette trong không gian huấn luyện và Silhouette tham chiếu (cố định trên toàn bộ cột đã chọn) là hai phép đo khác nhau. Không chọn mô hình chỉ dựa vào một điểm số.")

tab_overview, tab_tune, tab_segments, tab_predict, tab_method = st.tabs(
    ["🔎 Tổng quan", "🧪 Tối ưu & so sánh", "📈 Phân khúc & action", "🧾 Dự đoán khách mới", "ℹ️ Phương pháp & Deploy"])

with tab_overview:
    st.subheader("Tổng quan dữ liệu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số khách hàng", f"{len(df):,}")
    c2.metric("Số đặc trưng", len(features))
    c3.metric("Silhouette gợi ý", f"{start['silhouette']:.4f}")
    c4.metric("Cụm nhỏ nhất", f"{start['smallest_cluster_pct']:.1f}%")
    if source == "Wholesale" and features == WHOLESALE:
        st.success("Gợi ý sẵn từ thử nghiệm: sqrt → MinMaxScaler → PCA 2 chiều → K-Means (k=2).")
    if base:
        st.write(f"**Mốc gốc cùng bộ đặc trưng:** log1p + StandardScaler + K-Means k=2 (không PCA): **{base['silhouette']:.4f}**. "
                 f"Cấu hình gợi ý đạt **{start['silhouette']:.4f}** (được đo trên không gian PCA nếu bật PCA).")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Thống kê các đặc trưng**")
        st.dataframe(df[features].describe().round(2), use_container_width=True)
    with c2:
        sample_features = features[:min(6, len(features))]
        fig = px.box(df[sample_features].melt(var_name="Đặc trưng", value_name="Giá trị"),
                     x="Đặc trưng", y="Giá trị", points=False, title="Phân phối & outlier (giá trị gốc)")
        st.plotly_chart(fig, use_container_width=True)

with tab_tune:
    st.subheader("Tìm cấu hình theo nhiều yếu tố")
    st.write("Em có thể thay đổi thuật toán, số cụm, phương pháp biến đổi và chuẩn hóa, PCA, cắt outlier, hoặc chọn nhóm đặc trưng. "
             "Bảng xếp hạng chỉ ưu tiên các cấu hình thỏa điều kiện quy mô cụm tối thiểu.")
    with st.form("auto_form"):
        aa, bb, cc = st.columns(3)
        with aa:
            transforms = st.multiselect("Biến đổi", ["sqrt", "log1p", "raw"] if all_nonnegative else ["raw"],
                                         default=["sqrt", "log1p"] if all_nonnegative else ["raw"])
            scalers = st.multiselect("Chuẩn hóa", ["MinMax", "Standard", "Robust"],
                                     default=["MinMax", "Standard", "Robust"])
        with bb:
            algorithms = st.multiselect("Thuật toán", ["KMeans", "Ward", "GMM"],
                                        default=["KMeans", "Ward", "GMM"])
            k_hi = st.slider("Số cụm tối đa k (từ 2)", 2, min(8, len(df)-1), 6)
        with cc:
            dims = [0] + [v for v in (2, 3) if v < len(features)]
            dims_default = [v for v in (0, 2) if v in dims]
            pcas = st.multiselect("Số chiều PCA (0 = không PCA)", dims, default=dims_default)
            min_share = st.slider("Tỷ trọng cụm nhỏ nhất ≥ (%)", 5, 40, 20, 5)
        clip_vals = st.multiselect("Giới hạn cực trị theo phân vị (100 = không cắt)",
                                   [100, 99, 95, 90], default=[100])
        includes_subset = st.checkbox("Cho phép chọn riêng nhóm Milk–Grocery–Detergents_Paper", value=False,
                                     disabled=not set(["Milk", "Grocery", "Detergents_Paper"]).issubset(features))
        submitted = st.form_submit_button("🔬 Chạy tối ưu tự động", use_container_width=True, type="primary")
    if submitted:
        if not (transforms and scalers and algorithms and pcas and clip_vals):
            st.error("Cần chọn ít nhất một giá trị trong mỗi nhóm tham số.")
        else:
            sets = {"Toàn bộ cột đã chọn": tuple(features)}
            if includes_subset:
                sets["Retail (3 đặc trưng)"] = ("Milk", "Grocery", "Detergents_Paper")
            try:
                with st.spinner("Đang chạy nhiều cấu hình và đánh giá tất cả cụm..."):
                    search_table = compute_sweep(
                        df, tuple(sets.items()), tuple(transforms), tuple(scalers), tuple(algorithms),
                        tuple(range(2, k_hi + 1)), tuple(pcas), tuple(clip_vals), min_share, validation)
                st.session_state["sweep"] = search_table
                st.session_state["sweep_context"] = context
                st.session_state.pop("manual_config", None)
            except Exception as exc:
                st.error(f"Không chạy được vòng tối ưu: {exc}")
    if search_table is not None and not search_table.empty:
        valid_search = search_table[search_table["eligible"]].reset_index(drop=True)
        st.caption(f"Đã thử {len(search_table)} cấu hình hợp lệ về mặt tính toán; "
                   f"{len(valid_search)} cấu hình thỏa tỷ trọng cụm tối thiểu.")
        if valid_search.empty:
            st.warning("Chưa có cấu hình thỏa điều kiện. Hãy giảm tỷ trọng cụm tối thiểu hoặc mở rộng thuật toán/k.")
        else:
            display_cols = ["algorithm", "transform", "scaler", "pca", "k", "feature_group",
                            "silhouette", "reference_silhouette", "ari_vs_channel",
                            "smallest_cluster_pct", "pca_variance_pct", "cluster_counts"]
            st.dataframe(valid_search[display_cols].head(30).round(4), use_container_width=True, hide_index=True)
            ix = st.selectbox("Chọn cấu hình từ bảng xếp hạng để phân tích", list(range(min(30, len(valid_search)))),
                              format_func=lambda i: (f"#{i+1} · {valid_search.iloc[i]['algorithm']} · "
                                                     f"{valid_search.iloc[i]['transform']} / {valid_search.iloc[i]['scaler']} · "
                                                     f"PCA {int(valid_search.iloc[i]['pca']) or 'Off'} · "
                                                     f"k={int(valid_search.iloc[i]['k'])} · "
                                                     f"Sil={valid_search.iloc[i]['silhouette']:.4f}"))
            selected_cfg = config_from_row(valid_search.iloc[ix])
            if st.button("Áp dụng cấu hình đang chọn vào biểu đồ và action"):
                st.session_state.pop("manual_config", None)
            st.download_button("⬇️ Tải đầy đủ bảng thử nghiệm (.csv)",
                               search_table.to_csv(index=False).encode("utf-8-sig"),
                               "benchmark_all_configs.csv", "text/csv")
            st.caption("Nếu chọn nhóm 3 thay vì 6 đặc trưng, điểm Silhouette không đánh giá cùng một không gian. "
                       "Cột reference_silhouette dùng không gian cố định trên các đặc trưng chính để đối chiếu; "
                       "ARI là kiểm chứng phụ nếu có nhãn.")

    with st.expander("⚙️ Cấu hình thủ công / tinh chỉnh thêm"):
        with st.form("manual_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                mt = st.selectbox("Biến đổi", ["sqrt", "log1p", "raw"] if all_nonnegative else ["raw"])
                ms = st.selectbox("Scaler", ["MinMax", "Standard", "Robust"])
            with col2:
                ma = st.selectbox("Mô hình", ["KMeans", "Ward", "GMM"])
                mk = st.slider("k", 2, min(8, len(df)-1), 2, key="manual_k")
            with col3:
                mp = st.selectbox("PCA", [0] + [n for n in (2, 3, 4) if n < len(features)])
                mc = st.select_slider("Clip outliers", options=[90, 95, 99, 100], value=100)
            apply_manual = st.form_submit_button("Áp dụng mô hình thủ công")
        if apply_manual:
            st.session_state["manual_config"] = Config(tuple(features), mt, ms, mp, ma, mk, mc)
        if st.session_state.get("manual_config"):
            if st.button("Bỏ tùy chỉnh thủ công"):
                st.session_state.pop("manual_config", None)
                st.rerun()

if st.session_state.get("manual_config") is not None:
    selected_cfg = st.session_state["manual_config"]

try:
    metrics, labels, model, X_fit, fitted_prep = evaluate(
        df, selected_cfg, reference=ref, validation=df[validation] if validation else None)
except Exception as exc:
    st.error(f"Cấu hình đang chọn không hợp lệ: {exc}")
    st.stop()

with tab_segments:
    st.subheader("Kết quả mô hình đang chọn")
    st.code(f"{selected_cfg.algorithm} (k={selected_cfg.k}) | {selected_cfg.transform} → {selected_cfg.scaler} "
            f"→ PCA={selected_cfg.pca or 'Off'} | clip={selected_cfg.clip_percent}%")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette (không gian train)", f"{metrics['silhouette']:.4f}")
    m2.metric("Silhouette (tham chiếu cố định)", f"{metrics['reference_silhouette']:.4f}")
    m3.metric("Davies–Bouldin ↓", f"{metrics['davies_bouldin']:.4f}")
    m4.metric("Cụm nhỏ nhất", f"{metrics['smallest_cluster_pct']:.1f}%")
    if validation:
        st.caption(f"ARI so với `{validation}`: {metrics['ari_vs_channel']:.4f} (chỉ kiểm chứng sau phân cụm, không dùng để chọn tham số).")
    if selected_cfg.pca:
        st.caption(f"PCA giữ lại {metrics['pca_variance_pct']:.1f}% phương sai sau chuẩn hóa. "
                   "Để so sánh công bằng về mặt khoảng cách, xem thêm điểm tham chiếu cố định.")
    if metrics["smallest_cluster_pct"] < 10:
        st.warning("Một cụm rất nhỏ: có thể thuật toán đang tách nhóm ngoại lệ thay vì phân khúc khách hàng điển hình.")
    plot = X_fit if X_fit.shape[1] == 2 else PCA(n_components=2, random_state=0).fit_transform(X_fit)
    viz = pd.DataFrame({"Trục 1": plot[:, 0], "Trục 2": plot[:, 1],
                        "Segment": [f"Nhóm {int(v)}" for v in labels]})
    if validation:
        viz[validation] = df[validation].astype(str).values
    v1, v2 = st.columns([1.35, 1])
    with v1:
        fig = px.scatter(viz, x="Trục 1", y="Trục 2", color="Segment",
                         hover_data=[validation] if validation else [],
                         title="Trực quan hóa phân cụm (2D)", opacity=0.75)
        fig.update_traces(marker={"size": 8})
        st.plotly_chart(fig, use_container_width=True)
        if X_fit.shape[1] != 2:
            st.caption("Đây là phép chiếu PCA *chỉ để hiển thị*. Silhouette phía trên được tính trên không gian huấn luyện thực tế.")
    with v2:
        counts = pd.Series(labels).value_counts().sort_index().rename_axis("Segment").reset_index(name="Số khách")
        fig2 = px.bar(counts, x="Segment", y="Số khách", text="Số khách", title="Kích thước từng cụm")
        st.plotly_chart(fig2, use_container_width=True)
    profile = segment_profile(df, selected_cfg.features, labels)
    st.write("**Chân dung từng phân khúc (median, dữ liệu gốc)**")
    st.dataframe(profile.round(1), use_container_width=True)
    if validation:
        st.write("**Bảng đối chiếu nhãn kiểm chứng**")
        st.dataframe(pd.crosstab(pd.Series(labels, name="Segment"), df[validation]), use_container_width=True)
    st.subheader("💡 Đề xuất action theo phân khúc")
    actions = segment_actions(df, selected_cfg.features, labels)
    st.dataframe(actions, use_container_width=True, hide_index=True)
    st.caption("Action là giả thuyết kinh doanh gợi ý từ đặc trưng chi tiêu, không phải bằng chứng tác động hoặc dự báo doanh thu. "
               "Nên kiểm chứng với đội kinh doanh hoặc A/B test trước khi triển khai.")
    labeled = df.copy()
    labeled["Segment"] = labels
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("⬇️ Dữ liệu đã gắn cụm", labeled.to_csv(index=False).encode("utf-8-sig"),
                           "customer_segments.csv", "text/csv", use_container_width=True)
    with dl2:
        st.download_button("⬇️ Báo cáo phân khúc", actions.to_csv(index=False).encode("utf-8-sig"),
                           "action_recommendations.csv", "text/csv", use_container_width=True)
    with dl3:
        st.download_button("⬇️ Cấu hình mô hình", json.dumps({**selected_cfg.__dict__, "features": list(selected_cfg.features)},
                                                             indent=2, ensure_ascii=False).encode("utf-8"),
                           "selected_model_config.json", "application/json", use_container_width=True)

with tab_predict:
    st.subheader("Thử phân khúc cho khách hàng mới")
    if selected_cfg.algorithm == "Ward":
        st.warning("Agglomerative (Ward) không có phương thức predict khách mới trong scikit-learn. "
                   "Hãy chọn K-Means hoặc GMM trong tab Tối ưu để dùng tính năng này.")
    else:
        st.caption("Điểm mới được biến đổi bằng **chính** cấu hình và bộ scaler/PCA đã fit từ dữ liệu huấn luyện.")
        with st.form("prediction"):
            columns = st.columns(min(3, len(selected_cfg.features)))
            values = {}
            for i, feature in enumerate(selected_cfg.features):
                with columns[i % len(columns)]:
                    values[feature] = st.number_input(feature, value=float(df[feature].median()),
                                                       step=1.0, format="%.2f")
            do_predict = st.form_submit_button("Dự đoán phân khúc")
        if do_predict:
            new_data = pd.DataFrame([values])
            if selected_cfg.transform in ("sqrt", "log1p") and new_data.lt(0).any().any():
                st.error("Dữ liệu có số âm không thể áp dụng phép sqrt/log1p.")
            else:
                new_vector = prep_apply(new_data, selected_cfg, fitted_prep)
                segment = int(model.predict(new_vector)[0])
                st.success(f"Khách hàng mới thuộc **Nhóm {segment}** theo mô hình đang chọn.")
                suggested = actions.loc[actions["Segment"] == segment, "Action đề xuất"].iloc[0]
                st.write(f"Action gợi ý: {suggested}")

with tab_method:
    st.subheader("Phương pháp và giới hạn")
    st.markdown("""
- **Tiền xử lý:** giữ giá trị gốc, `sqrt` hoặc `log1p`; chuẩn hóa Standard/MinMax/Robust; có tùy chọn cắt outlier theo phân vị.
- **Giảm chiều:** PCA là tùy chọn (2, 3, 4 chiều tùy số đặc trưng), thay đổi cả mô hình lẫn thước đo khoảng cách.
- **Phân cụm:** K-Means, Agglomerative Ward, Gaussian Mixture Model (GMM), so sánh nhiều `k`.
- **Chất lượng:** Silhouette ↑, Davies–Bouldin ↓, quy mô cụm, phân vị PCA và Silhouette tham chiếu cố định. ARI kiểm chứng sau huấn luyện nếu có nhãn.
- **Cảnh báo:** tối đa Silhouette không đồng nghĩa phân khúc phù hợp nhất. Kết quả thăm dò trên cùng dữ liệu có thiên lệch lựa chọn mô hình; cần kiểm chứng ngoài mẫu, độ ổn định, và chuyên gia nghiệp vụ.
- **Dữ liệu Wholesale:** `Channel` và `Region` không dùng để huấn luyện theo mặc định. `Channel` chỉ là nhãn để kiểm chứng kết quả.
    """)
    st.subheader("Triển khai Streamlit Community Cloud")
    st.code("""# Chạy local từ thư mục ZIP đã giải nén
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
""", language="bash")
    st.write("Đưa thư mục mã nguồn này lên GitHub rồi vào [share.streamlit.io](https://share.streamlit.io). "
             "Tạo app, chọn repo/branch và đặt **Main file path = `streamlit_app.py`** (Python 3.12 được khuyến nghị).")
    st.caption("Nguồn dữ liệu: Mall Customers và Wholesale như trong bài Lab UEL. Ứng dụng chạy với CSV nội bộ và không cần API key.")
