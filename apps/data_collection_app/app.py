from __future__ import annotations

import streamlit as st

from image_quality import inspect_image
from storage import save_submission


DISEASES = {
    "Chưa xác định": "Unknown",
    "Lá khỏe mạnh": "Leaf_Healthy",
    "Bệnh đốm rong": "Leaf_Algal",
    "Bệnh cháy lá": "Leaf_Blight",
    "Bệnh thán thư": "Leaf_Colletotrichum",
    "Bệnh Phomopsis": "Leaf_Phomopsis",
    "Bệnh Rhizoctonia": "Leaf_Rhizoctonia",
}

st.set_page_config(
    page_title="Thu thập ảnh lá sầu riêng",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --leaf: #176b3a;
            --leaf-dark: #0d4d29;
            --mint: #eaf7ef;
            --line: #cfe7d8;
            --ink: #173c27;
        }

        .stApp {
            background:
                radial-gradient(circle at 95% 0%, #d8f3e2 0, transparent 30%),
                linear-gradient(180deg, #f7fcf8 0%, #edf8f1 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 680px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .hero {
            padding: 1.6rem;
            margin-bottom: 1.25rem;
            color: white;
            border-radius: 24px;
            background: linear-gradient(135deg, #0f5b31 0%, #269457 100%);
            box-shadow: 0 16px 38px rgba(20, 105, 57, 0.20);
        }

        .hero-badge {
            display: inline-block;
            padding: .35rem .7rem;
            margin-bottom: .7rem;
            border: 1px solid rgba(255,255,255,.35);
            border-radius: 999px;
            background: rgba(255,255,255,.13);
            font-size: .82rem;
            font-weight: 650;
        }

        .hero h1 {
            margin: 0 0 .45rem 0;
            color: white;
            font-size: clamp(1.65rem, 6vw, 2.25rem);
            line-height: 1.15;
        }

        .hero p {
            margin: 0;
            color: rgba(255,255,255,.9);
            line-height: 1.55;
        }

        .step {
            display: flex;
            align-items: center;
            gap: .7rem;
            margin: .15rem 0 .35rem;
        }

        .step-number {
            display: grid;
            width: 34px;
            height: 34px;
            flex: 0 0 34px;
            place-items: center;
            border-radius: 50%;
            color: white;
            background: var(--leaf);
            font-weight: 750;
        }

        .step-title {
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 750;
        }

        div[data-testid="stForm"] {
            padding: 1.3rem;
            border: 1px solid var(--line);
            border-radius: 22px;
            background: rgba(255,255,255,.94);
            box-shadow: 0 10px 30px rgba(35, 91, 56, .09);
        }

        div[data-testid="stCameraInput"] {
            padding: .65rem;
            border-radius: 16px;
            background: var(--mint);
        }

        div[data-testid="stSelectbox"] > label,
        div[data-testid="stCameraInput"] > label {
            font-weight: 650;
            color: var(--ink);
        }

        .stButton > button,
        .stFormSubmitButton > button {
            min-height: 3.15rem;
            border: 0;
            border-radius: 14px;
            color: white;
            background: linear-gradient(135deg, var(--leaf-dark), #23894f);
            font-size: 1rem;
            font-weight: 750;
            box-shadow: 0 8px 18px rgba(23, 107, 58, .20);
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            color: white;
            border: 0;
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(23, 107, 58, .27);
        }

        div[data-baseweb="select"] > div {
            min-height: 3rem;
            border-color: var(--line);
            border-radius: 13px;
            background: #fbfefc;
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        .privacy-note {
            margin-top: .9rem;
            text-align: center;
            color: #607568;
            font-size: .84rem;
        }

        @media (max-width: 640px) {
            .block-container { padding: 1rem .8rem 2rem; }
            .hero { padding: 1.35rem; border-radius: 20px; }
            div[data-testid="stForm"] { padding: 1rem; border-radius: 18px; }
        }
    </style>

    <section class="hero">
        <div class="hero-badge">🌱 DỮ LIỆU CỘNG ĐỒNG</div>
        <h1>Góp ảnh lá sầu riêng</h1>
        <p>Chỉ cần chụp một ảnh rõ nét và chọn tên bệnh. Mỗi ảnh của bạn sẽ giúp xây dựng bộ dữ liệu tốt hơn.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.form("submission_form", clear_on_submit=True):
    st.markdown(
        '<div class="step"><span class="step-number">1</span>'
        '<span class="step-title">Chụp ảnh lá</span></div>',
        unsafe_allow_html=True,
    )
    camera_image = st.camera_input(
        "Đặt lá vào giữa khung hình rồi nhấn nút chụp",
        help="Ảnh cần đủ sáng, rõ nét và nhìn thấy toàn bộ lá.",
    )
    st.caption("💡 Mẹo: chụp gần lá, tránh rung tay và ánh sáng quá chói.")

    st.markdown(
        '<div class="step"><span class="step-number">2</span>'
        '<span class="step-title">Chọn tên bệnh</span></div>',
        unsafe_allow_html=True,
    )
    disease_text = st.selectbox(
        "Tên bệnh trên lá",
        DISEASES,
        help="Nếu chưa biết chính xác, hãy chọn Chưa xác định.",
    )
    st.caption("Ứng dụng chỉ lưu lựa chọn của bạn, không tự dự đoán bệnh.")

    submitted = st.form_submit_button(
        "✓  Lưu ảnh vào bộ dữ liệu", type="primary", use_container_width=True
    )

st.markdown(
    '<p class="privacy-note">🔒 Ảnh chỉ được dùng để xây dựng bộ dữ liệu nghiên cứu.</p>',
    unsafe_allow_html=True,
)

if submitted:
    if camera_image is None:
        st.error("Bạn chưa chụp ảnh. Hãy mở camera và chụp một ảnh trước khi lưu.")
    else:
        result = inspect_image(camera_image.getvalue())
        if not result.acceptable or result.image is None:
            st.error(result.message)
        else:
            submission_id = save_submission(
                image=result.image,
                disease=DISEASES[disease_text],
            )
            st.success(
                f"Lưu ảnh thành công! Mã mẫu của bạn là {submission_id[:8]}."
            )
