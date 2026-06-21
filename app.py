from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image


APP_NAME = "VisionReco"
MODEL_NAME = "MobileNetV3"

JP_LABELS: dict[str, str] = {
    "golden retriever": "ゴールデン・レトリバー",
    "labrador retriever": "ラブラドール・レトリバー",
    "flat-coated retriever": "フラットコーテッド・レトリバー",
    "chesapeake bay retriever": "チェサピーク・ベイ・レトリバー",
    "curly-coated retriever": "カーリーコーテッド・レトリバー",
    "tabby": "トラ猫",
    "tiger cat": "トラ猫",
    "egyptian cat": "エジプシャン・キャット",
    "persian cat": "ペルシャ猫",
    "computer keyboard": "キーボード",
    "keyboard": "キーボード",
    "computer mouse": "パソコン用マウス",
    "mouse": "マウス",
    "headphones": "ヘッドホン・イヤホン",
    "microphone": "マイク",
    "speaker": "スピーカー",
    "remote control": "リモコン",
    "coffee maker": "コーヒーメーカー",
    "espresso maker": "エスプレッソマシン",
    "toaster": "トースター",
    "cellular telephone": "スマートフォン",
    "sports car": "スポーツカー",
    "minivan": "ミニバン",
    "banana": "バナナ",
    "orange": "オレンジ",
    "lemon": "レモン",
    "pineapple": "パイナップル",
    "pizza": "ピザ",
    "cheeseburger": "チーズバーガー",
}

CATEGORY_RULES: list[tuple[str, set[str], str]] = [
    ("ペット", {"retriever", "dog", "cat", "kitten", "terrier", "hound", "spaniel", "poodle"}, "🐾"),
    ("食べ物", {"banana", "orange", "lemon", "pizza", "burger", "fruit", "apple", "bread", "food"}, "🍊"),
    ("家電", {"keyboard", "mouse", "headphones", "speaker", "microphone", "coffee", "toaster", "television", "laptop"}, "💻"),
    ("乗り物", {"car", "truck", "bus", "train", "bicycle", "motorcycle", "airliner", "ship"}, "🚗"),
    ("植物", {"flower", "tree", "plant", "mushroom", "daisy", "rose"}, "🌿"),
    ("文房具", {"pen", "pencil", "binder", "envelope", "book"}, "✏️"),
    ("日用品", {"bottle", "cup", "clock", "umbrella", "backpack", "chair", "table"}, "📦"),
]

DEMO_PREDICTIONS = [
    {"label": "golden retriever", "jp_label": "ゴールデン・レトリバー", "score": 95.4},
    {"label": "labrador retriever", "jp_label": "ラブラドール・レトリバー", "score": 3.1},
    {"label": "flat-coated retriever", "jp_label": "フラットコーテッド・レトリバー", "score": 0.7},
    {"label": "chesapeake bay retriever", "jp_label": "チェサピーク・ベイ・レトリバー", "score": 0.4},
    {"label": "curly-coated retriever", "jp_label": "カーリーコーテッド・レトリバー", "score": 0.2},
]


st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;900&display=swap');
        html, body, [class*="css"] { font-family: "Noto Sans JP", sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 92% 4%, rgba(121, 97, 255, .22), transparent 22rem),
                radial-gradient(circle at 4% 0%, rgba(66, 153, 255, .12), transparent 18rem),
                linear-gradient(180deg, #f7faff 0%, #ffffff 52%, #f8fbff 100%);
            color: #07143d;
        }
        .block-container { padding-top: 1.6rem; max-width: 1480px; }
        section[data-testid="stSidebar"] { background: #ffffffcc; border-right: 1px solid #e6ebf5; }
        h1, h2, h3 { color: #07143d; letter-spacing: 0; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.86);
            border: 1px solid #e3eaf7;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(30, 58, 138, .08);
        }
        .hero {
            text-align: center;
            padding: 1rem 0 1.35rem;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: .75rem;
            font-size: 1.55rem;
            font-weight: 900;
            color: #07143d;
            margin-bottom: .25rem;
        }
        .logo {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: inline-grid;
            place-items: center;
            background: linear-gradient(135deg, #2563eb, #28d2ff 45%, #8b5cf6);
            color: white;
            font-weight: 900;
            letter-spacing: -1px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: .42rem 1.15rem;
            border-radius: 999px;
            color: white;
            font-weight: 800;
            background: linear-gradient(90deg, #2563eb, #9a4dff);
            box-shadow: 0 10px 22px rgba(79, 70, 229, .22);
        }
        .hero h1 {
            font-size: clamp(2.6rem, 6vw, 4.8rem);
            line-height: 1.06;
            margin: .65rem 0 .5rem;
            font-weight: 900;
        }
        .hero p { color: #4b587c; font-size: 1.08rem; margin: 0; }
        .panel {
            background: rgba(255,255,255,.9);
            border: 1px solid #e1e8f5;
            border-radius: 14px;
            padding: 1.3rem;
            box-shadow: 0 14px 30px rgba(30, 58, 138, .09);
            height: 100%;
        }
        .panel-title {
            display: flex;
            align-items: center;
            gap: .7rem;
            font-weight: 900;
            font-size: 1.18rem;
            margin-bottom: 1rem;
        }
        .iconbox {
            width: 42px;
            height: 42px;
            display: inline-grid;
            place-items: center;
            border-radius: 10px;
            background: #edf4ff;
            color: #2563eb;
            font-weight: 900;
        }
        .result-card {
            border: 1px solid #e2e9f4;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            background: #fff;
        }
        .topline { display:flex; justify-content:space-between; align-items:center; gap:1rem; }
        .rank {
            min-width: 44px;
            min-height: 44px;
            display: inline-grid;
            place-items: center;
            border-radius: 999px;
            background: #e9f1ff;
            color: #2563eb;
            font-size: 1.5rem;
            font-weight: 900;
        }
        .result-name { font-size: 1.45rem; font-weight: 900; }
        .score { color: #6d37ee; font-weight: 900; font-size: 2rem; }
        .bar-track { height: 9px; border-radius:999px; background:#edf1f7; overflow:hidden; }
        .bar-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#2563eb,#9a4dff); }
        .tag {
            display:inline-flex; align-items:center; gap:.25rem; padding:.26rem .75rem;
            border-radius:999px; background:#eaf2ff; color:#2563eb; font-weight:700;
            margin:.15rem .25rem .15rem 0;
        }
        .muted { color:#667392; }
        .small { font-size:.9rem; }
        .step {
            display: grid;
            grid-template-columns: 52px 58px 1fr;
            gap: .85rem;
            align-items: center;
            padding: .9rem;
            border: 1px solid #e2e9f4;
            border-radius: 12px;
            background: #fff;
            margin-bottom: .7rem;
        }
        .step-num {
            width: 44px; height: 44px; display:grid; place-items:center; border-radius:999px;
            background:#e9f1ff; color:#2563eb; font-weight:900; font-size:1.4rem;
        }
        .table-thumb img { border-radius: 8px; }
        .footer { text-align:center; color:#6b7898; padding: 1.8rem 0 .5rem; }
        div[data-testid="stFileUploader"] {
            border: 2px dashed #7aa2ff;
            border-radius: 14px;
            padding: 1rem;
            background: #f8fbff;
        }
        .stButton > button {
            border-radius: 10px;
            font-weight: 800;
            border: 1px solid #2f66ff;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #2563eb, #8b48ff);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("page", "トップ")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)


def logo() -> None:
    st.markdown('<div class="brand"><span class="logo">VR</span><span>VisionReco</span></div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    logo()
    st.markdown(
        f"""
        <div class="hero">
            <span class="pill">PyTorch × Streamlit</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="AIモデルを読み込んでいます...")
def load_model() -> tuple[Any, Any, list[str]] | None:
    try:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.eval()
        return model, weights.transforms(), weights.meta["categories"]
    except Exception:
        return None


def translate_label(label: str) -> str:
    normalized = label.lower().replace("_", " ")
    return JP_LABELS.get(normalized, normalized)


def infer_category(label: str) -> tuple[str, str]:
    normalized = label.lower().replace("_", " ")
    for category, keywords, icon in CATEGORY_RULES:
        if any(keyword in normalized for keyword in keywords):
            return category, icon
    return "その他", "🔎"


def predict(image: Image.Image) -> tuple[list[dict[str, Any]], bool]:
    loaded = load_model()
    if loaded is None:
        return DEMO_PREDICTIONS, False

    import torch

    model, preprocess, categories = loaded
    rgb_image = image.convert("RGB")
    batch = preprocess(rgb_image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.nn.functional.softmax(model(batch)[0], dim=0)
    scores, indices = torch.topk(probabilities, 5)
    predictions = []
    for score, index in zip(scores, indices):
        label = categories[index.item()].replace("_", " ")
        predictions.append(
            {
                "label": label,
                "jp_label": translate_label(label),
                "score": round(float(score.item()) * 100, 1),
            }
        )
    return predictions, True


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def add_history(image: Image.Image, filename: str, predictions: list[dict[str, Any]], model_available: bool) -> None:
    top = predictions[0]
    category, icon = infer_category(top["label"])
    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "filename": filename,
        "datetime": datetime.now(),
        "top_label": top["jp_label"],
        "top_label_en": top["label"],
        "score": top["score"],
        "category": category,
        "icon": icon,
        "image": image_to_bytes(image.copy()),
        "predictions": predictions,
        "model_available": model_available,
    }
    st.session_state.history.insert(0, entry)
    st.session_state.last_result = entry


def confidence_chart(predictions: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(predictions)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = ["#2563eb", "#8358f5", "#ff8a1f", "#48bf72", "#ef5350"]
    ax.barh(df["jp_label"][::-1], df["score"][::-1], color=colors[: len(df)][::-1])
    ax.set_xlim(0, 100)
    ax.set_xlabel("信頼度（%）")
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)


def category_charts(history: list[dict[str, Any]]) -> None:
    if not history:
        st.info("まだ判定履歴がありません。画像をアップロードすると分析グラフが表示されます。")
        return
    df = pd.DataFrame(history)
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    counts["ratio"] = counts["count"] / counts["count"].sum() * 100

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown('<div class="panel-title"><span class="iconbox">◔</span>カテゴリ別割合</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.pie(counts["count"], labels=counts["category"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="panel-title"><span class="iconbox">▥</span>カテゴリ別件数</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(counts["category"], counts["count"], color=["#2563eb", "#8358f5", "#ff8a1f", "#48bf72", "#ef5350"])
        ax.set_ylabel("件数")
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)


def prediction_comment(entry: dict[str, Any]) -> str:
    label = entry["top_label"]
    category = entry["category"]
    score = entry["score"]
    if score >= 80:
        certainty = "可能性が高いです"
    elif score >= 50:
        certainty = "可能性があります"
    else:
        certainty = "候補の一つとして判定されています"
    return f"この画像は、{category}カテゴリの「{label}」である{certainty}。上位候補と信頼度を合わせて確認すると、AIがどこに注目して判定したかを比較できます。"


def upload_area() -> None:
    uploaded_file = st.file_uploader("画像ファイルを選択", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1:
        sample_clicked = st.button("サンプル画像で試す", use_container_width=True, type="primary")
    with col2:
        clear_clicked = st.button("履歴をクリア", use_container_width=True)

    if clear_clicked:
        st.session_state.history = []
        st.session_state.last_result = None
        st.success("判定履歴をクリアしました。")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with st.spinner("AIが画像を分析しています..."):
            predictions, model_available = predict(image)
        add_history(image, uploaded_file.name, predictions, model_available)
        st.session_state.page = "画像認識結果"
        st.rerun()

    if sample_clicked:
        image = make_sample_image()
        add_history(image, "golden_retriever_sample.png", DEMO_PREDICTIONS, False)
        st.session_state.page = "画像認識結果"
        st.rerun()


def make_sample_image() -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=160)
    ax.set_facecolor("#d6efe0")
    ax.add_patch(plt.Circle((0.5, 0.48), 0.28, color="#d49a43"))
    ax.add_patch(plt.Circle((0.34, 0.5), 0.12, color="#b97932"))
    ax.add_patch(plt.Circle((0.66, 0.5), 0.12, color="#b97932"))
    ax.add_patch(plt.Circle((0.42, 0.56), 0.035, color="#1f2937"))
    ax.add_patch(plt.Circle((0.58, 0.56), 0.035, color="#1f2937"))
    ax.add_patch(plt.Circle((0.5, 0.46), 0.05, color="#111827"))
    ax.add_patch(plt.Circle((0.5, 0.33), 0.07, color="#ef7890"))
    ax.text(0.5, 0.12, "sample image", ha="center", color="#4b587c", weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def home_page() -> None:
    hero("AI画像認識アプリ", "画像をアップロードすると、AIが写っている物を判定し、候補と信頼度を分かりやすく表示します。")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="iconbox">⇧</span>画像をアップロード</div>', unsafe_allow_html=True)
        st.write("画像をアップロードして、AIによる画像認識を体験してみましょう。")
        upload_area()
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="iconbox">✦</span>このアプリでできること</div>', unsafe_allow_html=True)
        steps = [
            ("1", "⇧", "画像をアップロード", "手元の画像やサンプル画像を使って簡単に判定できます。"),
            ("2", "🧠", "AIが判定", "PyTorchの学習済みモデルで、画像に写っている物を解析します。"),
            ("3", "▥", "結果を確認", "上位5候補と信頼度をグラフで見やすく表示します。"),
            ("4", "↺", "履歴と分析", "過去の判定結果を保存し、カテゴリ傾向を分析できます。"),
        ]
        for num, icon, title, body in steps:
            st.markdown(
                f'<div class="step"><div class="step-num">{num}</div><div class="iconbox">{icon}</div><div><b>{title}</b><br><span class="muted small">{body}</span></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("主な画面")
    cards = st.columns(4)
    for col, icon, title, body, page in zip(
        cards,
        ["▣", "▧", "◷", "◔"],
        ["トップ画面", "画像認識結果画面", "判定履歴画面", "分析画面"],
        ["アプリの概要と使い方を確認できます。", "アップロード画像の判定結果と上位候補を確認できます。", "過去の判定履歴を一覧で確認できます。", "カテゴリ別の傾向や統計をグラフで分析できます。"],
        ["トップ", "画像認識結果", "判定履歴", "分析"],
    ):
        with col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-title"><span class="iconbox">{icon}</span>{title}</div>', unsafe_allow_html=True)
            st.write(body)
            if st.button("開く", key=f"open-{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def results_page() -> None:
    hero("画像認識結果", "アップロードした画像のAI判定結果を表示します。")
    entry = st.session_state.last_result
    if entry is None:
        st.info("まだ判定結果がありません。トップ画面から画像をアップロードしてください。")
        if st.button("トップへ戻る", type="primary"):
            st.session_state.page = "トップ"
            st.rerun()
        return

    image = Image.open(io.BytesIO(entry["image"]))
    predictions = entry["predictions"]
    top = predictions[0]
    category, icon = infer_category(top["label"])

    left, right = st.columns([.85, 2.1], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="iconbox">⇧</span>アップロード画像</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.write(f"📄 {entry['filename']}")
        st.write(f"🖼️ {image.width} × {image.height} px")
        st.write(f"🕒 {entry['datetime'].strftime('%Y/%m/%d %H:%M:%S')}")
        if st.button("別の画像をアップロード", use_container_width=True):
            st.session_state.page = "トップ"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="iconbox">✦</span>AI判定コメント</div>', unsafe_allow_html=True)
        st.write(prediction_comment(entry))
        if not entry["model_available"]:
            st.caption("PyTorchモデルを読み込めない環境ではデモ判定を表示します。")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="iconbox">🏆</span>判定結果（トップ予測）</div>', unsafe_allow_html=True)
        image_col, result_col = st.columns([.8, 1.45], gap="large")
        with image_col:
            st.image(image, use_container_width=True)
        with result_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="topline">
                        <div style="display:flex;align-items:center;gap:.8rem;"><span class="rank">1</span><span class="result-name">{top['jp_label']}</span></div>
                        <span class="score">{top['score']:.1f}%</span>
                    </div>
                    <div class="muted small" style="margin:.25rem 0 1.2rem 3.3rem;">{top['label']}</div>
                    <b>信頼度（確率）</b>
                    <div class="bar-track" style="margin-top:.65rem;"><div class="bar-fill" style="width:{top['score']}%;"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### 上位5候補")
            for i, pred in enumerate(predictions, start=1):
                st.markdown(
                    f"""
                    <div style="display:grid;grid-template-columns:3rem 1fr 160px 4.6rem;gap:.5rem;align-items:center;margin:.4rem 0;">
                        <b>{i}位</b><span>{pred['jp_label']}</span>
                        <div class="bar-track"><div class="bar-fill" style="width:{pred['score']}%;"></div></div>
                        <b style="text-align:right;">{pred['score']:.1f}%</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.divider()
        st.markdown("#### 判定サマリー")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("判定結果", top["jp_label"])
        c2.metric("信頼度", f"{top['score']:.1f}%")
        c3.metric("推定カテゴリ", f"{icon} {category}")
        c4.metric("使用モデル", MODEL_NAME)
        st.markdown("#### 信頼度グラフ")
        confidence_chart(predictions)
        tags = [category, top["jp_label"], top["label"], MODEL_NAME]
        st.markdown("#### 予測カテゴリタグ")
        st.markdown("".join(f'<span class="tag">{tag}</span>' for tag in tags), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def history_page() -> None:
    hero("判定履歴", "これまでに判定した画像と結果の履歴を確認できます。")
    history = st.session_state.history
    total = len(history)
    avg_score = sum(item["score"] for item in history) / total if total else 0
    most_common = pd.Series([item["category"] for item in history]).mode().iloc[0] if total else "-"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総判定数", f"{total} 件")
    c2.metric("今日の判定数", f"{sum(item['datetime'].date() == datetime.now().date() for item in history)} 件")
    c3.metric("平均信頼度", f"{avg_score:.1f} %")
    c4.metric("最多カテゴリ", most_common)

    left, right = st.columns([.75, 2.4], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">フィルター</div>', unsafe_allow_html=True)
        categories = ["すべてのカテゴリ"] + sorted({item["category"] for item in history})
        selected = st.selectbox("カテゴリ", categories)
        min_score = st.slider("信頼度（最小値）", 0, 100, 0)
        keyword = st.text_input("検索", placeholder="ファイル名・カテゴリ・判定結果")
        st.markdown("</div>", unsafe_allow_html=True)

    filtered = []
    for item in history:
        text = f"{item['filename']} {item['category']} {item['top_label']} {item['top_label_en']}".lower()
        if selected != "すべてのカテゴリ" and item["category"] != selected:
            continue
        if item["score"] < min_score:
            continue
        if keyword and keyword.lower() not in text:
            continue
        filtered.append(item)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if not filtered:
            st.info("条件に一致する判定履歴がありません。")
        for item in filtered:
            cols = st.columns([.55, 1.35, 1.15, 1.45, .9, .9])
            cols[0].image(Image.open(io.BytesIO(item["image"])), use_container_width=True)
            cols[1].markdown(f"**{item['filename']}**  \n<span class='muted small'>PNG/JPG</span>", unsafe_allow_html=True)
            cols[2].write(item["datetime"].strftime("%Y/%m/%d\n%H:%M:%S"))
            cols[3].markdown(f"**{item['top_label']}**  \n<span class='tag'>{item['category']}</span>", unsafe_allow_html=True)
            cols[4].markdown(f"**{item['score']:.1f}%**")
            if cols[5].button("詳細を見る", key=f"detail-{item['id']}", use_container_width=True):
                st.session_state.last_result = item
                st.session_state.page = "画像認識結果"
                st.rerun()
            st.divider()
        st.caption(f"{len(filtered)} / {total} 件を表示")
        st.markdown("</div>", unsafe_allow_html=True)


def analytics_page() -> None:
    hero("分析画面", "判定履歴をもとに、どのカテゴリが多く判定されたかを円グラフと棒グラフで可視化します。")
    history = st.session_state.history
    total = len(history)
    avg_score = sum(item["score"] for item in history) / total if total else 0
    category_count = len({item["category"] for item in history}) if total else 0
    most_common = pd.Series([item["category"] for item in history]).mode().iloc[0] if total else "-"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総判定数", f"{total} 件")
    c2.metric("対象カテゴリ数", f"{category_count} カテゴリ")
    c3.metric("最多カテゴリ", most_common)
    c4.metric("平均信頼度", f"{avg_score:.1f} %")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    category_charts(history)
    st.markdown("</div>", unsafe_allow_html=True)

    if history:
        df = pd.DataFrame(history)
        counts = df["category"].value_counts().reset_index()
        counts.columns = ["カテゴリ", "件数"]
        counts["割合"] = (counts["件数"] / counts["件数"].sum() * 100).round(1).astype(str) + "%"
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span class="iconbox">✦</span>分析サマリー</div>', unsafe_allow_html=True)
        top_category = counts.iloc[0]["カテゴリ"]
        st.write(f"{top_category}カテゴリが最も多く、全体の傾向として身近な対象が多く判定されています。")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def about_page() -> None:
    hero("アプリ説明", "使用技術、画像認識の仕組み、注意事項をまとめています。")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("使用技術")
    st.table(
        pd.DataFrame(
            [
                ("開発言語", "Python"),
                ("Webアプリ", "Streamlit"),
                ("画像認識", "PyTorch"),
                ("学習済みモデル", "torchvision / MobileNetV3"),
                ("画像処理", "Pillow"),
                ("データ処理", "pandas"),
                ("グラフ", "matplotlib"),
            ],
            columns=["分類", "使用技術"],
        )
    )
    st.subheader("画像認識の流れ")
    st.write("アップロード画像をモデル入力用に変換し、MobileNetV3の学習済みモデルで推論します。出力された確率から上位5件を取り出し、日本語ラベル、信頼度、カテゴリとして表示します。")
    st.subheader("注意事項")
    st.write("学習済みモデルはImageNetのカテゴリをもとに判定します。写真の撮り方、明るさ、背景、対象物の種類によって結果が変わる場合があります。")
    st.markdown("</div>", unsafe_allow_html=True)


def sidebar_nav() -> None:
    with st.sidebar:
        logo()
        st.caption("画像から、AIで見つける")
        pages = ["トップ", "画像認識結果", "判定履歴", "分析", "アプリ説明"]
        selected = st.radio("画面", pages, index=pages.index(st.session_state.page), label_visibility="collapsed")
        if selected != st.session_state.page:
            st.session_state.page = selected
            st.rerun()
        st.divider()
        st.write("対応形式: JPG / JPEG / PNG")
        st.write(f"使用モデル: {MODEL_NAME}")


def main() -> None:
    init_state()
    inject_styles()
    sidebar_nav()
    pages = {
        "トップ": home_page,
        "画像認識結果": results_page,
        "判定履歴": history_page,
        "分析": analytics_page,
        "アプリ説明": about_page,
    }
    pages[st.session_state.page]()
    st.markdown('<div class="footer">VisionReco ｜ 画像から、AIで見つける</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
