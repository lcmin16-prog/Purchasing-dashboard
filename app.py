import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import ASSETS_DIR, COLORS, INVENTORY_FILE
from modules.data_loader import load_inventory_data

st.set_page_config(page_title="장기재고현황 대시보드", layout="wide")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/app.log", encoding="utf-8"), logging.StreamHandler()],
)


def load_css(path: Path) -> None:
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def format_krw(value: float) -> str:
    if value >= 1e8:
        return f"{value / 1e8:.1f}억원"
    if value >= 1e4:
        return f"{value / 1e4:.0f}만원"
    return f"{value:,.0f}원"


def sum_amount(df: pd.DataFrame, cols: list[str]) -> float:
    available = [col for col in cols if col in df.columns]
    if not available:
        return 0.0
    return float(df[available].sum().sum())

def sum_amount_by_department(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    available = [col for col in cols if col in df.columns]
    if not available or "담당부서" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("담당부서")[available].sum().sum(axis=1)

@st.cache_data(show_spinner=False)
def get_inventory_data(source) -> pd.DataFrame:
    return load_inventory_data(source)


load_css(ASSETS_DIR / "styles.css")

st.markdown("<div class='dashboard-title'>🏢 장기재고현황 대시보드</div>", unsafe_allow_html=True)

base_date = datetime.now().strftime("%Y-%m-%d")
update_time = "오전 9:00"
st.markdown(
    f"<div class='dashboard-meta'>📅 데이터 기준일: {base_date} | 🔄 최종 업데이트: {update_time}</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("엑셀 업로드", type=["xlsx"])
data_source_label = None

if uploaded_file is None and not INVENTORY_FILE.exists():
    st.error("입력 파일을 찾을 수 없습니다: data/장기재고현황.xlsx")
    st.stop()

try:
    if uploaded_file is not None:
        df = get_inventory_data(uploaded_file.getvalue())
        data_source_label = f"📂 업로드 파일: {uploaded_file.name}"
    else:
        df = get_inventory_data(INVENTORY_FILE)
        data_source_label = f"📂 기본 파일: {INVENTORY_FILE.name}"
except Exception as exc:
    logging.exception("Failed to load inventory data")
    st.error(f"데이터 로딩 중 오류가 발생했습니다: {exc}")
    st.stop()

if data_source_label:
    st.caption(data_source_label)

amount_cols = [c for c in df.columns if c.endswith("_금액")]
current_amount = df["M_금액"].sum() if "M_금액" in df.columns else df[amount_cols].sum().sum()
prev_amount = df["M-1_금액"].sum() if "M-1_금액" in df.columns else current_amount

long_term_amount = df["12개월+_금액"].sum() if "12개월+_금액" in df.columns else 0
long_term_items = int((df["12개월+_금액"] > 0).sum()) if "12개월+_금액" in df.columns else 0

long_term_rate = (long_term_amount / current_amount * 100) if current_amount else 0

kpi_html = """
<div class='kpi-row'>
  <div class='kpi-card'>
    <div class='kpi-header'>💰 총재고금액</div>
    <div class='kpi-value'>{total_amount}</div>
    <div class='kpi-delta' style='color:{delta_color};'>▲ {delta_pct:.1f}% (전월대비)</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-header'>⚠️ 장기재고</div>
    <div class='kpi-value'>{long_amount}</div>
    <div class='kpi-delta' style='color:{danger};'>🔴 {long_items}품목</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-header'>📊 장기재고율</div>
    <div class='kpi-value'>{long_rate:.1f}%</div>
    <div class='kpi-delta' style='color:{warning};'>⚠️ 목표대비 +3%pt</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-header'>📦 품목수</div>
    <div class='kpi-value'>{item_count}개</div>
    <div class='kpi-delta' style='color:{text_gray};'>장기: {long_items}개</div>
  </div>
</div>
"""

delta_pct = ((current_amount - prev_amount) / prev_amount * 100) if prev_amount else 0

total_amount_display = format_krw(current_amount)

st.markdown(
    kpi_html.format(
        total_amount=total_amount_display,
        delta_pct=delta_pct,
        delta_color=COLORS["danger_red"] if delta_pct >= 0 else COLORS["primary_blue"],
        long_amount=format_krw(long_term_amount),
        long_items=long_term_items,
        long_rate=long_term_rate,
        danger=COLORS["danger_red"],
        warning=COLORS["warning_orange"],
        item_count=len(df),
        text_gray=COLORS["text_gray"],
    ),
    unsafe_allow_html=True,
)

st.markdown("<div class='section-title'>🎯 담당부서별 장기재고 현황</div>", unsafe_allow_html=True)

left_col, right_col = st.columns([0.4, 0.6], gap="large")

with left_col:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    period_filter = st.radio(
        "기간 선택",
        ["전체기간", "3개월이상", "6개월이상", "12개월 이상"],
        horizontal=True,
    )

    all_amount_cols = [
        col
        for col in df.columns
        if col.endswith("_금액")
        and (col.startswith("M") or col.startswith("12개월이상"))
    ]
    cols_3_plus = [
        col
        for col in df.columns
        if col.startswith("M-")
        and col.endswith("_금액")
        and col not in {"M-1_금액", "M-2_금액"}
    ] + (["12개월이상_금액"] if "12개월이상_금액" in df.columns else [])
    cols_6_plus = [
        col
        for col in df.columns
        if col.startswith("M-")
        and col.endswith("_금액")
        and col not in {"M-1_금액", "M-2_금액", "M-3_금액", "M-4_금액", "M-5_금액"}
    ] + (["12개월이상_금액"] if "12개월이상_금액" in df.columns else [])
    cols_12_plus = ["12개월이상_금액"] if "12개월이상_금액" in df.columns else []

    period_cols_map = {
        "전체기간": all_amount_cols,
        "3개월이상": cols_3_plus,
        "6개월이상": cols_6_plus,
        "12개월 이상": cols_12_plus,
    }

    selected_cols = period_cols_map.get(period_filter, all_amount_cols)
    dept_series = sum_amount_by_department(df, selected_cols)
    dept_amount = (
        dept_series.reset_index().rename(columns={0: "선택기간_금액"})
        if not dept_series.empty
        else pd.DataFrame(columns=["담당부서", "선택기간_금액"])
    )
    if not dept_amount.empty:
        dept_amount = dept_amount.sort_values("선택기간_금액", ascending=False)
    if not dept_amount.empty:
        dept_amount["표시금액"] = dept_amount["선택기간_금액"].apply(format_krw)
        bar_fig = px.bar(
            dept_amount,
            x="선택기간_금액",
            y="담당부서",
            orientation="h",
            text="표시금액",
            color="선택기간_금액",
            color_continuous_scale=[COLORS["danger_red"], "#FFD93D"],
        )
        bar_fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.info("부서별 데이터가 없습니다.")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    bucket_definitions = [
        (
            "3개월미만",
            ["M_금액", "M-1_금액", "M-2_금액"],
            ["M_수량", "M-1_수량", "M-2_수량"],
        ),
        (
            "6개월미만",
            ["M-3_금액", "M-4_금액", "M-5_금액"],
            ["M-3_수량", "M-4_수량", "M-5_수량"],
        ),
        (
            "12개월미만",
            ["M-6_금액", "M-7_금액", "M-8_금액", "M-9_금액", "M-10_금액", "M-11_금액"],
            ["M-6_수량", "M-7_수량", "M-8_수량", "M-9_수량", "M-10_수량", "M-11_수량"],
        ),
        ("12개월 이상", ["12개월이상_금액"], ["12개월이상_수량"]),
    ]

    bucket_rows = []
    for label, amount_cols, qty_cols in bucket_definitions:
        bucket_rows.append(
            {
                "기간": label,
                "금액": sum_amount(df, amount_cols),
                "수량": sum_amount(df, qty_cols),
            }
        )
    bucket_df = pd.DataFrame(bucket_rows)

    if not bucket_df.empty:
        combo_fig = make_subplots(specs=[[{"secondary_y": True}]])
        combo_fig.add_trace(
            go.Bar(
                x=bucket_df["기간"],
                y=bucket_df["금액"],
                name="금액",
                marker_color=COLORS["primary_blue"],
            ),
            secondary_y=False,
        )
        combo_fig.add_trace(
            go.Scatter(
                x=bucket_df["기간"],
                y=bucket_df["수량"],
                name="수량",
                mode="lines+markers",
                line=dict(width=3, color=COLORS["warning_orange"]),
                marker=dict(size=8),
            ),
            secondary_y=True,
        )
        combo_fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        combo_fig.update_yaxes(title_text="금액", secondary_y=False)
        combo_fig.update_yaxes(title_text="수량", secondary_y=True)
        st.plotly_chart(combo_fig, use_container_width=True)
    else:
        st.info("월별 추이 데이터가 없습니다.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>📊 기간별 담당부서 누적</div>", unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)

if "담당부서" in df.columns:
    dept_bucket_defs = {
        "3개월미만": ["M_금액", "M-1_금액", "M-2_금액"],
        "6개월미만": ["M-3_금액", "M-4_금액", "M-5_금액"],
        "12개월미만": ["M-6_금액", "M-7_금액", "M-8_금액", "M-9_금액", "M-10_금액", "M-11_금액"],
        "12개월 이상": ["12개월이상_금액"],
    }

    dept_rows = []
    for label, cols in dept_bucket_defs.items():
        available = [col for col in cols if col in df.columns]
        if not available:
            continue
        grouped = df.groupby("담당부서")[available].sum().sum(axis=1)
        for dept, amount in grouped.items():
            dept_rows.append({"기간구분": label, "담당부서": dept, "금액": amount})

    dept_bucket_df = pd.DataFrame(dept_rows)
    if not dept_bucket_df.empty:
        period_order = ["3개월미만", "6개월미만", "12개월미만", "12개월 이상"]
        dept_bucket_df["기간구분"] = pd.Categorical(
            dept_bucket_df["기간구분"], categories=period_order, ordered=True
        )
        dept_bucket_df = dept_bucket_df.sort_values("기간구분")

        stack_fig = px.bar(
            dept_bucket_df,
            y="기간구분",
            x="금액",
            color="담당부서",
            orientation="h",
        )
        stack_fig.update_layout(
            barmode="stack",
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="담당부서",
        )
        st.plotly_chart(stack_fig, use_container_width=True)
    else:
        st.info("부서별 누적 데이터가 없습니다.")
else:
    st.info("담당부서 컬럼이 없습니다.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='action-buttons'>", unsafe_allow_html=True)

mail_disabled = True
excel_disabled = True

st.button("📧 메일통보", disabled=mail_disabled)
st.button("📥 엑셀다운", disabled=excel_disabled)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>🔍 담당부서별 상세 현황</div>", unsafe_allow_html=True)

filter_cols = st.columns([0.25, 0.25, 0.25, 0.25])

with filter_cols[0]:
    departments = ["전체"] + sorted(df["담당부서"].dropna().unique().tolist()) if "담당부서" in df.columns else ["전체"]
    selected_departments = st.multiselect("담당부서", departments, default=["전체"])

with filter_cols[1]:
    aging_options = ["전체", "3개월 미만", "3~6개월", "6~12개월", "12개월+"]
    selected_aging = st.selectbox("장기재고 기간", aging_options)

with filter_cols[2]:
    amount_max = float(df["12개월+_금액"].max()) if "12개월+_금액" in df.columns else 0
    amount_range = st.slider("장기재고 금액", 0.0, float(amount_max) if amount_max > 0 else 1.0, (0.0, float(amount_max) if amount_max > 0 else 1.0))

with filter_cols[3]:
    search_text = st.text_input("검색", value="")

filtered = df.copy()

if selected_departments and "전체" not in selected_departments and "담당부서" in filtered.columns:
    filtered = filtered[filtered["담당부서"].isin(selected_departments)]

if selected_aging != "전체" and "에이징" in filtered.columns:
    filtered = filtered[filtered["에이징"] == selected_aging]

if "12개월+_금액" in filtered.columns:
    filtered = filtered[(filtered["12개월+_금액"] >= amount_range[0]) & (filtered["12개월+_금액"] <= amount_range[1])]

if search_text and "품목명" in filtered.columns:
    filtered = filtered[filtered["품목명"].astype(str).str.contains(search_text, case=False, na=False)]

if "소진계획" not in filtered.columns:
    filtered["소진계획"] = "❌ 미등록"

columns_to_show = [
    col
    for col in [
        "담당부서",
        "품목코드",
        "품목명",
        "규격",
        "12개월+_금액",
        "M_수량",
        "에이징",
        "소진계획",
    ]
    if col in filtered.columns
]

st.dataframe(filtered[columns_to_show], use_container_width=True, height=420)

st.markdown("<div class='action-buttons'>", unsafe_allow_html=True)

st.button("✅ 선택 항목 소진계획 등록", disabled=True)
st.button("📧 선택 부서에 메일 발송", disabled=True)
st.button("📊 선택 항목 차트 보기", disabled=True)
st.button("📥 필터된 데이터 엑셀 다운로드", disabled=True)

st.markdown("</div>", unsafe_allow_html=True)
