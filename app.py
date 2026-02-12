import logging
import re
import zipfile
import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    ASSETS_DIR,
    COLORS,
    DISPOSAL_UPLOAD_DIR,
    INVENTORY_FILE,
    LATEST_INVENTORY_FILE,
    LOGS_DIR,
)
from modules.data_loader import load_inventory_data

st.set_page_config(page_title="장기재고현황 대시보드", layout="wide")

log_handlers = [logging.StreamHandler()]
try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_handlers.insert(0, logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8"))
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
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


def calculate_inventory_total_amount(df: pd.DataFrame) -> float:
    amount_cols = [c for c in df.columns if c.endswith("_금액")]
    if "M_금액" in df.columns:
        return float(df["M_금액"].sum())
    if amount_cols:
        return float(df[amount_cols].sum().sum())
    return 0.0


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


def remove_header_like_rows(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [col for col in ["담당부서", "품목코드", "품목명"] if col in df.columns]
    if not key_cols:
        return df

    header_like = pd.Series(True, index=df.index)
    for col in key_cols:
        header_like &= df[col].astype(str).str.strip().eq(col)

    if header_like.any():
        return df.loc[~header_like].reset_index(drop=True)
    return df


def clean_department_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "담당부서" not in df.columns:
        return df

    dept_series = df["담당부서"].astype(str).str.strip()
    invalid_values = {"", "nan", "none", "null", "담당부서"}
    valid_mask = df["담당부서"].notna() & ~dept_series.str.lower().isin(invalid_values)

    cleaned = df.loc[valid_mask].copy()
    cleaned["담당부서"] = dept_series.loc[valid_mask]
    return cleaned.reset_index(drop=True)


def _safe_excel_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", str(name)).strip()
    return cleaned or "부서"


def get_github_config() -> dict:
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo = st.secrets.get("GITHUB_REPO", "")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    return {
        "enabled": bool(token and repo and branch),
        "token": token,
        "repo": repo,
        "branch": branch,
    }


DEFAULT_DEPARTMENTS = ["일본팀", "국내", "미주/유럽팀", "중국팀", "아시아/중동/CIS팀", "공통"]
INVENTORY_META_FILE = INVENTORY_FILE.parent / "inventory_upload_meta.json"
DISPOSAL_META_FILE = INVENTORY_FILE.parent / "disposal_upload_meta.json"


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _to_display_time(value: str) -> str:
    if not value:
        return "미업로드"
    return str(value)


def infer_department_from_filename(filename: str, known_departments: list[str]) -> str:
    stem = Path(filename).stem.strip()
    stem = re.sub(r"^소진계획[_\\-\\s]*", "", stem)
    stem = stem.strip()
    if stem in known_departments:
        return stem
    for dept in known_departments:
        if dept in stem or stem in dept:
            return dept
    return stem if stem else "미분류"


def _github_api_url(repo: str, path: str, branch: str, include_ref: bool = True) -> str:
    encoded_path = urlparse.quote(path, safe="/")
    base_url = f"https://api.github.com/repos/{repo}/contents/{encoded_path}"
    if not include_ref:
        return base_url
    encoded_branch = urlparse.quote(branch, safe="")
    return f"{base_url}?ref={encoded_branch}"


def _github_request(method: str, url: str, cfg: dict, payload: dict | None = None) -> dict | list | None:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url=url, method=method, data=body)
    req.add_header("Authorization", f"Bearer {cfg['token']}")
    req.add_header("Accept", "application/vnd.github+json")
    if body is not None:
        req.add_header("Content-Type", "application/json")

    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))
    except urlerror.HTTPError as exc:
        if exc.code == 404:
            return None
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GitHub API error ({exc.code}): {detail}") from exc


def github_get_file_bytes(cfg: dict, path: str) -> bytes | None:
    if not cfg.get("enabled"):
        return None
    url = _github_api_url(cfg["repo"], path, cfg["branch"], include_ref=True)
    result = _github_request("GET", url, cfg)
    if not result or not isinstance(result, dict) or result.get("type") != "file":
        return None
    content = str(result.get("content", "")).replace("\n", "")
    if not content:
        return None
    return base64.b64decode(content.encode("utf-8"))


def github_put_file_bytes(cfg: dict, path: str, data: bytes, message: str) -> None:
    if not cfg.get("enabled"):
        return

    read_url = _github_api_url(cfg["repo"], path, cfg["branch"], include_ref=True)
    write_url = _github_api_url(cfg["repo"], path, cfg["branch"], include_ref=False)
    existing = _github_request("GET", read_url, cfg)
    payload = {
        "message": message,
        "content": base64.b64encode(data).decode("utf-8"),
        "branch": cfg["branch"],
    }
    if isinstance(existing, dict) and existing.get("sha"):
        payload["sha"] = existing["sha"]
    _github_request("PUT", write_url, cfg, payload=payload)


def github_list_directory(cfg: dict, path: str) -> list[dict]:
    if not cfg.get("enabled"):
        return []
    url = _github_api_url(cfg["repo"], path, cfg["branch"], include_ref=True)
    result = _github_request("GET", url, cfg)
    if isinstance(result, list):
        return result
    return []


def load_upload_metadata(cfg: dict, local_path: Path, github_path: str, default_value: dict) -> dict:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("enabled"):
        data = github_get_file_bytes(cfg, github_path)
        if data:
            try:
                metadata = json.loads(data.decode("utf-8"))
                local_path.write_bytes(data)
                return metadata
            except Exception:
                logging.warning("Invalid metadata from github: %s", github_path)
    if local_path.exists():
        try:
            return json.loads(local_path.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("Invalid local metadata: %s", local_path.name)
    return default_value.copy()


def save_upload_metadata(cfg: dict, local_path: Path, github_path: str, metadata: dict, message: str) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
    local_path.write_bytes(payload)
    if cfg.get("enabled"):
        github_put_file_bytes(cfg, github_path, payload, message)


def sync_disposal_uploads_from_github(cfg: dict, upload_dir: Path) -> int:
    if not cfg.get("enabled"):
        return 0
    upload_dir.mkdir(parents=True, exist_ok=True)
    synced = 0
    for entry in github_list_directory(cfg, "data/disposal_uploads"):
        if entry.get("type") != "file":
            continue
        name = str(entry.get("name", ""))
        if not name.lower().endswith(".xlsx"):
            continue
        file_path = str(entry.get("path", ""))
        if not file_path:
            continue
        data = github_get_file_bytes(cfg, file_path)
        if data is None:
            continue
        (upload_dir / name).write_bytes(data)
        synced += 1
    return synced


def build_department_detail_zip(df: pd.DataFrame, export_cols: list[str]) -> bytes:
    period_sheets = [
        ("3개월미만", "3개월 미만"),
        ("3~6개월", "3~6개월"),
        ("6~12개월", "6~12개월"),
        ("12개월이상", "12개월+"),
    ]

    output = BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for department, dept_df in df.groupby("담당부서", dropna=True):
            workbook_buffer = BytesIO()
            with pd.ExcelWriter(workbook_buffer, engine="xlsxwriter") as writer:
                for sheet_name, aging_label in period_sheets:
                    period_df = dept_df
                    if "에이징" in dept_df.columns:
                        period_df = dept_df[dept_df["에이징"] == aging_label]
                    period_df[export_cols].to_excel(writer, sheet_name=sheet_name, index=False)

            workbook_buffer.seek(0)
            zip_file.writestr(
                f"소진계획_{_safe_excel_name(department)}.xlsx",
                workbook_buffer.getvalue(),
            )

    output.seek(0)
    return output.getvalue()


def persist_uploaded_file(uploaded, target_path: Path, github_cfg: dict) -> bool:
    data = uploaded.getvalue()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data)
    if github_cfg.get("enabled"):
        github_put_file_bytes(
            github_cfg,
            "data/latest_장기재고현황.xlsx",
            data,
            f"update latest inventory: {uploaded.name}",
        )
        return True
    return False


def persist_disposal_uploads(
    uploaded_files: list,
    github_cfg: dict,
    known_departments: list[str],
) -> tuple[int, dict[str, str]]:
    DISPOSAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    department_updates: dict[str, str] = {}
    for uploaded in uploaded_files:
        safe_name = _safe_excel_name(Path(uploaded.name).name)
        target = DISPOSAL_UPLOAD_DIR / safe_name
        data = uploaded.getvalue()
        target.write_bytes(data)
        if github_cfg.get("enabled"):
            github_put_file_bytes(
                github_cfg,
                f"data/disposal_uploads/{safe_name}",
                data,
                f"update disposal plan: {safe_name}",
            )
        department = infer_department_from_filename(uploaded.name, known_departments)
        department_updates[department] = _now_str()
        saved += 1
    return saved, department_updates


def _status_priority(status: str) -> int:
    order = {
        "❌ 미등록": 0,
        "✅ 계획 등록됨": 1,
        "⏳ 진행 중": 2,
        "🎯 완료": 3,
    }
    return order.get(status, 1)


def _normalize_plan_status(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "✅ 계획 등록됨"
    text = str(value).strip()
    if "완료" in text:
        return "🎯 완료"
    if "진행" in text:
        return "⏳ 진행 중"
    if "미등록" in text:
        return "❌ 미등록"
    if "등록" in text or "계획" in text:
        return "✅ 계획 등록됨"
    return text


def _find_column_name(columns: pd.Index, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().replace(" ", ""): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for col in columns:
        col_text = str(col).strip().replace(" ", "")
        if any(candidate in col_text for candidate in candidates):
            return col
    return None


def build_disposal_status_map(upload_dir: Path) -> dict[str, str]:
    status_map: dict[str, str] = {}
    if not upload_dir.exists():
        return status_map

    for file_path in sorted(upload_dir.glob("*.xlsx")):
        try:
            workbook = pd.ExcelFile(file_path)
        except Exception:
            logging.warning("Invalid disposal workbook skipped: %s", file_path.name)
            continue

        for sheet_name in workbook.sheet_names:
            try:
                sheet_df = pd.read_excel(workbook, sheet_name=sheet_name)
            except Exception:
                continue

            if sheet_df.empty:
                continue

            item_col = _find_column_name(sheet_df.columns, ["품목코드", "품목번호", "itemcode"])
            status_col = _find_column_name(sheet_df.columns, ["소진계획", "진행상태", "상태"])
            if item_col is None:
                continue

            for _, row in sheet_df.iterrows():
                item_code = str(row.get(item_col, "")).strip()
                if not item_code or item_code.lower() == "nan":
                    continue
                status_value = row.get(status_col) if status_col is not None else None
                normalized = _normalize_plan_status(status_value)
                prev = status_map.get(item_code)
                if prev is None or _status_priority(normalized) >= _status_priority(prev):
                    status_map[item_code] = normalized

    return status_map


@st.cache_data(show_spinner=False)
def get_inventory_data(source) -> pd.DataFrame:
    return load_inventory_data(source)


load_css(ASSETS_DIR / "styles.css")
st.markdown(
    """
    <style>
    div[data-testid="stRadio"] div[role="radiogroup"] {
      display: flex;
      flex-wrap: wrap;
      row-gap: 0.35rem;
      column-gap: 0.75rem;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
      flex: 0 0 calc(25% - 0.75rem);
      margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='dashboard-title'>🏢 장기재고현황 대시보드</div>", unsafe_allow_html=True)
data_source_label = None
github_cfg = get_github_config()
inventory_meta = load_upload_metadata(
    github_cfg,
    INVENTORY_META_FILE,
    "data/inventory_upload_meta.json",
    {
        "last_uploaded_at": "",
        "last_uploaded_file": "",
        "last_total_amount": None,
        "previous_uploaded_at": "",
        "previous_total_amount": None,
    },
)
disposal_meta = load_upload_metadata(
    github_cfg,
    DISPOSAL_META_FILE,
    "data/disposal_upload_meta.json",
    {"departments": {}},
)

if github_cfg.get("enabled") and not st.session_state.get("github_disposal_synced"):
    try:
        synced = sync_disposal_uploads_from_github(github_cfg, DISPOSAL_UPLOAD_DIR)
        st.session_state["github_disposal_synced"] = True
        st.session_state["github_disposal_synced_count"] = synced
    except Exception as exc:
        logging.exception("Failed to sync disposal uploads from GitHub")
        st.session_state["github_disposal_sync_error"] = str(exc)

inventory_uploaded_at = inventory_meta.get("last_uploaded_at", "")
base_date = inventory_uploaded_at[:10] if inventory_uploaded_at else "미설정"
inventory_update_text = _to_display_time(inventory_uploaded_at)
previous_inventory_update_text = _to_display_time(inventory_meta.get("previous_uploaded_at", ""))
st.markdown(
    f"<div class='dashboard-meta'>📅 데이터 기준일: {base_date} | 🗂 장기재고현황 업데이트: {inventory_update_text} | ⏮ 이전업데이트: {previous_inventory_update_text}</div>",
    unsafe_allow_html=True,
)

active_inventory_file = LATEST_INVENTORY_FILE if LATEST_INVENTORY_FILE.exists() else INVENTORY_FILE

try:
    github_inventory_bytes = None
    if github_cfg.get("enabled"):
        github_inventory_bytes = github_get_file_bytes(github_cfg, "data/latest_장기재고현황.xlsx")
    if github_inventory_bytes is not None:
        df = get_inventory_data(github_inventory_bytes)
        data_source_label = "📂 최신 반영 파일: GitHub data/latest_장기재고현황.xlsx"
    else:
        if not active_inventory_file.exists():
            st.error("입력 파일을 찾을 수 없습니다: data/장기재고현황.xlsx")
            st.stop()
        df = get_inventory_data(active_inventory_file.read_bytes())
        if active_inventory_file == LATEST_INVENTORY_FILE:
            data_source_label = f"📂 최신 반영 파일: {LATEST_INVENTORY_FILE.name}"
        else:
            data_source_label = f"📂 기본 파일: {INVENTORY_FILE.name}"
except Exception as exc:
    logging.exception("Failed to load inventory data")
    st.error(f"데이터 로딩 중 오류가 발생했습니다: {exc}")
    st.stop()

df = remove_header_like_rows(df)
df = clean_department_rows(df)
if "에이징" in df.columns:
    df = df[df["에이징"] != "미상"].reset_index(drop=True)

disposal_status_map = build_disposal_status_map(DISPOSAL_UPLOAD_DIR)
if "품목코드" in df.columns:
    item_codes = df["품목코드"].astype(str).str.strip()
    df["소진계획"] = item_codes.map(disposal_status_map).fillna("❌ 미등록")
else:
    df["소진계획"] = "❌ 미등록"

department_options = ["전체"]
if "담당부서" in df.columns:
    dept_values = df["담당부서"].dropna().astype(str).str.strip()
    department_options += sorted(dept_values.unique().tolist())

amount_cols = [c for c in df.columns if c.endswith("_금액")]
current_amount = calculate_inventory_total_amount(df)

long_term_amount = df["12개월+_금액"].sum() if "12개월+_금액" in df.columns else 0
long_term_items = int((df["12개월+_금액"] > 0).sum()) if "12개월+_금액" in df.columns else 0

long_term_rate = (long_term_amount / current_amount * 100) if current_amount else 0

kpi_html = """
<div class='kpi-row'>
  <div class='kpi-card'>
    <div class='kpi-header'>💰 총재고금액</div>
    <div class='kpi-value'>{total_amount}</div>
    <div class='kpi-delta' style='color:{delta_color};'>{delta_text}</div>
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

prev_update_amount = inventory_meta.get("previous_total_amount")
try:
    prev_update_amount = float(prev_update_amount) if prev_update_amount is not None else None
except Exception:
    prev_update_amount = None

if prev_update_amount and prev_update_amount != 0:
    delta_pct = (current_amount - prev_update_amount) / prev_update_amount * 100
    delta_text = f"{'▲' if delta_pct >= 0 else '▼'} {abs(delta_pct):.1f}% (이전업데이트 대비)"
    delta_color = COLORS["danger_red"] if delta_pct >= 0 else COLORS["primary_blue"]
else:
    delta_text = "이전업데이트 없음"
    delta_color = COLORS["text_gray"]

total_amount_display = format_krw(current_amount)

st.markdown(
    kpi_html.format(
        total_amount=total_amount_display,
        delta_text=delta_text,
        delta_color=delta_color,
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

left_col, right_col = st.columns([0.5, 0.5], gap="large")

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
    selected_right_department = st.radio(
        "담당부서 선택",
        department_options,
        horizontal=True,
        key="right_department_filter",
    )

    graph_df = df
    if selected_right_department != "전체" and "담당부서" in df.columns:
        graph_df = df[df["담당부서"] == selected_right_department]

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
                "금액": sum_amount(graph_df, amount_cols),
                "수량": sum_amount(graph_df, qty_cols),
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

st.markdown("<div class='section-title'>🔍 담당부서별 상세 현황</div>", unsafe_allow_html=True)

filter_cols = st.columns([0.25, 0.25, 0.25, 0.25])

with filter_cols[0]:
    selected_departments = st.multiselect("담당부서", department_options, default=["전체"])

with filter_cols[1]:
    aging_options = ["전체", "3개월 미만", "3~6개월", "6~12개월", "12개월+"]
    selected_aging = st.selectbox("장기재고 기간", aging_options)

with filter_cols[2]:
    amount_max = int(df["12개월+_금액"].max()) if "12개월+_금액" in df.columns else 0
    slider_max = amount_max if amount_max > 0 else 1
    amount_range = st.slider("장기재고 금액", 0, slider_max, (0, slider_max))

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

display_df = filtered[columns_to_show].copy()
amount_display_cols = [col for col in columns_to_show if col.endswith("_금액")]
for col in amount_display_cols:
    display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0).round(0).astype("int64")

column_config = {
    col: st.column_config.NumberColumn(col, format="%,d")
    for col in amount_display_cols
}

st.dataframe(
    display_df,
    use_container_width=True,
    height=420,
    column_config=column_config if column_config else None,
)

download_zip = None
if not filtered.empty and "담당부서" in filtered.columns and columns_to_show:
    download_zip = build_department_detail_zip(filtered, columns_to_show)

st.markdown("<div class='action-buttons'>", unsafe_allow_html=True)

st.download_button(
    label="담당부서별 상세현황 다운로드",
    data=download_zip if download_zip else b"",
    file_name=f"소진계획_담당부서별_{datetime.now().strftime('%Y%m%d')}.zip",
    mime="application/zip",
    disabled=download_zip is None,
)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>⬆️ 업로드 관리</div>", unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)

if github_cfg.get("enabled"):
    st.caption(f"GitHub 자동저장 모드: {github_cfg['repo']} ({github_cfg['branch']})")
else:
    st.caption("GitHub 자동저장 비활성화: 로컬 반영만 수행됩니다.")

if st.session_state.get("github_disposal_sync_error"):
    st.warning(f"GitHub 소진계획 동기화 실패: {st.session_state.get('github_disposal_sync_error')}")
elif st.session_state.get("github_disposal_synced_count", 0) > 0:
    st.caption(f"GitHub 소진계획 파일 동기화: {st.session_state.get('github_disposal_synced_count')}개")

uploaded_file = st.file_uploader("장기재고현황 업로드", type=["xlsx"], key="inventory_upload_widget")
uploaded_disposal_files = st.file_uploader(
    "소진계획 파일 업로드(부서별, 복수 선택 가능)",
    type=["xlsx"],
    accept_multiple_files=True,
    key="disposal_upload_widget",
)

upload_processed = False

if uploaded_file is not None:
    inventory_bytes = uploaded_file.getvalue()
    inventory_sig = f"{uploaded_file.name}:{len(inventory_bytes)}:{hash(inventory_bytes)}"
    if st.session_state.get("inventory_upload_sig") != inventory_sig:
        try:
            uploaded_df_preview = load_inventory_data(uploaded_file.getvalue())
            new_total_amount = calculate_inventory_total_amount(uploaded_df_preview)
            prev_last_amount = inventory_meta.get("last_total_amount")
            prev_last_uploaded_at = inventory_meta.get("last_uploaded_at", "")

            github_synced = persist_uploaded_file(uploaded_file, LATEST_INVENTORY_FILE, github_cfg)
            if prev_last_amount is not None:
                inventory_meta["previous_total_amount"] = float(prev_last_amount)
            if prev_last_uploaded_at:
                inventory_meta["previous_uploaded_at"] = prev_last_uploaded_at
            inventory_meta["last_uploaded_at"] = _now_str()
            inventory_meta["last_uploaded_file"] = uploaded_file.name
            inventory_meta["last_total_amount"] = float(new_total_amount)
            save_upload_metadata(
                github_cfg,
                INVENTORY_META_FILE,
                "data/inventory_upload_meta.json",
                inventory_meta,
                "update inventory upload metadata",
            )

            st.session_state["inventory_upload_sig"] = inventory_sig
            st.success(
                "장기재고현황 최신 파일을 GitHub와 로컬에 반영했습니다."
                if github_synced
                else "장기재고현황 최신 파일을 로컬에 저장했습니다."
            )
            upload_processed = True
        except Exception as exc:
            logging.exception("Failed to persist inventory upload")
            st.error(f"장기재고현황 저장 중 오류가 발생했습니다: {exc}")

if uploaded_disposal_files:
    disposal_sig = tuple((f.name, f.size) for f in uploaded_disposal_files)
    if st.session_state.get("disposal_upload_sig") != disposal_sig:
        try:
            saved_count, dept_updates = persist_disposal_uploads(
                uploaded_disposal_files,
                github_cfg,
                DEFAULT_DEPARTMENTS,
            )
            if "departments" not in disposal_meta or not isinstance(disposal_meta["departments"], dict):
                disposal_meta["departments"] = {}
            disposal_meta["departments"].update(dept_updates)
            save_upload_metadata(
                github_cfg,
                DISPOSAL_META_FILE,
                "data/disposal_upload_meta.json",
                disposal_meta,
                "update disposal upload metadata",
            )

            st.session_state["disposal_upload_sig"] = disposal_sig
            st.success(
                f"소진계획 파일 {saved_count}개를 GitHub와 로컬에 반영했습니다."
                if github_cfg.get("enabled")
                else f"소진계획 파일 {saved_count}개를 로컬에 반영했습니다."
            )
            upload_processed = True
        except Exception as exc:
            logging.exception("Failed to persist disposal uploads")
            st.error(f"소진계획 파일 저장 중 오류가 발생했습니다: {exc}")

disposal_departments = disposal_meta.get("departments", {}) if isinstance(disposal_meta, dict) else {}
display_departments = DEFAULT_DEPARTMENTS + sorted([d for d in disposal_departments.keys() if d not in DEFAULT_DEPARTMENTS])
update_rows = [
    {
        "담당부서": dept,
        "마지막 업로드": _to_display_time(disposal_departments.get(dept, "")),
    }
    for dept in display_departments
]
st.dataframe(pd.DataFrame(update_rows), use_container_width=True, hide_index=True, height=245)

if data_source_label:
    st.caption(data_source_label)
st.caption(f"소진계획 반영 품목 수: {len(disposal_status_map)}")

st.markdown("</div>", unsafe_allow_html=True)

if upload_processed:
    st.rerun()

