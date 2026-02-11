from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"

INVENTORY_FILE = DATA_DIR / "장기재고현황.xlsx"
LATEST_INVENTORY_FILE = DATA_DIR / "latest_장기재고현황.xlsx"
DISPOSAL_UPLOAD_DIR = DATA_DIR / "disposal_uploads"
EMAILS_FILE = DATA_DIR / "email_addresses.csv"
DISPOSAL_FILE = DATA_DIR / "disposal_plans.csv"
EMAIL_LOG_FILE = LOGS_DIR / "email_logs.csv"
APP_LOG_FILE = LOGS_DIR / "app.log"

COLORS = {
    "primary_blue": "#2196F3",
    "primary_dark": "#1976D2",
    "primary_light": "#BBDEFB",
    "success_green": "#4CAF50",
    "warning_yellow": "#FFC107",
    "warning_orange": "#FF9800",
    "danger_red": "#F44336",
    "background_gray": "#F5F5F5",
    "border_gray": "#E0E0E0",
    "text_dark": "#212121",
    "text_gray": "#757575",
    "aging_safe": "#4CAF50",
    "aging_caution": "#FFC107",
    "aging_warning": "#FF9800",
    "aging_danger": "#F44336",
}

FONTS = {
    "title_family": "'Noto Sans KR', sans-serif",
    "subtitle_family": "'Noto Sans KR', sans-serif",
    "body_family": "'Noto Sans KR', sans-serif",
    "number_family": "'Roboto', sans-serif",
}

LAYOUT = {
    "max_width": "1400px",
    "sidebar_width": "300px",
    "border_radius": "8px",
    "box_shadow": "0 2px 8px rgba(0,0,0,0.1)",
}
