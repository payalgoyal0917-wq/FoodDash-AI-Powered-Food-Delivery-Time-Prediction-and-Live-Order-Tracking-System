"""
FoodDash — Food Delivery + Delivery Time Prediction (Streamlit App)
Author: Abdul Rehman
Model: XGBoost Regressor (Test R2 ~0.83, Test RMSE ~3.9 min)

This single file is both the "frontend" (Streamlit UI) and "backend"
(loading the trained model/scaler + running inference) — no separate
API server is needed, Streamlit handles both.

Requires: streamlit>=1.40, xgboost, scikit-learn, joblib, pandas, numpy
"""

import inspect
import math
import re
import time

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

# --------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="FoodDash · Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Newer Streamlit uses width="stretch"; older uses use_container_width=True
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
           else {"use_container_width": True})

# --------------------------------------------------------------------------
# Constants — MUST mirror the notebook's feature engineering exactly
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.json"
SCALER_PATH = BASE_DIR / "standard_scaler.pkl"

TRAFFIC_MAPPING = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

TYPE_OF_ORDER_OPTIONS = ["Buffet", "Drinks", "Meal", "Snack"]
TYPE_OF_VEHICLE_OPTIONS = ["bicycle", "electric_scooter", "motorcycle", "scooter"]
CITY_OPTIONS = ["Metropolitian", "Semi-Urban", "Urban"]
WEATHER_OPTIONS = ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]
FESTIVAL_OPTIONS = ["No", "Yes"]

# Exact column order the scaler/model were fitted on (from X_train.columns)
FEATURE_COLUMNS = [
    "Delivery_person_Age", "Delivery_person_Ratings", "distance_km",
    "order_hour", "Road_traffic_density", "Vehicle_condition",
    "multiple_deliveries", "Type_of_order_Drinks", "Type_of_order_Meal",
    "Type_of_order_Snack", "Type_of_vehicle_electric_scooter",
    "Type_of_vehicle_motorcycle", "Type_of_vehicle_scooter", "Festival_Yes",
    "City_Semi-Urban", "City_Urban", "Weatherconditions_Fog",
    "Weatherconditions_Sandstorms", "Weatherconditions_Stormy",
    "Weatherconditions_Sunny", "Weatherconditions_Windy",
]

SOCIAL_LINKS = {
    "GitHub": "https://github.com/datawithabdulrehman",
    "Kaggle": "https://www.kaggle.com/datawithabxrehman",
    "LinkedIn": "https://www.linkedin.com/in/datawithabdulrehman",
}

# --------------------------------------------------------------------------
# App data (restaurants, coupons, defaults) — front-end only, no model impact
# --------------------------------------------------------------------------
RESTAURANTS = [
    {"name": "Spice Route", "emoji": "🍛", "cuisine": "Indian", "veg": False, "rating": 4.6,
     "mins": 30, "for_two": 250, "lat": 22.745049, "lon": 75.892471, "tag": "Bestseller",
     "menu": [("Butter Chicken", 280, "🍗", False, False), ("Paneer Tikka", 220, "🧀", True, False),
              ("Garlic Naan", 50, "🫓", True, False), ("Dal Makhani", 190, "🍲", True, False)],
     "reviews": [(5, "Hot and fragrant, the butter chicken was spot on.", "Aman"),
                 (4, "Generous portions. Naan could be softer.", "Riya")]},
    {"name": "Pizza Palace", "emoji": "🍕", "cuisine": "Italian", "veg": True, "rating": 4.4,
     "mins": 25, "for_two": 300, "lat": 22.7231, "lon": 75.8839, "tag": "Pure veg",
     "menu": [("Margherita", 249, "🍕", True, True), ("Farmhouse", 329, "🍕", True, True),
              ("Pasta Alfredo", 260, "🍝", True, False), ("Garlic Bread", 120, "🥖", True, False)],
     "reviews": [(5, "Crisp base and plenty of cheese.", "Karan"),
                 (4, "Arrived earlier than the app said.", "Sneha")]},
    {"name": "Burger Barn", "emoji": "🍔", "cuisine": "Fast Food", "veg": False, "rating": 4.2,
     "mins": 20, "for_two": 180, "lat": 22.7532, "lon": 75.8937, "tag": "Quick bites",
     "menu": [("Classic Burger", 150, "🍔", False, True), ("Veggie Burger", 130, "🥬", True, True),
              ("Fries", 90, "🍟", True, False), ("Chicken Wings", 210, "🍗", False, False)],
     "reviews": [(4, "Juicy patty, fries stayed crunchy.", "Vikram"),
                 (4, "Good value for the price.", "Meera")]},
    {"name": "Dragon Wok", "emoji": "🥡", "cuisine": "Chinese", "veg": False, "rating": 4.3,
     "mins": 35, "for_two": 220, "lat": 22.7196, "lon": 75.8577, "tag": "Wok-fired",
     "menu": [("Hakka Noodles", 170, "🍜", True, False), ("Manchurian", 160, "🥘", True, False),
              ("Dumplings", 190, "🥟", False, False), ("Fried Rice", 150, "🍚", True, False)],
     "reviews": [(5, "Proper wok flavour in the noodles.", "Ishaan"),
                 (3, "Tasty but a bit oily.", "Pooja")]},
    {"name": "Sweet Tooth", "emoji": "🍰", "cuisine": "Dessert", "veg": True, "rating": 4.7,
     "mins": 15, "for_two": 150, "lat": 22.7410, "lon": 75.9002, "tag": "Top rated",
     "menu": [("Chocolate Cake", 180, "🍰", True, False), ("Gulab Jamun", 80, "🍡", True, False),
              ("Ice Cream", 110, "🍨", True, False)],
     "reviews": [(5, "The cake arrived perfectly intact.", "Naina"),
                 (5, "Gulab jamun was warm and soft.", "Rohit")]},
    {"name": "Green Bowl", "emoji": "🥗", "cuisine": "Healthy", "veg": True, "rating": 4.5,
     "mins": 22, "for_two": 200, "lat": 22.7654, "lon": 75.8851, "tag": "Fresh daily",
     "menu": [("Quinoa Salad", 210, "🥗", True, False), ("Smoothie Bowl", 190, "🍓", True, False),
              ("Veg Wrap", 160, "🌯", True, False)],
     "reviews": [(5, "Fresh, filling and not heavy.", "Ayesha"),
                 (4, "Wrap was big enough for two meals.", "Dev")]},
]
for _r in RESTAURANTS:
    _r["key"] = re.sub(r"[^a-z0-9]+", "_", _r["name"].lower()).strip("_")
REST_BY_NAME = {r["name"]: r for r in RESTAURANTS}

COUPONS = {
    "SAVE20": "20% off your food (up to ₹150)",
    "FREEDEL": "Free delivery",
}
CHEESE_PRICE = 30

# Same defaults the original form used
DEFAULT_COND = dict(
    age=29, ratings=4.6, vehicle_type="motorcycle", vehicle_condition=1,
    order_type="Snack", order_hour=13, multiple_deliveries=0, festival="No",
    rest_lat=22.745049, rest_lon=75.892471, del_lat=22.765049, del_lon=75.912471,
    weather="Sunny", traffic="Medium", city="Urban",
)

SPEEDS = {
    "Demo: 1 min = 1 sec": 1.0,
    "Slow: 1 min = 3 sec": 3.0,
    "Real time": 60.0,
}

STAGES = ["Order confirmed", "Preparing food", "Packing", "Out for delivery", "Delivered"]
STAGE_ICONS = ["✅", "🍳", "📦", "🛵", "🎉"]
RIDER_NAMES = ["Ravi", "Imran", "Suresh", "Amit", "Farhan", "Deepak", "Arjun", "Manoj"]

# Sample platform analytics shown on the dashboard (illustrative data)
SAMPLE_HOURLY = [2, 1, 0, 0, 1, 3, 5, 9, 12, 10, 14, 22, 28, 20, 15, 12, 14, 18, 30, 38, 42, 35, 20, 8]
SAMPLE_WEEK_REVENUE = {"Mon": 42.0, "Tue": 38.0, "Wed": 45.0, "Thu": 51.0, "Fri": 74.0, "Sat": 92.0, "Sun": 86.0}
SAMPLE_TOP_DISHES = {"Butter Chicken": 94, "Margherita": 81, "Classic Burger": 70,
                     "Hakka Noodles": 52, "Chocolate Cake": 44}

# --------------------------------------------------------------------------
# Styling — premium: deep pine + saffron, light & dark themes
# --------------------------------------------------------------------------
THEMES = {
    "light": dict(bg="#F4F6F2", surface="#FFFFFF", ink="#10241E", muted="#5E6D67", line="#E1E7E2",
                  soft="#EDF2EE", accent="#F2A20C", chili="#D8452B", ok="#1E8E5A", pine="#10241E"),
    "dark": dict(bg="#0D1612", surface="#15231D", ink="#EAF1ED", muted="#9CADA5", line="#27382F",
                 soft="#1B2C25", accent="#F2A20C", chili="#E5593F", ok="#34B77A", pine="#2A4A3D"),
}

STATIC_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .stApp { background-color: var(--bg); color: var(--ink); }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1180px; }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stMarkdown p, .stMarkdown li, [data-testid="stWidgetLabel"] p { color: var(--ink); }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }
    label { font-weight: 500 !important; color: var(--ink) !important; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, #0E2A21 0%, #173B2F 100%);
        border-radius: 22px; padding: 2.3rem 2.6rem 2.1rem; margin-bottom: 1.2rem;
        box-shadow: 0 14px 36px rgba(14, 42, 33, 0.22);
    }
    .hero h1 {
        font-family: 'Bricolage Grotesque', 'Inter', sans-serif;
        font-size: 2.3rem; font-weight: 800; margin: 1.1rem 0 0.4rem 0;
        color: #FFFFFF !important; letter-spacing: -0.02em;
    }
    .hero p { font-size: 1.02rem; color: #CFE0D8 !important; margin: 0; max-width: 760px; }
    .hero .badge {
        display: inline-block; background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.18); padding: 4px 12px; border-radius: 20px;
        font-size: 0.78rem; color: #EAF3EE; margin-top: 1rem; margin-right: 8px;
    }
    .hero-route { position: relative; height: 26px; }
    .hero-route .hr-line { position: absolute; left: 0; right: 0; top: 13px; border-top: 2px dashed rgba(242,162,12,0.55); }
    .hero-route .hr-dot { position: absolute; top: 7px; width: 12px; height: 12px; border-radius: 50%; background: #F2A20C; left: 0; }
    .hero-route .hr-end { left: auto; right: 0; background: #34B77A; }
    .hero-route .hr-rider { position: absolute; top: -2px; left: 0; font-size: 1.4rem; animation: ride 7s ease-in-out infinite alternate; }
    @keyframes ride { from { left: 0; } to { left: calc(100% - 28px); } }
    @media (prefers-reduced-motion: reduce) { .hero-route .hr-rider { animation: none; left: 45%; } }

    /* ---------- Navigation (radio as pills) ---------- */
    div[role="radiogroup"] { gap: 0.4rem; flex-wrap: wrap; }
    div[role="radiogroup"] > label {
        background: var(--surface); border: 1px solid var(--line); border-radius: 999px;
        padding: 0.35rem 1rem; cursor: pointer; margin: 0 !important;
    }
    div[role="radiogroup"] > label > div:first-child { display: none; }
    div[role="radiogroup"] > label:has(input:checked) { background: var(--pine); border-color: var(--pine); }
    div[role="radiogroup"] > label:has(input:checked) p { color: #FFFFFF !important; }

    /* ---------- Cards (st.container(key="card_*")) ---------- */
    [class*="st-key-card_"] {
        background: var(--surface); border: 1px solid var(--line); border-radius: 18px;
        padding: 1.1rem 1.2rem; box-shadow: 0 1px 2px rgba(16,36,30,0.05); margin-bottom: 0.4rem;
    }
    .section-title {
        font-family: 'Bricolage Grotesque', 'Inter', sans-serif;
        font-size: 1.12rem; font-weight: 700; color: var(--ink);
        margin: 0.2rem 0 0.8rem 0; display: flex; align-items: center; gap: 8px;
    }
    .rc-top { display: flex; justify-content: space-between; align-items: flex-start; }
    .rc-emoji { font-size: 2.6rem; line-height: 1; }
    .rc-badge { background: var(--soft); color: var(--ink); border-radius: 8px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; }
    .rc-name { font-family: 'Bricolage Grotesque', 'Inter', sans-serif; font-weight: 700; font-size: 1.1rem; color: var(--ink); margin-top: 0.4rem; }
    .rc-meta { color: var(--muted); font-size: 0.84rem; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.55rem 0 0.4rem; }
    .chips span { background: var(--soft); color: var(--ink); border-radius: 8px; padding: 2px 8px; font-size: 0.78rem; font-weight: 500; }
    .chips span.ai { background: rgba(242,162,12,0.18); }
    .dot-veg, .dot-non { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
    .dot-veg { background: #1E8E5A; } .dot-non { background: #D8452B; }
    .item-name { font-weight: 600; color: var(--ink); }
    .item-price { color: var(--muted); font-size: 0.88rem; }
    .review { border-left: 3px solid var(--accent); padding: 4px 12px; margin: 8px 0; color: var(--ink); font-size: 0.9rem; }

    /* ---------- Buttons ---------- */
    div.stButton > button, button[data-testid^="stBaseButton-secondary"] {
        background: var(--surface); color: var(--ink); border: 1px solid var(--line);
        border-radius: 12px; font-weight: 600; transition: all 0.15s ease;
    }
    div.stButton > button:hover, button[data-testid^="stBaseButton-secondary"]:hover {
        border-color: var(--accent); color: var(--ink); transform: translateY(-1px);
    }
    button[data-testid="stBaseButton-secondaryFormSubmit"] {
        background: var(--pine); color: #FFFFFF; border: none; padding: 0.75rem 1.5rem;
        font-size: 1rem; width: 100%;
    }
    button[data-testid="stBaseButton-secondaryFormSubmit"]:hover { background: #1E4A3B; color: #FFFFFF; }
    button[data-testid^="stBaseButton-primary"] {
        background: var(--chili); color: #FFFFFF; border: none; border-radius: 12px;
        font-weight: 700; padding: 0.75rem 1.5rem; font-size: 1rem; width: 100%;
    }
    button[data-testid^="stBaseButton-primary"]:hover { filter: brightness(1.08); color: #FFFFFF; }
    button[data-testid^="stBaseButton-primary"]:disabled { opacity: 0.45; }
    .stApp button p { color: inherit !important; }

    /* ---------- Inputs ---------- */
    .stApp input, .stApp textarea { color: var(--ink) !important; -webkit-text-fill-color: var(--ink); }
    [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
        background-color: var(--surface) !important; border-color: var(--line) !important;
    }
    [data-baseweb="select"] span, [data-baseweb="select"] div { color: var(--ink); }
    [data-baseweb="popover"] ul, [data-baseweb="menu"] { background-color: var(--surface) !important; }
    [data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: var(--ink) !important; }
    [data-testid="stExpander"] { background: var(--surface); border-color: var(--line); border-radius: 14px; }
    div[data-testid="stMetricValue"] { color: var(--ink); }

    /* ---------- Result card (original) ---------- */
    .result-card {
        background: linear-gradient(135deg, #ECFDF3 0%, #F7FBF8 100%);
        border: 1px solid #A6F4C5; border-radius: 18px; padding: 2rem; text-align: center; margin-top: 1rem;
    }
    .result-minutes { font-family: 'Bricolage Grotesque', 'Inter', sans-serif; font-size: 3.4rem; font-weight: 800; color: #067647; line-height: 1; margin: 0.3rem 0; }
    .result-label { color: #475467; font-size: 0.95rem; font-weight: 500; }

    /* ---------- Bill ---------- */
    .bill-row { display: flex; justify-content: space-between; padding: 3px 0; color: var(--ink); font-size: 0.95rem; }
    .bill-row.total { border-top: 1px dashed var(--line); margin-top: 6px; padding-top: 8px; font-weight: 800; font-size: 1.1rem; }
    .bill-row.save { color: var(--ok); }

    /* ---------- Live tracking ---------- */
    .eta-banner {
        display: flex; flex-wrap: wrap; gap: 1rem; justify-content: space-between; align-items: center;
        background: linear-gradient(135deg, #0E2A21 0%, #173B2F 100%); border-radius: 20px; padding: 1.4rem 1.7rem; margin-bottom: 0.9rem;
    }
    .eta-lbl { color: #CFE0D8; font-size: 0.9rem; }
    .eta-big { font-family: 'Bricolage Grotesque', 'Inter', sans-serif; font-weight: 800; font-size: 3rem; line-height: 1.05; color: #F2A20C; }
    .eta-msg { color: #FFFFFF; font-size: 1.05rem; font-weight: 600; max-width: 520px; }
    .eta-msg small { display: block; color: #CFE0D8; font-weight: 400; margin-top: 4px; font-size: 0.88rem; }
    .stp { position: relative; display: grid; grid-template-columns: repeat(5, 1fr); margin: 1.1rem 0 0.4rem; }
    .stp-track { position: absolute; top: 19px; left: 10%; right: 10%; height: 4px; background: var(--line); border-radius: 4px; }
    .stp-track i { display: block; height: 100%; background: var(--ok); border-radius: 4px; transition: width 0.8s linear; }
    .stp-n { position: relative; text-align: center; font-size: 0.82rem; color: var(--muted); }
    .stp-n b { display: grid; place-items: center; width: 40px; height: 40px; margin: 0 auto 6px; border-radius: 50%;
               background: var(--surface); border: 2px solid var(--line); font-size: 1.1rem; }
    .stp-n.done b { background: var(--ok); border-color: var(--ok); color: #FFFFFF; }
    .stp-n.active { color: var(--ink); font-weight: 700; }
    .stp-n.active b { border-color: var(--accent); box-shadow: 0 0 0 5px rgba(242,162,12,0.22); }
    .stp-n small { display: block; font-weight: 500; color: var(--muted); }
    .stp-n.active small { color: var(--chili); font-weight: 700; }
    .rider { display: flex; gap: 12px; align-items: center; }
    .rider .av { font-size: 2.2rem; }
    .kpi { background: var(--soft); border-radius: 14px; padding: 0.7rem 0.9rem; }
    .kpi span { color: var(--muted); font-size: 0.8rem; display: block; }
    .kpi b { font-family: 'Bricolage Grotesque', 'Inter', sans-serif; font-size: 1.35rem; color: var(--ink); }

    /* ---------- Footer (original) ---------- */
    .footer { margin-top: 3rem; padding-top: 1.8rem; border-top: 1px solid var(--line); text-align: center; }
    .footer-icons { display: flex; justify-content: center; gap: 16px; margin-bottom: 0.9rem; }
    .footer-icons a {
        display: flex; align-items: center; justify-content: center; width: 42px; height: 42px;
        border-radius: 50%; background: var(--surface); border: 1px solid var(--line);
        transition: all 0.15s ease; box-shadow: 0 1px 2px rgba(16,24,40,0.05);
    }
    .footer-icons a:hover { background: var(--pine); transform: translateY(-2px); }
    .footer-icons a:hover svg { fill: #FFFFFF; }
    .footer-icons svg { width: 20px; height: 20px; fill: var(--muted); transition: fill 0.15s ease; }
    .footer-text { color: var(--muted); font-size: 0.85rem; }

    @media (max-width: 640px) {
        .hero { padding: 1.5rem 1.2rem; } .hero h1 { font-size: 1.6rem; }
        .eta-big { font-size: 2.3rem; } .stp-n { font-size: 0.7rem; }
        .stp-n b { width: 34px; height: 34px; } .stp-track { top: 16px; }
    }
</style>
"""


def inject_theme(dark: bool):
    t = THEMES["dark" if dark else "light"]
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in t.items()) + "}"
    st.markdown(f"<style>{root}</style>", unsafe_allow_html=True)
    st.markdown(STATIC_CSS, unsafe_allow_html=True)


def H(html: str):
    """Render an HTML snippet (indentation/newlines stripped so markdown never treats it as code)."""
    st.markdown("".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


def inr(x: float) -> str:
    return f"₹{x:,.0f}"


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state():
    defaults = {
        "page": "home", "cart": {}, "cart_rest": None, "coupon": None, "fav": set(),
        "orders": [], "user": None, "cond": {}, "open_rest": None,
        "dark": False, "speed_label": "Demo: 1 min = 1 sec", "order_seq": 1000,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
if "_goto" in st.session_state:
    st.session_state["page"] = st.session_state.pop("_goto")

inject_theme(st.session_state["dark"])

# --------------------------------------------------------------------------
# Load model + scaler (cached so it only happens once per session)
# --------------------------------------------------------------------------
import xgboost as xgb
import joblib

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    
    # Naya aur mehfooz tareeqa
    model = xgb.XGBRegressor() # Agar classification hai toh XGBClassifier() likhein
    model.load_model(str(MODEL_PATH))
    
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def haversine_km(lat1, lon1, lat2, lon2):
    """Same great-circle distance formula used in the training notebook."""
    r = 6371  # earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * r


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Turns the raw form inputs into the exact 21-column row the scaler/model expect."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    row["Delivery_person_Age"] = inputs["age"]
    row["Delivery_person_Ratings"] = inputs["ratings"]
    row["distance_km"] = inputs["distance_km"]
    row["order_hour"] = inputs["order_hour"]
    row["Road_traffic_density"] = TRAFFIC_MAPPING[inputs["traffic"]]
    row["Vehicle_condition"] = inputs["vehicle_condition"]
    row["multiple_deliveries"] = inputs["multiple_deliveries"]

    # One-hot: Type_of_order (base category dropped: Buffet)
    if inputs["order_type"] != "Buffet":
        row[f"Type_of_order_{inputs['order_type']}"] = 1

    # One-hot: Type_of_vehicle (base category dropped: bicycle)
    if inputs["vehicle_type"] != "bicycle":
        row[f"Type_of_vehicle_{inputs['vehicle_type']}"] = 1

    # One-hot: Festival (base category dropped: No)
    if inputs["festival"] == "Yes":
        row["Festival_Yes"] = 1

    # One-hot: City (base category dropped: Metropolitian)
    if inputs["city"] != "Metropolitian":
        row[f"City_{inputs['city']}"] = 1

    # One-hot: Weatherconditions (base category dropped: Cloudy)
    if inputs["weather"] != "Cloudy":
        row[f"Weatherconditions_{inputs['weather']}"] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_minutes(model, scaler, feature_row: pd.DataFrame) -> float:
    scaled = scaler.transform(feature_row)
    pred = model.predict(scaled)
    return float(pred[0])


# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
H("""
<div class="hero">
    <div class="hero-route"><span class="hr-dot"></span><span class="hr-line"></span><span class="hr-rider">🛵</span><span class="hr-dot hr-end"></span></div>
    <h1>🛵 FoodDash: Food Delivery Time Predictor</h1>
    <p>Order from your favourite restaurants and watch your delivery live. Machine learning estimates delivery
    time (in minutes) from order, rider and route conditions — powered by a tuned XGBoost regressor.</p>
    <span class="badge">📊 Test R² ≈ 0.83</span>
    <span class="badge">⏱️ Test RMSE ≈ 3.9 min</span>
    <span class="badge">🌲 XGBoost Regressor</span>
</div>
""")

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error(
        "Model files not found. Make sure `xgb_best.pkl` and `standard_scaler.pkl` "
        "are in the same folder as `app.py`."
    )
    st.stop()


# --------------------------------------------------------------------------
# Helpers: conditions, ETA, cart, bill
# --------------------------------------------------------------------------
def cond_now() -> dict:
    return {**DEFAULT_COND, **st.session_state["cond"]}


def inputs_from_cond(c: dict, rest: dict | None = None) -> dict:
    """Form values -> the `inputs` dict build_feature_row expects (distance recomputed)."""
    rl, ro = (rest["lat"], rest["lon"]) if rest else (c["rest_lat"], c["rest_lon"])
    dist = float(haversine_km(rl, ro, c["del_lat"], c["del_lon"]))
    return dict(
        age=c["age"], ratings=c["ratings"], distance_km=dist, order_hour=c["order_hour"],
        traffic=c["traffic"], vehicle_condition=c["vehicle_condition"],
        multiple_deliveries=c["multiple_deliveries"], order_type=c["order_type"],
        vehicle_type=c["vehicle_type"], festival=c["festival"], city=c["city"], weather=c["weather"],
    )


def eta_for(inputs: dict) -> float:
    return max(predict_minutes(model, scaler, build_feature_row(inputs)), 0)


def cart_rest_obj():
    return REST_BY_NAME.get(st.session_state["cart_rest"]) if st.session_state["cart_rest"] else None


def cart_count() -> int:
    return sum(l["qty"] for l in st.session_state["cart"].values())


def cart_subtotal() -> float:
    return sum(l["qty"] * l["price"] for l in st.session_state["cart"].values())


def route_km() -> float:
    c = cond_now()
    r = cart_rest_obj()
    rl, ro = (r["lat"], r["lon"]) if r else (c["rest_lat"], c["rest_lon"])
    return float(haversine_km(rl, ro, c["del_lat"], c["del_lon"]))


def delivery_fee(km: float) -> float:
    return float(round(20 + 6 * km))


def compute_bill(subtotal: float, km: float, coupon: str | None) -> dict:
    fee = delivery_fee(km) if subtotal else 0.0
    disc = min(subtotal * 0.20, 150.0) if coupon == "SAVE20" else 0.0
    if coupon == "FREEDEL":
        fee = 0.0
    taxes = (subtotal - disc) * 0.05
    return dict(subtotal=subtotal, discount=disc, taxes=taxes, fee=fee,
                total=subtotal - disc + taxes + fee, km=km)


def render_bill(b: dict, coupon: str | None):
    rows = [f'<div class="bill-row"><span>Subtotal</span><span>{inr(b["subtotal"])}</span></div>']
    if b["discount"]:
        rows.append(f'<div class="bill-row save"><span>Coupon {coupon}</span><span>−{inr(b["discount"])}</span></div>')
    rows.append(f'<div class="bill-row"><span>Taxes (5%)</span><span>{inr(b["taxes"])}</span></div>')
    fee_txt = "Free" if (b["fee"] == 0 and coupon == "FREEDEL") else inr(b["fee"])
    rows.append(f'<div class="bill-row"><span>Delivery fee ({b["km"]:.1f} km)</span><span>{fee_txt}</span></div>')
    rows.append(f'<div class="bill-row total"><span>Total</span><span>{inr(b["total"])}</span></div>')
    H("".join(rows))


# ---- callbacks ----
def go(page: str):
    st.session_state["page"] = page


def toggle_fav(name: str):
    favs = st.session_state["fav"]
    favs.symmetric_difference_update({name})


def open_rest(name: str):
    st.session_state["open_rest"] = name


def close_rest():
    st.session_state["open_rest"] = None


def add_item(rname: str, idx: int):
    r = REST_BY_NAME[rname]
    name, price, emoji, veg, addon = r["menu"][idx]
    cheese = bool(addon and st.session_state.get(f"cheese_{r['key']}_{idx}", False))
    if st.session_state["cart"] and st.session_state["cart_rest"] != rname:
        st.session_state["cart"] = {}
        st.session_state["coupon"] = None
        st.toast("Cart cleared: one restaurant per order.", icon="🧺")
    st.session_state["cart_rest"] = rname
    key = name + (" + extra cheese" if cheese else "")
    unit = price + (CHEESE_PRICE if cheese else 0)
    line = st.session_state["cart"].get(key)
    if line:
        line["qty"] += 1
    else:
        st.session_state["cart"][key] = dict(name=key, price=unit, qty=1, emoji=emoji, veg=veg)
    st.toast(f"Added {key}", icon="🛒")


def change_qty(key: str, delta: int):
    line = st.session_state["cart"].get(key)
    if not line:
        return
    line["qty"] += delta
    if line["qty"] <= 0:
        del st.session_state["cart"][key]
    if not st.session_state["cart"]:
        st.session_state["cart_rest"] = None
        st.session_state["coupon"] = None


def apply_coupon():
    code = st.session_state.get("coupon_in", "").strip().upper()
    if code in COUPONS:
        st.session_state["coupon"] = code
        st.toast(f"{code} applied: {COUPONS[code]}", icon="🎟️")
    else:
        st.toast("That coupon code is not valid.", icon="⚠️")


def reorder(order_id: int):
    o = next(x for x in st.session_state["orders"] if x["id"] == order_id)
    st.session_state["cart"] = {k: dict(v) for k, v in o["lines"].items()}
    st.session_state["cart_rest"] = o["rest"]
    st.session_state["coupon"] = None
    st.session_state["page"] = "cart"
    st.toast(f"Items from #{order_id} added to your cart.", icon="🔁")


def logout():
    st.session_state["user"] = None


# --------------------------------------------------------------------------
# Order timeline engine — stage timings shrink as the rider gets closer
# --------------------------------------------------------------------------
def create_order(inputs: dict, minutes: float, feature_row: pd.DataFrame, cond: dict):
    rest = cart_rest_obj()
    total_min = max(minutes, 6.0)
    confirm = max(0.5, 0.04 * total_min)
    prep = 0.34 * total_min
    pack = 0.10 * total_min
    ride = total_min - confirm - prep - pack
    bill = compute_bill(cart_subtotal(), inputs["distance_km"], st.session_state["coupon"])
    st.session_state["order_seq"] += 1
    order = dict(
        id=st.session_state["order_seq"], rest=rest["name"] if rest else "Restaurant",
        lines={k: dict(v) for k, v in st.session_state["cart"].items()},
        bill=bill, coupon=st.session_state["coupon"], T=total_min,
        durs=[confirm, prep, pack, ride], bounds=[confirm, confirm + prep, confirm + prep + pack, total_min],
        t0=time.time(), secs_per_min=SPEEDS[st.session_state["speed_label"]],
        distance=inputs["distance_km"], inputs=inputs, feature_row=feature_row,
        rest_pos=(cond["rest_lat"], cond["rest_lon"]), cust_pos=(cond["del_lat"], cond["del_lon"]),
        rider_start=(cond["rest_lat"] - 0.009, cond["rest_lon"] - 0.007),
        rider_name=RIDER_NAMES[st.session_state["order_seq"] % len(RIDER_NAMES)],
        celebrated=False,
    )
    st.session_state["orders"].insert(0, order)
    st.session_state["cart"] = {}
    st.session_state["cart_rest"] = None
    st.session_state["coupon"] = None
    return order


def lerp(a, b, f):
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def order_state(o: dict, now: float | None = None) -> dict:
    now = now or time.time()
    e = min(max((now - o["t0"]) / o["secs_per_min"], 0.0), o["T"])
    b = o["bounds"]
    idx = 4 if e >= o["T"] else sum(1 for x in b[:3] if e >= x)
    starts = [0.0] + b[:3]
    stage_rem = [max(b[i] - e, 0.0) if i == idx else (b[i] - starts[i] if i > idx else 0.0) for i in range(4)]
    frac_in_stage = 1.0 if idx == 4 else (e - starts[idx]) / max(o["durs"][idx], 1e-6)
    progress = min((idx + frac_in_stage) / 4, 1.0) if idx < 4 else 1.0
    if idx >= 3:
        f = 1.0 if idx == 4 else (e - b[2]) / max(o["durs"][3], 1e-6)
        rider = lerp(o["rest_pos"], o["cust_pos"], f)
        dist_left = o["distance"] * (1 - f)
    else:
        a = e / max(b[2], 1e-6)
        rider = lerp(o["rider_start"], o["rest_pos"], a)
        dist_left = o["distance"]
    to_rest = float(haversine_km(rider[0], rider[1], o["rest_pos"][0], o["rest_pos"][1])) if idx < 3 else 0.0
    return dict(e=e, idx=idx, remaining=max(o["T"] - e, 0.0), stage_rem=stage_rem,
                progress=progress, rider=rider, dist_left=dist_left, to_rest=to_rest)


def dist_text(km: float) -> str:
    return f"{km * 1000:.0f} m" if km < 1 else f"{km:.1f} km"


# --------------------------------------------------------------------------
# Top bar: navigation + account + settings
# --------------------------------------------------------------------------
PAGE_KEYS = ["home", "cart", "checkout", "track", "dash"]


def page_label(k: str) -> str:
    n = cart_count()
    return {
        "home": "🍽️ Restaurants",
        "cart": f"🛒 Cart ({n})" if n else "🛒 Cart",
        "checkout": "🔮 Checkout & Predictor",
        "track": "📍 Track Order",
        "dash": "📊 Dashboard",
    }[k]


nav_c, acc_c, set_c = st.columns([6.2, 1.3, 1.1], gap="small", vertical_alignment="center")
with nav_c:
    st.radio("Navigate", PAGE_KEYS, key="page", horizontal=True,
             format_func=page_label, label_visibility="collapsed")
with acc_c:
    with st.popover("👤 " + (st.session_state["user"] or "Login"), **STRETCH):
        if st.session_state["user"]:
            st.write(f"Signed in as **{st.session_state['user']}**")
            st.button("Log out", on_click=logout, key="logout_btn")
        else:
            st.markdown("**Login or sign up**")
            email = st.text_input("Email", key="login_email")
            pw = st.text_input("Password", type="password", key="login_pw")
            if st.button("Continue", key="login_btn", type="primary"):
                if not re.match(r"^\S+@\S+\.\S+$", email or ""):
                    st.error("Enter a valid email address, like name@example.com.")
                elif len(pw or "") < 6:
                    st.error("Password needs at least 6 characters.")
                else:
                    st.session_state["user"] = email.split("@")[0]
                    st.rerun()
with set_c:
    with st.popover("⚙️", **STRETCH):
        st.toggle("🌙 Dark mode", key="dark")
        st.selectbox("Live tracking speed", list(SPEEDS.keys()), key="speed_label",
                     help="Applies to the next order you place.")


# --------------------------------------------------------------------------
# Page: Restaurants
# --------------------------------------------------------------------------
def keep(keys):
    """Persist widget values while their widget is not on screen."""
    for k in keys:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def card_html(r: dict, eta: float) -> str:
    return (
        f'<div class="rc-top"><span class="rc-emoji">{r["emoji"]}</span><span class="rc-badge">{r["tag"]}</span></div>'
        f'<div class="rc-name">{r["name"]}</div>'
        f'<div class="rc-meta">{r["cuisine"]} cuisine, {"pure veg" if r["veg"] else "veg & non-veg"}</div>'
        f'<div class="chips"><span>⭐ {r["rating"]}</span><span>{inr(r["for_two"])} for two</span>'
        f'<span class="ai">🤖 ~{eta:.0f} min to you</span></div>'
    )


def page_menu(r: dict):
    st.button("← All restaurants", on_click=close_rest, key="back_btn")
    with st.container(key="card_menu_head"):
        H(f'<div class="rc-top"><span class="rc-emoji">{r["emoji"]}</span><span class="rc-badge">{r["tag"]}</span></div>'
          f'<div class="rc-name" style="font-size:1.5rem">{r["name"]}</div>'
          f'<div class="rc-meta">{r["cuisine"]} cuisine · ⭐ {r["rating"]} · {inr(r["for_two"])} for two</div>')
    st.markdown('<div class="section-title">Menu</div>', unsafe_allow_html=True)
    for idx, (name, price, emoji, veg, addon) in enumerate(r["menu"]):
        with st.container(key=f"card_item_{r['key']}_{idx}"):
            c1, c2, c3 = st.columns([4, 2, 1.3], vertical_alignment="center")
            c1.markdown(
                f'<span class="{"dot-veg" if veg else "dot-non"}"></span>'
                f'<span class="item-name">{emoji} {name}</span><br>'
                f'<span class="item-price">{inr(price)}</span>', unsafe_allow_html=True)
            if addon:
                c2.checkbox(f"Extra cheese +{inr(CHEESE_PRICE)}", key=f"cheese_{r['key']}_{idx}")
            c3.button("Add", key=f"add_{r['key']}_{idx}", on_click=add_item, args=(r["name"], idx),
                      **STRETCH)
    if cart_count():
        st.button(f"View cart ({cart_count()} items · {inr(cart_subtotal())})", type="primary",
                  on_click=go, args=("cart",), key="menu_to_cart")
    st.markdown('<div class="section-title" style="margin-top:1rem">What people say</div>', unsafe_allow_html=True)
    for stars, text, who in r["reviews"]:
        H(f'<div class="review">{"⭐" * stars}<br>“{text}” <span class="rc-meta">— {who}</span></div>')


def page_home():
    keep(["f_q", "f_cu", "f_sort", "f_veg", "f_fav"])
    if st.session_state["open_rest"]:
        page_menu(REST_BY_NAME[st.session_state["open_rest"]])
        return

    st.markdown('<div class="section-title">What are you craving today?</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([3, 1.5, 1.5])
    q = f1.text_input("Search", placeholder="Search restaurants or cuisines…",
                      label_visibility="collapsed", key="f_q")
    cuisines = ["All cuisines"] + sorted({r["cuisine"] for r in RESTAURANTS})
    cu = f2.selectbox("Cuisine", cuisines, label_visibility="collapsed", key="f_cu")
    sort_by = f3.selectbox("Sort by", ["Top rated", "Fastest to you", "Price: low to high"],
                           label_visibility="collapsed", key="f_sort")
    t1, t2, _ = st.columns([1, 1.5, 4])
    veg_only = t1.toggle("Veg only", key="f_veg")
    fav_only = t2.toggle("❤️ Favourites only", key="f_fav")

    cond = cond_now()
    etas = {r["name"]: eta_for(inputs_from_cond(cond, r)) for r in RESTAURANTS}

    top = sorted(RESTAURANTS, key=lambda r: -r["rating"])[:3]
    st.caption("Popular near you")
    pc = st.columns(len(top))
    for col, r in zip(pc, top):
        col.button(f"{r['emoji']} {r['name']} · ⭐ {r['rating']}", key=f"pop_{r['key']}",
                   on_click=open_rest, args=(r["name"],), **STRETCH)

    rows = [r for r in RESTAURANTS
            if (cu == "All cuisines" or r["cuisine"] == cu)
            and (not veg_only or r["veg"])
            and (not fav_only or r["name"] in st.session_state["fav"])
            and (q.lower() in (r["name"] + " " + r["cuisine"]).lower())]
    if sort_by == "Top rated":
        rows.sort(key=lambda r: -r["rating"])
    elif sort_by == "Fastest to you":
        rows.sort(key=lambda r: etas[r["name"]])
    else:
        rows.sort(key=lambda r: r["for_two"])

    st.markdown('<div class="section-title" style="margin-top:0.8rem">All restaurants</div>', unsafe_allow_html=True)
    if not rows:
        st.info("No restaurants match these filters. Clear the search or switch off Veg only to see more.")
        return
    for i in range(0, len(rows), 3):
        cols = st.columns(3)
        for col, r in zip(cols, rows[i:i + 3]):
            with col:
                with st.container(key=f"card_rest_{r['key']}"):
                    H(card_html(r, etas[r["name"]]))
                    b1, b2 = st.columns([1, 3])
                    b1.button("❤️" if r["name"] in st.session_state["fav"] else "🤍", key=f"fav_{r['key']}",
                              on_click=toggle_fav, args=(r["name"],))
                    b2.button("View menu", key=f"open_{r['key']}", on_click=open_rest,
                              args=(r["name"],), **STRETCH)


# --------------------------------------------------------------------------
# Page: Cart
# --------------------------------------------------------------------------
def page_cart():
    st.markdown('<div class="section-title">🛒 Your cart</div>', unsafe_allow_html=True)
    cart = st.session_state["cart"]
    if not cart:
        with st.container(key="card_cart_empty"):
            st.markdown("**Your cart is empty.** Pick a restaurant and add something tasty.")
            st.button("Browse restaurants", on_click=go, args=("home",), key="empty_browse", type="primary")
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.caption(f"Ordering from {st.session_state['cart_rest']}")
        for key, line in list(cart.items()):
            with st.container(key=f"card_line_{re.sub(r'[^a-z0-9]+', '_', key.lower())}"):
                c1, c2, c3 = st.columns([3.2, 1.9, 1.3], vertical_alignment="center")
                c1.markdown(
                    f'<span class="{"dot-veg" if line["veg"] else "dot-non"}"></span>'
                    f'<span class="item-name">{line["emoji"]} {key}</span><br>'
                    f'<span class="item-price">{inr(line["price"])} each</span>', unsafe_allow_html=True)
                q1, q2, q3 = c2.columns([1, 1, 1], vertical_alignment="center")
                q1.button("−", key=f"dec_{key}", on_click=change_qty, args=(key, -1))
                q2.markdown(f"**{line['qty']}**")
                q3.button("+", key=f"inc_{key}", on_click=change_qty, args=(key, 1))
                c3.markdown(f"**{inr(line['price'] * line['qty'])}**")
    with right:
        with st.container(key="card_bill"):
            st.markdown("**Coupon**")
            cc1, cc2 = st.columns([2.2, 1])
            cc1.text_input("Coupon code", key="coupon_in", placeholder="Try SAVE20 or FREEDEL",
                           label_visibility="collapsed")
            cc2.button("Apply", on_click=apply_coupon, key="coupon_btn", **STRETCH)
            if st.session_state["coupon"]:
                st.success(f"{st.session_state['coupon']}: {COUPONS[st.session_state['coupon']]}")
            st.markdown("**Bill details**")
            render_bill(compute_bill(cart_subtotal(), route_km(), st.session_state["coupon"]),
                        st.session_state["coupon"])
            st.caption("Delivery fee updates with your exact distance at checkout.")
            st.button("Continue to checkout", type="primary", on_click=go, args=("checkout",), key="cart_checkout")


# --------------------------------------------------------------------------
# Page: Checkout & Predictor (the original prediction form lives here)
# --------------------------------------------------------------------------
def page_checkout():
    cond = cond_now()
    rest = cart_rest_obj()
    cart_ready = bool(st.session_state["cart"])

    if cart_ready:
        with st.container(key="card_order_summary"):
            st.markdown(f"**Order from {rest['name']}** · {cart_count()} items")
            render_bill(compute_bill(cart_subtotal(), route_km(), st.session_state["coupon"]),
                        st.session_state["coupon"])
            st.caption("Your route below starts at this restaurant. Adjust rider, order and conditions, "
                       "then place your order — the model predicts your delivery time.")
    else:
        st.info("Your cart is empty, so you can use the predictor on its own. Add items from "
                "Restaurants to place an order with a live ETA.")

    # Route defaults come from the restaurant in the cart (if any)
    d_rest_lat = rest["lat"] if rest else cond["rest_lat"]
    d_rest_lon = rest["lon"] if rest else cond["rest_lon"]

    with st.form("prediction_form"):

        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            with st.container(key="card_f_rider"):
                st.markdown('<div class="section-title">🧑‍🍳 Delivery Person</div>', unsafe_allow_html=True)
                age = st.slider("Age", min_value=15, max_value=50, value=int(cond["age"]))
                ratings = st.slider("Rating", min_value=1.0, max_value=5.0, value=float(cond["ratings"]), step=0.1)
                vehicle_type = st.selectbox("Vehicle Type", TYPE_OF_VEHICLE_OPTIONS,
                                            index=TYPE_OF_VEHICLE_OPTIONS.index(cond["vehicle_type"]),
                                            format_func=lambda x: x.replace("_", " ").title())
                vehicle_condition = st.select_slider(
                    "Vehicle Condition (0 = worst, 3 = best)", options=[0, 1, 2, 3], value=int(cond["vehicle_condition"])
                )

            with st.container(key="card_f_order"):
                st.markdown('<div class="section-title">📦 Order Details</div>', unsafe_allow_html=True)
                order_type = st.selectbox("Type of Order", TYPE_OF_ORDER_OPTIONS,
                                          index=TYPE_OF_ORDER_OPTIONS.index(cond["order_type"]))
                order_hour = st.slider("Order Hour (24h)", min_value=0, max_value=23, value=int(cond["order_hour"]))
                multiple_deliveries = st.select_slider(
                    "Multiple Deliveries (same trip)", options=[0, 1, 2, 3], value=int(cond["multiple_deliveries"])
                )
                festival = st.radio("Festival Day?", FESTIVAL_OPTIONS,
                                    index=FESTIVAL_OPTIONS.index(cond["festival"]), horizontal=True)

        with col_right:
            with st.container(key="card_f_route"):
                st.markdown('<div class="section-title">📍 Route (Restaurant → Customer)</div>', unsafe_allow_html=True)
                rc1, rc2 = st.columns(2)
                with rc1:
                    rest_lat = st.number_input("Restaurant Latitude", value=float(d_rest_lat), format="%.6f")
                    rest_lon = st.number_input("Restaurant Longitude", value=float(d_rest_lon), format="%.6f")
                with rc2:
                    del_lat = st.number_input("Delivery Latitude", value=float(cond["del_lat"]), format="%.6f")
                    del_lon = st.number_input("Delivery Longitude", value=float(cond["del_lon"]), format="%.6f")
                st.caption("Distance is calculated automatically using the haversine formula (same as training).")

            with st.container(key="card_f_cond"):
                st.markdown('<div class="section-title">🌦️ Conditions</div>', unsafe_allow_html=True)
                weather = st.selectbox("Weather", WEATHER_OPTIONS, index=WEATHER_OPTIONS.index(cond["weather"]))
                traffic = st.selectbox("Road Traffic Density", list(TRAFFIC_MAPPING.keys()),
                                       index=list(TRAFFIC_MAPPING.keys()).index(cond["traffic"]))
                city = st.selectbox("City Type", CITY_OPTIONS, index=CITY_OPTIONS.index(cond["city"]))

        s1, s2 = st.columns(2)
        with s1:
            submitted = st.form_submit_button("🔮 Predict Delivery Time")
        with s2:
            place_clicked = st.form_submit_button(
                "✅ Place order", type="primary", disabled=not cart_ready,
                help=None if cart_ready else "Add items to your cart to place an order.")

    if submitted or place_clicked:
        # Remember the choices so cards, tracking and the dashboard use them
        st.session_state["cond"] = dict(
            age=age, ratings=ratings, vehicle_type=vehicle_type, vehicle_condition=vehicle_condition,
            order_type=order_type, order_hour=order_hour, multiple_deliveries=multiple_deliveries,
            festival=festival, rest_lat=rest_lat, rest_lon=rest_lon, del_lat=del_lat, del_lon=del_lon,
            weather=weather, traffic=traffic, city=city,
        )

        distance_km = haversine_km(rest_lat, rest_lon, del_lat, del_lon)

        if distance_km > 100:
            st.warning(
                f"Calculated distance is {distance_km:.1f} km, which is outside the model's "
                "reliable training range (< 100 km). Double check the coordinates — the "
                "prediction below may not be accurate."
            )

        inputs = dict(
            age=age, ratings=ratings, distance_km=distance_km, order_hour=order_hour,
            traffic=traffic, vehicle_condition=vehicle_condition,
            multiple_deliveries=multiple_deliveries, order_type=order_type,
            vehicle_type=vehicle_type, festival=festival, city=city, weather=weather,
        )

        feature_row = build_feature_row(inputs)
        predicted_minutes = predict_minutes(model, scaler, feature_row)
        predicted_minutes = max(predicted_minutes, 0)

        if place_clicked:
            if distance_km > 100:
                st.error("This address is too far to deliver to. Check the coordinates and try again.")
            else:
                create_order(inputs, predicted_minutes, feature_row, st.session_state["cond"])
                st.session_state["_goto"] = "track"
                st.rerun()

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Delivery Time</div>
            <div class="result-minutes">{predicted_minutes:.0f} min</div>
            <div class="result-label">Distance: {distance_km:.2f} km &nbsp;•&nbsp; Model: XGBoost Regressor</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("See the exact feature values sent to the model"):
            st.dataframe(feature_row.T.rename(columns={0: "value"}), **STRETCH)


# --------------------------------------------------------------------------
# Page: Track Order (live)
# --------------------------------------------------------------------------
def render_tracker(o: dict):
    s = order_state(o)
    idx = s["idx"]
    inp = o["inputs"]
    remaining_min = math.ceil(s["remaining"] - 1e-9)

    msgs = [
        f"Order confirmed by {o['rest']}",
        f"{o['rest']} is preparing your food",
        "Packing your order",
        f"{o['rider_name']} is on the way to you",
        "Delivered. Enjoy your meal!",
    ]
    if idx == 4:
        sub = "Thanks for ordering with FoodDash."
    elif idx == 3:
        sub = f"{o['rider_name']} is {dist_text(s['dist_left'])} away. The timer drops as the rider gets closer."
    else:
        sub = f"{o['rider_name']} is {dist_text(s['to_rest'])} from the restaurant, heading in for pickup."
    big = "Delivered" if idx == 4 else f"{remaining_min} min"
    lbl = "Order complete" if idx == 4 else "Arriving in"
    H(f'<div class="eta-banner"><div><div class="eta-lbl">{lbl} · Order #{o["id"]}</div><div class="eta-big">{big}</div></div>'
      f'<div class="eta-msg">{STAGE_ICONS[idx]} {msgs[idx]}<small>{sub}</small></div></div>')

    nodes = []
    for i, (nm, ic) in enumerate(zip(STAGES, STAGE_ICONS)):
        cls = "done" if (i < idx or idx == 4) else "active" if i == idx else "todo"
        if i < 4:
            if cls == "done":
                small = "done"
            elif cls == "active":
                small = f"{math.ceil(s['stage_rem'][i] - 1e-9)} min left"
            else:
                small = f"~{max(1, round(o['durs'][i]))} min"
        else:
            small = "arrived" if cls == "done" else ""
        nodes.append(f'<div class="stp-n {cls}"><b>{"✓" if cls == "done" else ic}</b>{nm}<small>{small}</small></div>')
    H(f'<div class="stp"><div class="stp-track"><i style="width:{s["progress"] * 100:.1f}%"></i></div>{"".join(nodes)}</div>')

    mc, dc = st.columns([3, 2], gap="large")
    with mc:
        rest_pos, cust_pos = o["rest_pos"], o["cust_pos"]
        df = pd.DataFrame([
            dict(lat=rest_pos[0], lon=rest_pos[1], color="#F2A20C", size=70),
            dict(lat=cust_pos[0], lon=cust_pos[1], color="#1E8E5A", size=70),
            dict(lat=s["rider"][0], lon=s["rider"][1], color="#2563EB", size=110),
        ])
        try:
            st.map(df, latitude="lat", longitude="lon", color="color", size="size")
        except TypeError:
            st.map(df[["lat", "lon"]])
        st.caption("Orange: restaurant · Blue: your rider · Green: you")
    with dc:
        with st.container(key=f"card_rider_{o['id']}"):
            H(f'<div class="rider"><span class="av">🧑‍🍳</span><div><b>{o["rider_name"]}</b><br>'
              f'<span class="rc-meta">{inp["age"]} yrs · ⭐ {inp["ratings"]} · '
              f'{inp["vehicle_type"].replace("_", " ").title()} (condition {inp["vehicle_condition"]}/3)</span></div></div>')
            H(f'<div class="chips"><span>🌦️ {inp["weather"]}</span><span>🚦 {inp["traffic"]} traffic</span>'
              f'<span>🏙️ {inp["city"]}</span>{"<span>🎉 Festival day</span>" if inp["festival"] == "Yes" else ""}</div>')
            k1, k2 = st.columns(2)
            k1.markdown(f'<div class="kpi"><span>Distance to you</span><b>{dist_text(s["dist_left"])}</b></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi"><span>Model ETA</span><b>{o["T"]:.0f} min</b></div>', unsafe_allow_html=True)
        with st.container(key=f"card_ordersum_{o['id']}"):
            st.markdown(f"**{o['rest']}**")
            for k, line in o["lines"].items():
                st.write(f"{line['qty']} × {k}")
            render_bill(o["bill"], o["coupon"])


if hasattr(st, "fragment"):
    @st.fragment(run_every=1.0)
    def live_tracker(order_id: int):
        o = next(x for x in st.session_state["orders"] if x["id"] == order_id)
        render_tracker(o)
        if order_state(o)["idx"] == 4:
            st.rerun()  # full rerun: stops the timer and shows the finished state
else:
    def live_tracker(order_id: int):
        o = next(x for x in st.session_state["orders"] if x["id"] == order_id)
        render_tracker(o)
        st.button("🔄 Refresh", key="refresh_btn")


def page_track():
    orders = st.session_state["orders"]
    st.markdown('<div class="section-title">📍 Live order tracking</div>', unsafe_allow_html=True)
    if not orders:
        with st.container(key="card_track_empty"):
            st.markdown("**No orders yet.** Place an order and your rider will appear here in real time.")
            st.button("Browse restaurants", on_click=go, args=("home",), key="track_browse", type="primary")
        return

    ids = [o["id"] for o in orders]
    chosen = ids[0]
    if len(ids) > 1:
        chosen = st.selectbox("Order", ids, format_func=lambda i: f"#{i}", key="track_pick")
    o = next(x for x in orders if x["id"] == chosen)

    if order_state(o)["idx"] == 4:
        render_tracker(o)
        if not o["celebrated"]:
            o["celebrated"] = True
            st.balloons()
        if hasattr(st, "feedback"):
            st.caption("How was your delivery?")
            st.feedback("stars", key=f"stars_{o['id']}")
    else:
        live_tracker(o["id"])

    with st.expander("See the exact feature values sent to the model"):
        st.dataframe(o["feature_row"].T.rename(columns={0: "value"}), **STRETCH)

    st.markdown('<div class="section-title" style="margin-top:1.2rem">Order history</div>', unsafe_allow_html=True)
    for h in orders:
        hs = order_state(h)
        with st.container(key=f"card_hist_{h['id']}"):
            c1, c2, c3 = st.columns([4, 1.6, 1.4], vertical_alignment="center")
            items = ", ".join(f"{l['qty']}× {k}" for k, l in h["lines"].items())
            c1.markdown(f"**#{h['id']} · {h['rest']}**  \n<span class='rc-meta'>{items}</span>", unsafe_allow_html=True)
            c2.markdown(f"**{inr(h['bill']['total'])}**  \n<span class='rc-meta'>{'Delivered' if hs['idx'] == 4 else STAGES[hs['idx']]}</span>",
                        unsafe_allow_html=True)
            c3.button("🔁 Reorder", key=f"re_{h['id']}", on_click=reorder, args=(h["id"],), **STRETCH)


# --------------------------------------------------------------------------
# Page: Dashboard
# --------------------------------------------------------------------------
def page_dash():
    st.markdown('<div class="section-title">📊 Analytics dashboard</div>', unsafe_allow_html=True)
    orders = st.session_state["orders"]

    st.caption("Your session")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders placed", len(orders))
    k2.metric("Total spent", inr(sum(o["bill"]["total"] for o in orders)))
    k3.metric("Avg predicted ETA", f"{np.mean([o['T'] for o in orders]):.0f} min" if orders else "–")
    k4.metric("Favourites", len(st.session_state["fav"]))

    st.caption("Platform analytics (sample data)")
    a1, a2 = st.columns(2, gap="large")
    with a1:
        with st.container(key="card_d_hours"):
            st.markdown("**Orders by hour** · most orders land between 7 and 9 PM")
            st.bar_chart(pd.DataFrame({"orders": SAMPLE_HOURLY}, index=range(24)), color="#F2A20C")
    with a2:
        with st.container(key="card_d_rev"):
            st.markdown("**Revenue this week (₹ thousands)**")
            st.line_chart(pd.Series(SAMPLE_WEEK_REVENUE, name="revenue"), color="#1E8E5A")
    a3, a4 = st.columns(2, gap="large")
    with a3:
        with st.container(key="card_d_dishes"):
            st.markdown("**Top-selling dishes**")
            st.bar_chart(pd.Series(SAMPLE_TOP_DISHES, name="orders"), color="#D8452B")
    with a4:
        with st.container(key="card_d_status"):
            st.markdown("**Order status**")
            st.bar_chart(pd.Series({"Delivered": 86, "In progress": 9, "Cancelled": 5}, name="%"), color="#10241E")

    st.caption("Model insights (live predictions from your trained XGBoost model)")
    base = inputs_from_cond(cond_now(), cart_rest_obj() or REST_BY_NAME["Spice Route"])
    m1, m2 = st.columns(2, gap="large")
    with m1:
        with st.container(key="card_m_hour"):
            st.markdown("**Predicted delivery time by order hour**")
            hrs = {h: eta_for({**base, "order_hour": h}) for h in range(24)}
            st.line_chart(pd.Series(hrs, name="minutes"), color="#F2A20C")
    with m2:
        with st.container(key="card_m_traffic"):
            st.markdown("**Predicted delivery time by traffic**")
            tr = {t: eta_for({**base, "traffic": t}) for t in TRAFFIC_MAPPING}
            st.bar_chart(pd.Series(tr, name="minutes"), color="#10241E")
    with st.container(key="card_m_weather"):
        st.markdown("**Predicted delivery time by weather**")
        we = {w: eta_for({**base, "weather": w}) for w in WEATHER_OPTIONS}
        st.bar_chart(pd.Series(we, name="minutes"), color="#2563EB")
        st.caption(f"Based on a {base['distance_km']:.1f} km route and your current rider and order settings.")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
{"home": page_home, "cart": page_cart, "checkout": page_checkout,
 "track": page_track, "dash": page_dash}[st.session_state["page"]]()

# --------------------------------------------------------------------------
# Footer with social icons
# --------------------------------------------------------------------------
st.markdown(f"""
<div class="footer">
    <div class="footer-icons">
        <a href="{SOCIAL_LINKS['GitHub']}" target="_blank" title="GitHub">
            <svg viewBox="0 0 24 24"><path d="M12 .5C5.73.5.98 5.24.98 11.52c0 5.02 3.26 9.28 7.77 10.78.57.1.78-.25.78-.55 0-.27-.01-1-.02-1.96-3.16.69-3.83-1.52-3.83-1.52-.52-1.31-1.26-1.66-1.26-1.66-1.03-.7.08-.69.08-.69 1.14.08 1.74 1.17 1.74 1.17 1.01 1.74 2.65 1.24 3.3.95.1-.73.4-1.24.72-1.53-2.52-.29-5.17-1.26-5.17-5.6 0-1.24.44-2.25 1.17-3.04-.12-.29-.51-1.45.11-3.02 0 0 .96-.31 3.14 1.16a10.9 10.9 0 0 1 5.72 0c2.18-1.47 3.14-1.16 3.14-1.16.62 1.57.23 2.73.11 3.02.73.79 1.17 1.8 1.17 3.04 0 4.35-2.65 5.31-5.18 5.59.41.35.77 1.04.77 2.1 0 1.52-.01 2.74-.01 3.11 0 .3.2.66.79.55A11.03 11.03 0 0 0 23.02 11.5C23.02 5.24 18.27.5 12 .5z"/></svg>
        </a>
        <a href="{SOCIAL_LINKS['Kaggle']}" target="_blank" title="Kaggle">
            <svg viewBox="0 0 24 24"><path d="M18.83 21.06h-3.4c-.13 0-.25-.06-.33-.16l-4.99-6.22-1.4 1.34v4.71c0 .18-.15.33-.33.33H5.62a.33.33 0 0 1-.33-.33V2.94c0-.18.15-.33.33-.33h2.76c.18 0 .33.15.33.33v11.6l5.94-6.03a.5.5 0 0 1 .36-.15h3.53c.27 0 .41.33.22.53l-6.2 6.13 6.63 8.5a.33.33 0 0 1-.26.54z"/></svg>
        </a>
        <a href="{SOCIAL_LINKS['LinkedIn']}" target="_blank" title="LinkedIn">
            <svg viewBox="0 0 24 24"><path d="M20.45 20.45h-3.55v-5.57c0-1.33-.02-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.95v5.66H9.36V9h3.41v1.56h.05c.47-.9 1.63-1.85 3.36-1.85 3.6 0 4.27 2.37 4.27 5.45v6.29zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45z"/></svg>
        </a>
    </div>
    <div class="footer-text">Built by Abdul Rehman &nbsp;•&nbsp; BS Data Science &nbsp;•&nbsp; Powered by Streamlit + XGBoost</div>
</div>
""", unsafe_allow_html=True)
