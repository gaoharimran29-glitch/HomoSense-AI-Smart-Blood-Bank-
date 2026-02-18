import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import plotly.graph_objects as go
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import pydeck as pdk

load_dotenv()

# ================= CREDENTIALS =================
EMAIL = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")

if not EMAIL or not APP_PASSWORD:
    st.error("Email credentials not configured properly.")
    st.stop()

# ================= CONFIG & THEME =================
st.set_page_config(page_title="HomoSense AI | Control Center", layout="wide", initial_sidebar_state="collapsed")

BLOOD_GROUPS = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]
MODELS_FOLDER = r"model/Hosp_A_XGB_Results"
CURRENT_STOCK_FILE = r"dataset/current_stock.csv"
NEAREST_HOSPITAL_FILE = r"dataset/nearest_hospitals.csv"

# UI CSS
st.markdown("""
    <style>
    html, body, [class*="css"] { font-size: 18px; }
    .blood-box {
        padding: 25px; border-radius: 15px; margin-bottom: 20px; color: white;
        text-align: center; box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
        height: 280px; display: flex; flex-direction: column; justify-content: center;
    }
    .blood-box h2 { font-size: 45px !important; margin: 0; }
    .blood-box h4 { font-size: 26px !important; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.3); }
    .blood-box p { font-size: 20px !important; font-weight: 500; margin: 5px 0; }
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; height: 55px; font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

# ================= SESSION STATES =================
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "modal_bg" not in st.session_state: st.session_state.modal_bg = None
if "email_feedback" not in st.session_state: st.session_state.email_feedback = None
if "run_login_alerts" not in st.session_state: st.session_state.run_login_alerts = False

# ================= HELPERS =================

def build_features(df):
    df = df.sort_values("Date").copy()
    df["Lag_1"], df["Lag_3"], df["Lag_7"] = df["UnitsUsed"].shift(1), df["UnitsUsed"].shift(3), df["UnitsUsed"].shift(7)
    df["RollingMean_7"] = df["UnitsUsed"].rolling(7).mean()
    df["DemandVolatility_7"] = df["UnitsUsed"].rolling(7).std()
    df["GrowthRate"] = (df["Lag_1"] - df["Lag_3"]) / (df["Lag_3"] + 1)
    df["ExpiryPressure"] = df["UnitsExpired"] / (df["UnitsAvailable"] + 1)
    df["StockCoverageDays"] = df["UnitsAvailable"] / (df["RollingMean_7"] + 1)
    df["DayOfWeek"], df["Month"] = df["Date"].dt.dayofweek, df["Date"].dt.month
    return df

feature_cols = ["RollingMean_7","Lag_1","Lag_3","Lag_7","GrowthRate","DemandVolatility_7","ExpiryPressure","StockCoverageDays","EmergencyCases","TrafficIndex","AvgTemperature","HolidayFlag","WeekendFlag","DayOfWeek","Month"]

def play_alert_sound():
    st.markdown("""<audio autoplay style="display:none;"><source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg"></audio>""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    models, metrics = {}, {}
    for bg in BLOOD_GROUPS:
        m_path = os.path.join(MODELS_FOLDER, bg, f"{bg}_xgb_model.pkl")
        met_path = os.path.join(MODELS_FOLDER, bg, f"{bg}_metrics.json")
        if os.path.exists(m_path): models[bg] = joblib.load(m_path)
        if os.path.exists(met_path):
            with open(met_path, "r") as f: metrics[bg] = json.load(f)
    return models, metrics

models, metrics_dict = load_models()

def fetch_and_store_data():
    df = pd.read_csv(CURRENT_STOCK_FILE)
    df["Date"] = pd.to_datetime(df["Date"], format='mixed')
    st.session_state.stable_df = df

def add_random_data():
    try:
        current_data = pd.read_csv(CURRENT_STOCK_FILE)
        new_rows = []
        for blood in BLOOD_GROUPS:
            scenario = np.random.choice(["safe", "shortage", "expiry", "both"])
            if scenario == "safe": avail, exp = np.random.randint(80, 150), 0
            elif scenario == "shortage": avail, exp = np.random.randint(5, 20), 0
            elif scenario == "expiry": avail, exp = np.random.randint(50, 80), np.random.randint(5, 12)
            else: avail, exp = np.random.randint(5, 20), np.random.randint(2, 6)
            
            new_rows.append({
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "HospitalID": "HOSP_A", "BloodGroup": blood, "UnitsAvailable": avail,
                "UnitsCollected": np.random.randint(5, 20), "UnitsUsed": np.random.randint(5, 15),
                "UnitsExpired": exp, "EmergencyCases": np.random.randint(1, 8),
                "TrafficIndex": round(np.random.uniform(0.3, 0.9), 2),
                "AvgTemperature": 28.5, "HolidayFlag": 0, "WeekendFlag": 0
            })
        updated_df = pd.concat([current_data, pd.DataFrame(new_rows)], ignore_index=True)
        updated_df.to_csv(CURRENT_STOCK_FILE, index=False)
        st.toast("🚀 Database updated!", icon="🩸")
    except Exception as e: st.error(f"Sync Error: {e}")

# ================= LOGIN SHIELD =================
if not st.session_state.authenticated:
    st.title("🩸 Hospital Admin Login")
    st.write("Username: admin | Password: admin123")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "admin123":
            st.session_state.authenticated = True
            st.session_state.run_login_alerts = True # Set flag for dashboard
            st.rerun()
    st.stop()

# ================= PROTECTED DASHBOARD =================
if st.session_state.email_feedback:
    fb = st.session_state.email_feedback
    if fb["success"]:
        st.toast(f"✅ Email sent to {fb['hospital']} successfully!", icon="📧")
        play_alert_sound()
    else:
        st.error(f"❌ Failed to send email: {fb['message']}")
        play_alert_sound()
    
    st.session_state.email_feedback = None

if "stable_df" not in st.session_state: fetch_and_store_data()

# 1. HEADER
hcol1, hcol2 = st.columns([6, 4])
hcol1.title("🛡️ HomoSense AI Control Center")
with hcol2:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("➕ Add Data"): add_random_data()
    if c2.button("🔄 Fetch Again", type="primary"):
        fetch_and_store_data()
        st.rerun()
    if c3.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

# --- HACKATHON DEMO GUIDE ---
with st.expander("*Quick Demo Guide for Judges**", expanded=True):
    st.markdown("""
    Welcome to the **HomoSense AI** Demo. Follow these steps to see the AI in action:
    1.  **Click [➕ Add Data]:** This simulates real-time hospital logs by generating new rows in the CSV.
    2.  **Click [🔄 Fetch Again]:** This triggers the data pipeline, refreshes the UI, and runs the **XGBoost AI models** on the new data.
    3.  **Observe Indicators:** Notice how the boxes turn <span style='color:#e74c3c; font-weight:bold;'>RED (Shortage)</span> or <span style='color:#f39c12; font-weight:bold;'>ORANGE (Expiry)</span> based on AI predictions.
    4.  **Supply Route:** Click **🔍 Action** on any shortage group and then scroll to see the Delhi-NCR supply optimization map.
    5.  **Email Alerts:** From the map view, click **📧 Request Transfer** to simulate sending an email alert to partner hospitals.
    6.  I added my own email now as partner hospital email for demo purpose. Therefore email comes to me only now. I added email screenshots below.
    7.  **Analytics:** Scroll down to see the AI performance charts and metrics.
    8.  **Github:** Check github repo for detailed documentation https://github.com/gaoharimran29-glitch/HomoSense-AI-Smart-Blood-Bank-
    9. **Logout:** Click the logout button to end the session.
    """, unsafe_allow_html=True)

# 3. GRID DASHBOARD
st.divider()
data = st.session_state.stable_df
for i in range(0, len(BLOOD_GROUPS), 4):
    cols = st.columns(4)
    for j, bg in enumerate(BLOOD_GROUPS[i:i+4]):
        if bg not in models: continue
        bg_df = data[data["BloodGroup"] == bg]
        latest = bg_df.iloc[-1]
        
        # Prediction for color logic
        fe_df = build_features(bg_df).fillna(0)
        pred_val = int(models[bg].predict(fe_df[feature_cols].iloc[-1:])[0])
        
        shortage = latest["UnitsAvailable"] < pred_val
        expiry = latest["UnitsExpired"] > 0
        bg_col, icon, txt = "#2ecc71", "✅", "SAFE"
        if shortage: bg_col, icon, txt = "#c0392b", "⚠️", "SHORTAGE"
        if expiry: bg_col, icon, txt = "#f39c12", "🕒", "EXPIRY"
        if shortage and expiry: bg_col, icon, txt = "#e74c3c", "🚨", "CRITICAL"

        with cols[j]:
            st.markdown(f"""<div class="blood-box" style="background-color:{bg_col};">
                <h4>{bg} Group</h4><h2>{latest['UnitsAvailable']} U</h2>
                <p>Forecast: {pred_val} U</p><p><b>{icon} {txt}</b></p></div>""", unsafe_allow_html=True)
            if txt != "SAFE":
                if st.button(f"🔍 Action: {bg}", key=f"btn_{bg}"):
                    st.session_state.modal_bg = bg
                    st.rerun()

# ================= HOSPITAL SEARCH SECTION =================
def send_email_smtp(to_email, subject, body_html):

    if not EMAIL or not APP_PASSWORD:
        return False, "Email credentials missing."

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(EMAIL, APP_PASSWORD)
            server.sendmail(EMAIL, to_email, msg.as_string())

        return True, "Email Sent Successfully"

    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Check app password."
    except Exception as e:
        return False, str(e)

if "email_feedback" not in st.session_state:
    st.session_state.email_feedback = None

stock_df = pd.read_csv(CURRENT_STOCK_FILE)
stock_df["Date"] = pd.to_datetime(stock_df["Date"], format='mixed')
nearest_df = pd.read_csv(NEAREST_HOSPITAL_FILE)

st.markdown('<div id="resource_section"></div>', unsafe_allow_html=True)

# ================= NEW MAP & ACTION SECTION =================
if st.session_state.modal_bg:
    bg = st.session_state.modal_bg
    df = st.session_state.stable_df
    st.markdown("---")

    # DATA CONTEXT
    bg_df = df[df["BloodGroup"] == bg]
    bg_data = bg_df.iloc[-1]

    current_units = bg_data["UnitsAvailable"]
    expired_units = bg_data["UnitsExpired"]

    fe_df = build_features(bg_df).fillna(0)
    X = fe_df[feature_cols].iloc[-1:]
    pred = models[bg].predict(X)[0]
    is_shortage = current_units < pred

    st.markdown("<div id='opti'></div>", unsafe_allow_html=True)
    st.subheader(f"🚚 Supply Route Optimization: {bg}")

    # HOME HUB COORDINATES
    home_lat, home_lon = 28.6129 , 77.2090

    # CLEAN NEAREST DATA
    nearest_df["BloodGroup"] = nearest_df["BloodGroup"].astype(str).str.strip().str.upper()
    nearest_df["UnitsAvailable"] = pd.to_numeric(nearest_df["UnitsAvailable"], errors="coerce")
    nearest_df["Lat"] = pd.to_numeric(nearest_df["Lat"], errors="coerce")
    nearest_df["Lon"] = pd.to_numeric(nearest_df["Lon"], errors="coerce")

    session_key = f"nearby_{bg}"
    
    # Clean the data FIRST so filtering works on correct types
    nearest_df["BloodGroup"] = nearest_df["BloodGroup"].astype(str).str.strip().str.upper()
    nearest_df["UnitsAvailable"] = pd.to_numeric(nearest_df["UnitsAvailable"], errors="coerce")
    nearest_df["Lat"] = pd.to_numeric(nearest_df["Lat"], errors="coerce")
    nearest_df["Lon"] = pd.to_numeric(nearest_df["Lon"], errors="coerce")
    
    # Always re-filter when the modal opens to ensure accuracy
    nearby_filtered = nearest_df[
        (nearest_df["BloodGroup"] == bg.upper()) & 
        (nearest_df["UnitsAvailable"] > 0)
    ].dropna(subset=["Lat", "Lon"]).copy()

    # Add a visible color for the scatter points (RGBA)
    nearby_filtered["color"] = [[255, 0, 0, 160] for _ in range(len(nearby_filtered))]
    
    # Store in session state
    st.session_state[session_key] = nearby_filtered
    nearby = st.session_state[session_key]

    nearby = st.session_state[session_key]

    # BUILD MAP DATA
    # Hub dataframe
    hub_df = pd.DataFrame({
        "HospitalID": ["HOSPITAL A (Main Hub)"],
        "Lat": [float(home_lat)],
        "Lon": [float(home_lon)],
        "color": [[0, 100, 255, 255]]  # Solid Blue
    })

    # PyDeck Layers
    hub_layer = pdk.Layer(
        "ScatterplotLayer",
        data=hub_df,
        get_position=["Lon", "Lat"],
        get_color="color",
        get_radius=300,
        pickable=True,
    )

    # PyDeck Layers
    hub_layer = pdk.Layer(
        "ScatterplotLayer",
        data=hub_df,
        get_position=["Lon", "Lat"],
        get_color="color",
        get_radius=300,
        pickable=True,
    )

    hospital_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nearby,
        get_position=["Lon", "Lat"],
        get_color="color",
        get_radius=200,
        pickable=True,
    )

    # Automatically center the map on the results if they exist
    view_lat = nearby["Lat"].mean() if not nearby.empty else home_lat
    view_lon = nearby["Lon"].mean() if not nearby.empty else home_lon

    deck = pdk.Deck(
        map_style="light",
        layers=[hub_layer, hospital_layer],
        initial_view_state=pdk.ViewState(
            latitude=view_lat, 
            longitude=view_lon, 
            zoom=10, 
            pitch=0
        ),
        tooltip={"text": "{HospitalID}\nStock: {UnitsAvailable}"}
    )

    # UI LAYOUT
    col_list, col_map = st.columns([1, 1.2])
    with col_map:
        st.pydeck_chart(deck)

    with col_list:
        st.write(f"### 🏥 Found {len(nearby)} Partners")
        for index, row in nearby.iterrows():
            with st.expander(f"📍 {row['HospitalID']} ({row.get('Distance_km','N/A')} km)"):
                st.write(f"Stock: **{row['UnitsAvailable']} Units**")

                # EMAIL LOGIC
                if is_shortage:
                    btn_label = "📧 Request Transfer"
                    subject = f"URGENT: Blood Request {bg}"
                    body_msg = f"""
                    <html>
                        <body style="font-family: Arial; background:#f4f6f9; padding:20px;">
                            <div style="background:white; padding:25px; border-radius:10px;">
                                <h2 style="color:#d63031;">🩸 Blood Shortage Alert</h2>
                                <p>Dear Admin,</p>
                                <p>Our AI system predicts a shortage of <b>{bg}</b> blood units.</p>
                                <ul>
                                    <li><b>Current Units:</b> {current_units}</li>
                                    <li><b>Predicted Demand:</b> {int(pred)}</li>
                                    <li><b>Distance:</b> {row.get('Distance_km','N/A')} km</li>
                                </ul>
                                <p>We kindly request immediate transfer support.</p>
                                <hr>
                                <p style="color:gray;">HomoSense AI System (Hospital A)</p>
                            </div>
                        </body>
                    </html>
                    """
                else:
                    btn_label = "📧 Offer Stock"
                    subject = f"IMPORTANT: Stock Offer {bg}"
                    body_msg = f"""
                    <html>
                    <body style="font-family: Arial; background:#f4f6f9; padding:20px;">
                        <div style="background:white; padding:25px; border-radius:10px;">
                            <h2 style="color:#0984e3;">🩸 Blood Stock Offer</h2>
                            <p>Dear Admin,</p>
                            <p>We currently have <b>{expired_units}</b> units of <b>{bg}</b> nearing expiry.</p>
                            <p>We are offering this stock to avoid wastage.</p>
                            <hr>
                            <p style="color:gray;">HomoSense AI System (Hospital A)</p>
                        </div>
                    </body>
                    </html>
                    """

                if st.button(btn_label, key=f"btn_{bg}_{index}"):
                    with st.spinner("Sending..."):
                        success, msg = send_email_smtp(row["Email"], subject, body_msg)
                        st.session_state.email_feedback = {
                            "success": success,
                            "message": msg,
                            "hospital": row["HospitalID"]
                        }
                        
                        st.rerun()

    if st.button("✅ Close Optimization View"):
        st.session_state.modal_bg = None
        st.rerun()

# --- Email demo screenshots ---
st.markdown("---") # Visual separator

# --- STATIC EMAIL PROOF SECTION ---
st.header("📧 Email Alert Demonstration")
st.write("Our AI system triggers real-time email notifications to hospitals and blood banks. Below is the confirmation of the automated alerts.")

col1, col2 = st.columns(2)

with col1:
    st.info("**Stock offer to save blood wastage**")
    try:
        st.image(r"Screenshots/Stock Offer.jpeg", 
                 caption="Alert received by hospital for stock offer.",width=400)
    except:
        st.warning("Sent email screenshot missing in 'screenshots/' folder.")

with col2:
    st.info("**Urgent Request due to Blood Shortage**")
    try:
        st.image(r"Screenshots/Urgent Blood Request.jpeg", 
                 caption="Alert received by the hospital with shortage details.", width=400)
    except:
        st.warning("Received email screenshot missing in 'screenshots/' folder.")


# ================= ANALYTICS =================
st.markdown("---")
st.header("📊 AI Performance Analytics (30-Day Window)")
for bg in BLOOD_GROUPS:
    if bg not in models: continue
    bg_df = stock_df[stock_df["BloodGroup"]==bg]
    fe_df = build_features(bg_df).fillna(0)
    X_plot = fe_df[feature_cols].iloc[-30:]
    y_actual, y_pred = fe_df["UnitsUsed"].iloc[-30:], models[bg].predict(X_plot)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_actual.values, name="Actual Usage", line=dict(color='#00d2ff', width=3)))
    fig.add_trace(go.Scatter(y=y_pred, name="XGBoost Forecast", line=dict(color='#ff4b4b', width=3, dash='dot')))
    fig.update_layout(title=f"{bg} Demand Tracking", template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    if bg in metrics_dict:
        m = metrics_dict[bg]
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE (Error)", f"{m['RMSE']:.2f}")
        m2.metric("MAE (Avg Error)", f"{m['MAE']:.2f}")
        m3.metric("R² (Model Fit)", f"{m['R2']:.3f}")
    st.markdown("<br><br>", unsafe_allow_html=True)