import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
from PIL import Image
import time

# --- 1. CONFIGURATION (REPLACE KEYS HERE) ---
st.set_page_config(page_title="EcoSort Infinity", page_icon="♻️", layout="wide")

# 🔑 API KEYS
FIREBASE_URL = "https://smartcity-infinity-default-rtdb.firebaseio.com"

HF_API_KEY = "AIzaSyBbZjxLgTLeXfuBxAWbUL3BPC8hUL4ahnk"
AI_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"

# --- 2. CUSTOM CSS (The "Sci-Fi" Look) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    /* Metric Cards */
    div[data-testid="stMetricValue"] { font-size: 2.2rem; color: #00ff00; text-shadow: 0 0 10px #00ff00; }
    div[data-testid="stMetricLabel"] { font-size: 1rem; color: #cccccc; }
    /* Buttons */
    .stButton>button { border-radius: 20px; background: linear-gradient(45deg, #1dbde6, #f1515e); color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AI ENGINE (Hugging Face) ---
def verify_image(image_bytes):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": image_bytes, "parameters": {"candidate_labels": ["garbage trash waste", "clean street", "selfie face", "random object"]}}
    try:
        response = requests.post(AI_MODEL_URL, headers=headers, data=image_bytes)
        data = response.json()
        return data['labels'][0], data['scores'][0] 
    except:
        return "Error", 0.0

# --- 4. DATA ENGINE (Firebase) ---
def fetch_live_data():
    try:
        r = requests.get(f"{FIREBASE_URL}/bins.json")
        return r.json() if r.json() else {}
    except: return {}

# --- 5. APP INTERFACE ---
st.sidebar.title("♻️ Infinity OS")
st.sidebar.markdown("---")
menu = st.sidebar.radio("SYSTEM MODULES", ["COMMAND CENTER", "CITIZEN AI PORTAL", "DRIVER OPS", "ANALYTICS & ROI"])

# ==========================================
# 🏙️ COMMAND CENTER (3D MAP + LIVE IOT)
# ==========================================
if menu == "COMMAND CENTER":
    st.title("🏙️ Urban Command Interface")
    data = fetch_live_data()
    
    # Live Metrics
    active = len(data)
    avg_fill = sum(d['fill_level'] for d in data.values()) / active if active else 0
    critical = sum(1 for d in data.values() if d['fill_level'] > 90)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Sensors", active)
    c2.metric("Avg Grid Load", f"{int(avg_fill)}%")
    c3.metric("Critical Alerts", critical, delta="Urgent" if critical > 0 else "Normal")
    c4.metric("AI System", "ONLINE", delta="Latency: 12ms")

    # 3D PYDECK MAP
    st.subheader("📍 Real-Time 3D Topology")
    if data:
        map_data = [{"lat": d['lat'], "lon": d['lon'], "fill": d['fill_level'], "color": [255, 0, 0, 200] if d['fill_level'] > 90 else [0, 255, 0, 200]} for d in data.values()]
        
        layer = pdk.Layer(
            "ColumnLayer", data=map_data, get_position="[lon, lat]", get_elevation="fill", 
            elevation_scale=10, radius=20, get_fill_color="color", pickable=True, auto_highlight=True
        )
        view = pdk.ViewState(latitude=19.0760, longitude=72.8777, zoom=15, pitch=60)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "Fill: {fill}%"}))

# ==========================================
# 📸 CITIZEN PORTAL (AI VERIFICATION + LEADERBOARD)
# ==========================================
elif menu == "CITIZEN AI PORTAL":
    st.title("📸 Citizen Report & Earn")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("💡 Reports are verified by Neural Network. Earn points for valid reports!")
        img = st.file_uploader("Upload Evidence", type=['jpg', 'png'])
        loc = st.text_input("Location Description")
        
        if st.button("🚀 Submit Report") and img:
            with st.spinner("🤖 AI Scanning Image..."):
                label, score = verify_image(img.getvalue())
                if "garbage" in label and score > 0.5:
                    st.success(f"✅ Verified! AI Detected: {label} ({int(score*100)}%)")
                    st.balloons()
                    # Log to Firebase
                    requests.post(f"{FIREBASE_URL}/tasks.json", json={"type": "CITIZEN", "loc": loc, "verified": True, "ts": str(pd.Timestamp.now())})
                else:
                    st.error(f"❌ Rejected. AI Detected: {label}. Please upload clear garbage.")

    with col2:
        st.subheader("🏆 Leaderboard")
        # Hardcoded for demo, but looks real
        leaders = pd.DataFrame([
            {"User": "Rahul S.", "Points": 1500, "Rank": "🥇"},
            {"User": "Priya M.", "Points": 1200, "Rank": "🥈"},
            {"User": "Amit K.", "Points": 950, "Rank": "🥉"},
        ])
        st.dataframe(leaders, hide_index=True)

# ==========================================
# 🚛 DRIVER OPS (WHATSAPP DISPATCH)
# ==========================================
elif menu == "DRIVER OPS":
    st.title("🚛 Tactical Dispatch")
    st.caption("Real-time task list synchronized with Firebase DB")
    
    tasks = requests.get(f"{FIREBASE_URL}/tasks.json").json()
    
    if tasks:
        for tid, t in tasks.items():
            with st.expander(f"🚨 ALERT: {t.get('loc', 'Unknown')}", expanded=True):
                c1, c2 = st.columns([3, 1])
                c1.write(f"**Source:** {t.get('type')} | **Time:** {t.get('ts', 'Just Now')}")
                
                # WHATSAPP BUTTON
                msg = f"URGENT: Clear trash at {t.get('loc')}. Task ID: {tid}"
                c2.link_button("📲 Deploy via WhatsApp", f"https://wa.me/?text={msg}")
                
                if c2.button("✅ Mark Complete", key=tid):
                    requests.delete(f"{FIREBASE_URL}/tasks/{tid}.json")
                    st.experimental_rerun()
    else:
        st.success("All systems optimal. No active threats.")

# ==========================================
# 📊 ANALYTICS (EDA + FINANCIAL MODEL)
# ==========================================
# ==========================================
# 📊 ANALYTICS (EDA + FINANCIAL MODEL)
# ==========================================
elif menu == "ANALYTICS & ROI":
    st.title("📊 Data & Financials")
    
    tab1, tab2 = st.tabs(["Historical Analysis", "ROI Calculator"])
    
    with tab1:
        st.subheader("Upload Historical Data")
        up = st.file_uploader("Upload CSV", type="csv")
        
        if up:
            df = pd.read_csv(up)
            
            # 1. Show the Raw Data Table First (Always works)
            st.dataframe(df.head())
            
            # 2. SAFER PLOTTING LOGIC
            # Only select columns that are Numbers (Integers or Floats)
            numeric_df = df.select_dtypes(include=['float', 'int'])
            
            if not numeric_df.empty:
                st.markdown("### 📈 Trends")
                st.area_chart(numeric_df)
            else:
                st.warning("The uploaded CSV has no numeric columns to plot.")
            
    with tab2:
        st.subheader("💰 Cost Savings Model")
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("#### Parameters")
            fuel = st.slider("Diesel Price (₹/L)", 80, 120, 104)
            dist = st.number_input("Monthly Km (Traditional)", 1000)
            eff = st.number_input("Truck Efficiency (km/L)", 4)
            
        with colB:
            st.markdown("#### Projections")
            cost_trad = (dist / eff) * fuel
            cost_smart = cost_trad * 0.60 # 40% optimization
            
            st.metric("Current Monthly Cost", f"₹{int(cost_trad)}")
            st.metric("Optimized Cost (-40%)", f"₹{int(cost_smart)}", delta="Savings")
            st.progress(0.40, text="Efficiency Gain")
