import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model import rf_model, xgb_model

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="EduTrack - Academic Performance",
    page_icon="🎓",
    layout="wide"
)

# ===== THEME & STYLE =====
st.markdown("""
    <style>
        body { font-family: 'Lexend', sans-serif; background-color: #F2F2F2; }
        .sidebar .sidebar-content { background-color: #FFFFFF; }
        .main-title { font-size: 2rem; font-weight: bold; }
        .metric-card { background-color: #FFFFFF; border-radius: 12px; padding: 20px; border: 1px solid #E0E0E0; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.title("EduTrack")

# --- Profile Picture Upload ---
st.sidebar.markdown("### Upload Your Profile Picture")
profile_pic = st.sidebar.file_uploader("Choose a profile picture", type=["png", "jpg", "jpeg"])
if profile_pic:
    st.sidebar.image(profile_pic, width=100)
else:
    st.sidebar.info("No profile picture uploaded")

st.sidebar.markdown("**Ishan**  \nB.Tech AIML")
if st.sidebar.button("Log Out"):
    st.warning("You have been logged out!")

# ===== NAVIGATION TABS =====
tabs = ["Dashboard", "Academics"]
selected_tab = st.sidebar.radio("Navigate", tabs)

# ===== DATA VARIABLES =====
uploaded_file = None
df = None

# ===== DASHBOARD TAB =====
if selected_tab == "Dashboard":
    st.markdown("<p class='main-title'>Student Overview</p>", unsafe_allow_html=True)
    
    st.info("Upload CSV in Academics tab to see detailed data")
    
    # Display default/basic data (if CSV not uploaded yet)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Overall GPA", "3.85")
    with col2: st.metric("Class Rank", "8th")
    with col3: st.metric("Attendance Rate", "98%")

# ===== ACADEMICS TAB =====
elif selected_tab == "Academics":
    st.markdown("<p class='main-title'>Academic Performance</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=['float64', 'int64'])
        
        # --- Metrics ---
        col1, col2, col3 = st.columns(3)
        overall_gpa = numeric_cols.get('GPA', pd.Series([0])).mean()
        attendance_rate = numeric_cols.get('Attendance', pd.Series([0])).mean()
        class_rank = numeric_cols.get('Rank', pd.Series([0])).mean()
        with col1: st.metric("Overall GPA", f"{overall_gpa:.2f}")
        with col2: st.metric("Class Rank", f"{int(class_rank)}")
        with col3: st.metric("Attendance Rate", f"{attendance_rate:.0f}%")

        # --- Model Predictions ---
        if rf_model and xgb_model and not numeric_cols.empty:
            try:
                input_data = numeric_cols.values
                rf_pred = rf_model.predict(input_data)
                xgb_pred = xgb_model.predict(input_data)
                avg_pred = (rf_pred + xgb_pred) / 2
                st.subheader("📊 Predicted Performance (Average GPA)")
                st.dataframe(pd.DataFrame(avg_pred, columns=["Predicted GPA"]))
            except Exception as e:
                st.warning(f"Model prediction failed: {e}")

        # --- GPA Over Time Chart ---
        st.markdown("### 📈 GPA Over Time")
        if "GPA" in df.columns:
            gpa_values = df["GPA"]
        else:
            gpa_values = avg_pred if 'avg_pred' in locals() else numeric_cols.iloc[:, 0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=gpa_values, mode='lines+markers', line=dict(color="#4A90E2", width=3)))
        fig.update_layout(height=300, xaxis_title="Semester", yaxis_title="GPA", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # --- Subject Performance ---
        st.markdown("### 🧮 Performance by Subject")
        for col in numeric_cols.columns:
            if col.lower() not in ["gpa", "attendance", "rank"]:
                score = numeric_cols[col].mean()
                st.write(f"**{col}** - {score:.1f}%")
                st.progress(min(int(score), 100))

        # --- Assignments (if available) ---
        if all(col in df.columns for col in ["Assignment", "Subject", "Due Date", "Score", "Status"]):
            st.markdown("### 🗂️ Recent Assignments")
            st.dataframe(df[["Assignment", "Subject", "Due Date", "Score", "Status"]], use_container_width=True)
