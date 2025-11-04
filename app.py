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
        .sidebar .sidebar-content { background-color: #FFFFFF; padding: 20px; }
        .main-title { font-size: 2.2rem; font-weight: bold; color: #2E86C1; }
        .metric-card { background-color: #FFFFFF; border-radius: 12px; padding: 20px; border: 1px solid #E0E0E0; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);}
        .progress-label { font-weight: bold; }
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

    # Display placeholders (dynamic later after CSV upload)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Overall GPA", "0.0")
    with col2: st.metric("Class Rank", "0")
    with col3: st.metric("Attendance Rate", "0%")

# ===== ACADEMICS TAB =====
elif selected_tab == "Academics":
    st.markdown("<p class='main-title'>Academic Performance</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head(), use_container_width=True)

        numeric_cols = df.select_dtypes(include=['float64', 'int64'])

        # --- Dynamic Dashboard Metrics ---
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
                st.dataframe(pd.DataFrame(avg_pred, columns=["Predicted GPA"]), use_container_width=True)
            except Exception as e:
                st.warning(f"Model prediction failed: {e}")

        # --- GPA Over Time Chart ---
        st.markdown("### 📈 GPA Over Time")
        if "GPA" in df.columns:
            gpa_values = df["GPA"]
        else:
            gpa_values = avg_pred if 'avg_pred' in locals() else numeric_cols.iloc[:, 0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=gpa_values,
            mode='lines+markers',
            line=dict(color="#2E86C1", width=4),
            marker=dict(size=10, color="#F39C12")
        ))
        fig.update_layout(
            height=350,
            xaxis_title="Semester",
            yaxis_title="GPA",
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="#F2F2F2"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Subject Performance ---
        st.markdown("### 🧮 Performance by Subject")
        for col in numeric_cols.columns:
            if col.lower() not in ["gpa", "attendance", "rank"]:
                score = numeric_cols[col].mean()
                st.markdown(f"<div class='progress-label'>{col}: {score:.1f}%</div>", unsafe_allow_html=True)
                st.progress(min(int(score), 100))

        # --- Recent Assignments (if available) ---
        if all(col in df.columns for col in ["Assignment", "Subject", "Due Date", "Score", "Status"]):
            st.markdown("### 🗂️ Recent Assignments")
            st.dataframe(df[["Assignment", "Subject", "Due Date", "Score", "Status"]], use_container_width=True)
