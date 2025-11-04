import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from io import StringIO

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="EduTrack - Academic Performance",
    page_icon="🎓",
    layout="wide"
)

# ========== THEME & STYLE ==========
st.markdown("""
    <style>
        body {
            font-family: 'Lexend', sans-serif;
            background-color: #F2F2F2;
        }
        .sidebar .sidebar-content {
            background-color: #FFFFFF;
        }
        .main-title {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #E0E0E0;
            text-align: center;
        }
        .upload-box {
            border: 2px dashed #E0E0E0;
            border-radius: 12px;
            padding: 30px;
            background-color: #FFFFFF;
            text-align: center;
        }
        .progress-bar {
            background-color: #E0E0E0;
            border-radius: 9999px;
            height: 10px;
            width: 100%;
        }
        .progress-fill {
            background-color: #4A90E2;
            height: 10px;
            border-radius: 9999px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
st.sidebar.title("EduTrack")
st.sidebar.image("https://lh3.googleusercontent.com/aida-public/AB6AXuAV9EHmnsMRFUfV3A4yF1NOFhqLjnWoRloJoWyyiPsEJYi5ShWVfObqJJ1svtah3273Srb1gzyVzTRA23QrgN6EoTIGKqIpuCxdJhGVc-8iWNFpIcOpU176B4BqYmlC28ZjHPPV7MkTKZGF8boixd2IATiUA0ajwJleVoN9HCz-8ND6VWRd9Z5CkDKF9pY8h9FGpuTHXMxp_9ZNw_ozJ3FZnk_CCkZUV3gruNRduOizr9cYf-M_vL2_eKCtnFpAhKXuLArG6HU2UdCk", width=80)
st.sidebar.markdown("**Ishan**  \nB.Tech AIML")

tabs = ["Dashboard", "Academics", "Sports", "Calendar", "Messages", "Settings"]
selected_tab = st.sidebar.radio("Navigate", tabs)
st.sidebar.markdown("---")
st.sidebar.button("Log Out")

# ========== LOAD MODELS ==========
with open("C:\Users\dtcgr\Desktop\CodeSmasher\Student_Performance_Prediction\rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("C:\Users\dtcgr\Desktop\CodeSmasher\Student_Performance_Prediction\xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# ========== TAB CONTENT ==========
if selected_tab == "Academics":
    st.markdown("<p class='main-title'>Academic Performance</p>", unsafe_allow_html=True)
    st.write("A detailed overview of David's academic progress.")

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall GPA", "3.85", "+0.12 from last semester")
    with col2:
        st.metric("Class Rank", "8th", "+2 from last semester")
    with col3:
        st.metric("Attendance Rate", "98%", "-1% from last semester")

    # --- File Upload for CSV ---
    st.markdown("### 📤 Upload Performance Records")
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head())

        # Example: Predict GPA using both models
        try:
            rf_pred = rf_model.predict(df)
            xgb_pred = xgb_model.predict(df)
            avg_pred = (rf_pred + xgb_pred) / 2
            st.subheader("📊 Predicted Performance (Average GPA)")
            st.write(avg_pred)
        except Exception as e:
            st.warning("Model prediction failed. Ensure CSV columns match model training features.")

    # --- GPA Over Time Chart ---
    st.markdown("### 📈 GPA Over Time")
    gpa_values = [3.7, 3.8, 3.6, 3.4, 3.9, 3.5, 3.85]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=gpa_values, mode='lines+markers', line=dict(color="#4A90E2", width=3)))
    fig.update_layout(
        height=300,
        xaxis_title="Semester",
        yaxis_title="GPA",
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Subject Performance ---
    st.markdown("### 🧮 Performance by Subject")
    subject_scores = {
        "Mathematics": 92,
        "English": 85,
        "Science": 88,
        "History": 78,
        "Art": 98
    }
    for subject, score in subject_scores.items():
        st.write(f"**{subject}** - {score}%")
        st.progress(score / 100)

    # --- Recent Assignments ---
    st.markdown("### 🗂️ Recent Assignments")
    assignments = pd.DataFrame({
        "Assignment": [
            "Algebra II: Chapter 5 Test",
            "The Great Gatsby Essay",
            "Photosynthesis Lab Report",
            "World War II Presentation",
            "Impressionism Study Sketch"
        ],
        "Subject": ["Mathematics", "English", "Science", "History", "Art"],
        "Due Date": ["May 20, 2024", "May 18, 2024", "May 15, 2024", "May 25, 2024", "May 28, 2024"],
        "Score": ["95/100", "88/100", "72/100", "--/--", "--/--"],
        "Status": ["Graded", "Graded", "Graded", "Submitted", "Upcoming"]
    })
    st.dataframe(assignments, use_container_width=True)

elif selected_tab == "Dashboard":
    st.title("📊 Dashboard Overview")
    st.write("Quick summary of your academic and extracurricular progress.")

elif selected_tab == "Sports":
    st.title("🏅 Sports Activity")
    st.write("Track your sports achievements and attendance here.")

elif selected_tab == "Calendar":
    st.title("🗓️ Academic Calendar")
    st.write("View upcoming classes, exams, and events.")

elif selected_tab == "Messages":
    st.title("✉️ Messages")
    st.write("Check your messages and announcements here.")

elif selected_tab == "Settings":
    st.title("⚙️ Settings")
    st.write("Manage your account, preferences, and themes here.")

