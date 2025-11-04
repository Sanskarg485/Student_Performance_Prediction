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
st.sidebar.image(
    "https://lh3.googleusercontent.com/aida-public/AB6AXuAV9EHmnsMRFUfV3A4yF1NOFhqLjnWoRloJoWyyiPsEJYi5ShWVfObqJJ1svtah3273Srb1gzyVzTRA23QrgN6EoTIGKqIpuCxdJhGVc-8iWNFpIcOpU176B4BqYmlC28ZjHPPV7MkTKZGF8boixd2IATiUA0ajwJleVoN9HCz-8ND6VWRd9Z5CkDKF9pY8h9FGpuTHXMxp_9ZNw_ozJ3FZnk_CCkZUV3gruNRduOizr9cYf-M_vL2_eKCtnFpAhKXuLArG6HU2UdCk",
    width=80
)
st.sidebar.markdown("**Ishan**  \nB.Tech AIML")

tabs = ["Dashboard", "Academics", "Sports", "Calendar", "Messages", "Settings"]
selected_tab = st.sidebar.radio("Navigate", tabs)
st.sidebar.markdown("---")
if st.sidebar.button("Log Out"):
    st.warning("You have been logged out!")

# ===== TAB CONTENT =====
uploaded_file = None
df = None
if selected_tab == "Academics":
    st.markdown("<p class='main-title'>Academic Performance</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head())

# ===== DISPLAY DATA AND METRICS =====
if uploaded_file and df is not None:
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])

    if selected_tab == "Academics":
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

        # --- Assignments (Dynamic from CSV if available) ---
        if all(col in df.columns for col in ["Assignment", "Subject", "Due Date", "Score", "Status"]):
            st.markdown("### 🗂️ Recent Assignments")
            st.dataframe(df[["Assignment", "Subject", "Due Date", "Score", "Status"]], use_container_width=True)

    elif selected_tab == "Dashboard":
        st.title("📊 Dashboard Overview")
        st.metric("Average GPA", f"{numeric_cols.get('GPA', pd.Series([0])).mean():.2f}")
        st.metric("Subjects Count", len(numeric_cols.columns))

    elif selected_tab == "Sports":
        st.title("🏅 Sports Activity")
        if "Sports" in df.columns:
            st.dataframe(df[["Sports"]])
        else:
            st.info("No sports data found in CSV.")

    elif selected_tab == "Calendar":
        st.title("🗓️ Academic Calendar")
        if all(col in df.columns for col in ["Event", "Date"]):
            st.dataframe(df[["Event", "Date"]])
        else:
            st.info("No calendar events found in CSV.")

    elif selected_tab == "Messages":
        st.title("✉️ Messages")
        if all(col in df.columns for col in ["Message", "Date"]):
            st.dataframe(df[["Date", "Message"]])
        else:
            st.info("No messages found in CSV.")

    elif selected_tab == "Settings":
        st.title("⚙️ Settings")
        st.write("Manage your account, preferences, and themes here.")

else:
    if selected_tab == "Academics":
        st.info("Upload a CSV in Academics tab to display all metrics and charts.")
