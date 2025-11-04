import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from model import rf_model, xgb_model

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="EduTrack - Student Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ===== THEME & STYLE =====
st.markdown("""
    <style>
        body { font-family: 'Lexend', sans-serif; background-color: #F4F6F9; }
        .main-title { font-size: 2.6rem; font-weight: bold; color: #2C3E50; margin-bottom: 20px; }
        .metric-card { background-color: #FFFFFF; border-radius: 15px; padding: 25px; 
                       border: 1px solid #E0E0E0; text-align: center; 
                       box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        .metric-value { font-size: 1.8rem; font-weight: bold; color: #34495E; }
        .metric-label { font-size: 1rem; color: #7F8C8D; }
        .progress-label { font-weight: bold; margin-top: 10px; margin-bottom: 5px; }
        .section-title { font-size: 1.6rem; font-weight: bold; color: #2E86C1; margin-top: 25px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.title("EduTrack")

# --- Profile Picture Upload ---
st.sidebar.markdown("### Upload Your Profile Picture")
profile_pic = st.sidebar.file_uploader("Choose a profile picture", type=["png", "jpg", "jpeg"])
if profile_pic:
    st.sidebar.image(profile_pic, width=120)
else:
    st.sidebar.info("No profile picture uploaded")

st.sidebar.markdown("**Ishan**  \nB.Tech AIML")
if st.sidebar.button("Log Out"):
    st.warning("You have been logged out!")

# ===== DASHBOARD =====
st.markdown("<p class='main-title'>📊 Student Performance Dashboard</p>", unsafe_allow_html=True)

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV file with student data", type=["csv"])
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")
    st.dataframe(df.head(), use_container_width=True)

# --- Metrics Cards ---
numeric_cols = df.select_dtypes(include=['float64', 'int64']) if df is not None else None
overall_gpa = numeric_cols.get('GPA', pd.Series([0])).mean() if numeric_cols is not None else 0
class_rank = numeric_cols.get('Rank', pd.Series([0])).mean() if numeric_cols is not None else 0
attendance_rate = numeric_cols.get('Attendance', pd.Series([0])).mean() if numeric_cols is not None else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{overall_gpa:.2f}</div>
            <div class='metric-label'>Overall GPA</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{int(class_rank)}</div>
            <div class='metric-label'>Class Rank</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{attendance_rate:.0f}%</div>
            <div class='metric-label'>Attendance Rate</div>
        </div>
    """, unsafe_allow_html=True)

# --- Performance Graph (GPA Hike) ---
if df is not None and "GPA" in df.columns:
    st.markdown("<div class='section-title'>📈 GPA Performance Over Semesters</div>", unsafe_allow_html=True)
    semesters = list(range(1, len(df["GPA"])+1))
    gpa_values = df["GPA"].tolist()
    gpa_change = [0] + [round(gpa_values[i]-gpa_values[i-1],2) for i in range(1,len(gpa_values))]

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=semesters,
        y=gpa_values,
        mode='lines+markers+text',
        text=[f"{chg:+}" for chg in gpa_change],
        textposition="top center",
        line=dict(color="#2E86C1", width=4),
        marker=dict(size=10, color="#F39C12")
    ))
    fig_perf.update_layout(
        height=400,
        xaxis_title="Semester",
        yaxis_title="GPA",
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="#F4F6F9"
    )
    st.plotly_chart(fig_perf, use_container_width=True)

# --- Ratio Graph (Toppers vs Other Students) ---
if df is not None and "GPA" in df.columns:
    st.markdown("<div class='section-title'>🏆 Topper vs Other Students Ratio</div>", unsafe_allow_html=True)
    topper_threshold = st.slider("Set Topper GPA Threshold", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    num_toppers = (df["GPA"] >= topper_threshold).sum()
    num_others = len(df) - num_toppers

    ratio_df = pd.DataFrame({
        "Category": ["Toppers", "Other Students"],
        "Count": [num_toppers, num_others]
    })

    fig_ratio = px.pie(
        ratio_df, 
        names="Category", 
        values="Count", 
        color="Category",
        color_discrete_map={"Toppers":"#F39C12", "Other Students":"#2E86C1"},
        hole=0.4
    )
    fig_ratio.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_ratio, use_container_width=True)

# --- Subject Performance ---
if numeric_cols is not None:
    st.markdown("<div class='section-title'>🧮 Subject Performance</div>", unsafe_allow_html=True)
    subject_cols = [col for col in numeric_cols.columns if col.lower() not in ["gpa", "attendance", "rank"]]
    subject_scores = numeric_cols[subject_cols].mean() if subject_cols else pd.Series()

    col1, col2 = st.columns([2,1])
    with col1:
        if not subject_scores.empty:
            fig_subj = px.bar(
                subject_scores, 
                x=subject_scores.index, 
                y=subject_scores.values, 
                text=subject_scores.values,
                labels={'x':'Subject', 'y':'Score'},
                color=subject_scores.values,
                color_continuous_scale='Blues'
            )
            fig_subj.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=20, b=20),
                coloraxis_showscale=False,
                yaxis=dict(range=[0,100])
            )
            st.plotly_chart(fig_subj, use_container_width=True)
    
    with col2:
        st.markdown("### Progress Bars")
        for subj, score in subject_scores.items():
            st.markdown(f"<div class='progress-label'>{subj}: {score:.1f}%</div>", unsafe_allow_html=True)
            st.progress(min(int(score),100))

# --- Model Predictions ---
if rf_model and xgb_model and numeric_cols is not None and not numeric_cols.empty:
    try:
        input_data = numeric_cols.values
        rf_pred = rf_model.predict(input_data)
        xgb_pred = xgb_model.predict(input_data)
        avg_pred = (rf_pred + xgb_pred) / 2
        st.markdown("<div class='section-title'>📊 Predicted Performance (Average GPA)</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(avg_pred, columns=["Predicted GPA"]), use_container_width=True)
    except Exception as e:
        st.warning(f"Model prediction failed: {e}")
