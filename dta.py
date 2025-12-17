# =============================================================================
# 📊 COMPREHENSIVE DATA ANALYSIS & VISUALIZATION PLATFORM
# =============================================================================
# Developer: Dr. Merwan Roudane
# Specialization: Econometrics, Time Series Analysis, Statistical Software Development
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import (
    shapiro, normaltest, jarque_bera, anderson, kstest,
    pearsonr, spearmanr, kendalltau,
    ttest_1samp, ttest_ind, ttest_rel,
    mannwhitneyu, wilcoxon, kruskal,
    f_oneway, levene, bartlett,
    chi2_contingency, fisher_exact
)
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import io
import base64
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# =============================================================================

st.set_page_config(
    page_title="Data Analysis Platform | Dr. Merwan Roudane",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful BRILLIANT light design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Nunito:wght@300;400;600;700;800&family=Fira+Code:wght@400;500&display=swap');

    /* Main app styling - BRIGHT background */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 25%, #fef3f2 50%, #f0fdf4 75%, #fffbeb 100%);
        font-family: 'Nunito', sans-serif;
    }

    /* Header styling - BRILLIANT gradient */
    .main-header {
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 25%, #ec4899 50%, #f97316 75%, #06b6d4 100%);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        padding: 2.5rem 3rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 50px rgba(139, 92, 246, 0.35);
        position: relative;
        overflow: hidden;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.25) 0%, transparent 70%);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }

    .main-header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }

    .main-header p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.95);
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    .developer-badge {
        background: rgba(255,255,255,0.25);
        padding: 0.6rem 1.4rem;
        border-radius: 30px;
        display: inline-block;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .developer-badge span {
        color: #fef08a;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }

    /* Section cards - Brilliant accent */
    .section-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.1);
        margin-bottom: 1.5rem;
        border-left: 6px solid;
        border-image: linear-gradient(180deg, #06b6d4, #8b5cf6, #ec4899) 1;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .section-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(139, 92, 246, 0.2);
    }

    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        background: linear-gradient(135deg, #06b6d4, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #f0f0f0;
    }

    /* Info boxes - Brilliant cyan */
    .info-box {
        background: linear-gradient(135deg, #ecfeff 0%, #cffafe 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 5px solid #06b6d4;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.15);
    }

    .info-box-title {
        font-weight: 700;
        color: #0e7490;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    .info-box-content {
        color: #155e75;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Success box - Brilliant emerald */
    .success-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 5px solid #10b981;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.15);
    }

    /* Warning box - Brilliant amber */
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 5px solid #f59e0b;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.15);
    }

    /* Metric cards - Brilliant rainbow top border */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899, #f97316, #10b981);
    }

    .metric-value {
        font-family: 'Fira Code', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    /* Sidebar styling - Brilliant gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }

    [data-testid="stSidebar"] label {
        color: rgba(255,255,255,0.95) !important;
    }

    /* Button styling - Brilliant violet */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 50%, #c084fc 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.5);
    }

    /* Download button - Brilliant emerald */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #ffffff;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }

    /* Tab styling - Brilliant colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #f0f9ff 0%, #fef3f2 50%, #f0fdf4 100%);
        padding: 0.5rem;
        border-radius: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        color: #64748b;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        font-weight: 700;
        color: #475569;
    }

    /* Table styling */
    .dataframe {
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
    }

    /* Footer - Brilliant gradient */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%);
        border-radius: 20px;
        color: #ffffff;
        box-shadow: 0 10px 40px rgba(139, 92, 246, 0.3);
    }

    /* Code styling */
    code {
        font-family: 'Fira Code', monospace;
        background: linear-gradient(135deg, #f0f9ff 0%, #fef3f2 100%);
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        color: #8b5cf6;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Scrollbar styling - Brilliant */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #8b5cf6 0%, #ec4899 100%);
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #7c3aed 0%, #db2777 100%);
    }

    /* Additional brilliant styling */
    .stMultiSelect > div > div {
        border-radius: 12px;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER SECTION
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>📊 Data Analysis & Visualization Platform</h1>
    <p>A comprehensive statistical analysis toolkit for researchers, data scientists, and analysts</p>
    <div class="developer-badge">
        Developed by <span>Dr. Merwan Roudane</span> | Econometrician & Statistical Software Developer
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_download_link(df, filename, file_format='xlsx'):
    """Create download link for dataframe"""
    if file_format == 'xlsx':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        return f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
    else:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'data:text/csv;base64,{b64}'


def interpret_pvalue(p, alpha=0.05):
    """Interpret p-value with detailed explanation"""
    if p < 0.001:
        return "Very Strong Evidence (p < 0.001)", "🔴"
    elif p < 0.01:
        return "Strong Evidence (p < 0.01)", "🟠"
    elif p < alpha:
        return f"Significant (p < {alpha})", "🟡"
    else:
        return f"Not Significant (p ≥ {alpha})", "🟢"


def get_normality_interpretation(stat, p, test_name):
    """Get interpretation for normality test"""
    interp, emoji = interpret_pvalue(p)
    if p >= 0.05:
        conclusion = "The data appears to follow a normal distribution (fail to reject H₀)"
    else:
        conclusion = "The data significantly deviates from normality (reject H₀)"
    return f"{emoji} **{test_name}**: Statistic = {stat:.4f}, p-value = {p:.4f}\n\n*Interpretation*: {interp}\n\n*Conclusion*: {conclusion}"


# =============================================================================
# SIDEBAR - FILE UPLOAD & SETTINGS
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #ffffff; font-family: Poppins, sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>⚙️ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # File Upload Section
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your Excel file (.xlsx)",
        type=['xlsx', 'xls'],
        help="Upload an Excel file containing your data. The first row should contain column headers."
    )

    st.markdown("---")

    # Global Settings
    st.markdown("### 🎨 Display Settings")

    color_palette = st.selectbox(
        "Color Palette",
        ["viridis", "plasma", "inferno", "magma", "cividis",
         "Blues", "Greens", "Reds", "Purples", "Oranges",
         "coolwarm", "RdYlBu", "RdYlGn", "Spectral", "seismic"],
        index=0,
        help="Select the color palette for visualizations"
    )

    significance_level = st.slider(
        "Significance Level (α)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="The threshold for statistical significance (commonly 0.05)"
    )

    decimal_places = st.slider(
        "Decimal Places",
        min_value=2,
        max_value=8,
        value=4,
        help="Number of decimal places for displaying results"
    )

    st.markdown("---")

    # About Section
    st.markdown("### 📖 About")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 14px; font-size: 0.85rem; backdrop-filter: blur(10px);'>
        <p style='color: #ffffff; font-weight: 600;'>
        This platform provides comprehensive tools for:
        </p>
        <ul style='color: rgba(255,255,255,0.95); font-size: 0.8rem;'>
            <li>Descriptive Statistics</li>
            <li>Data Transformations</li>
            <li>Normality Testing</li>
            <li>Correlation Analysis</li>
            <li>Hypothesis Testing</li>
            <li>Advanced Visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

if uploaded_file is not None:
    # Load Data
    try:
        df = pd.read_excel(uploaded_file)

        # Data Info Display
        st.markdown("""
        <div class="section-card">
            <div class="section-title">✅ Data Successfully Loaded</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Observations</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-label">Variables</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(numeric_cols)}</div>
                <div class="metric-label">Numeric</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(categorical_cols)}</div>
                <div class="metric-label">Categorical</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =============================================================================
        # MAIN TABS
        # =============================================================================

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Data Preview",
            "📊 Descriptive Statistics",
            "🔄 Transformations",
            "📈 Normality Tests",
            "🔗 Correlation Analysis",
            "🧪 Hypothesis Tests",
            "🎨 Visualizations"
        ])

        # =============================================================================
        # TAB 1: DATA PREVIEW
        # =============================================================================

        with tab1:
            st.markdown("""
            <div class="info-box">
                <div class="info-box-title">📋 What is Data Preview?</div>
                <div class="info-box-content">
                    Data preview allows you to inspect your dataset before analysis. It's essential to understand 
                    the structure, types, and potential issues (missing values, outliers) in your data before 
                    proceeding with any statistical analysis.
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### 🔍 Dataset Preview")
                n_rows = st.slider("Number of rows to display", 5, 100, 20)
                st.dataframe(df.head(n_rows), use_container_width=True, height=400)

            with col2:
                st.markdown("### 📌 Variable Types")
                dtype_df = pd.DataFrame({
                    'Variable': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Missing': df.isnull().sum().values,
                    'Missing %': np.round(df.isnull().sum().values / len(df) * 100, 2)
                })
                st.dataframe(dtype_df, use_container_width=True, height=400)

            # Missing Values Analysis
            st.markdown("### 🔴 Missing Values Analysis")

            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)

            if missing_data.sum() > 0:
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_pct.values,
                    labels={'x': 'Variables', 'y': 'Missing Percentage (%)'},
                    title='Missing Values by Variable',
                    color=missing_pct.values,
                    color_continuous_scale=color_palette
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family='Source Sans Pro')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values detected in the dataset!")

            # Download Options
            st.markdown("### 💾 Download Data")
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="data_export.csv",
                    mime="text/csv"
                )
            with col2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                output.seek(0)
                st.download_button(
                    label="📥 Download as Excel",
                    data=output,
                    file_name="data_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        # =============================================================================
        # TAB 2: DESCRIPTIVE STATISTICS
        # =============================================================================

        with tab2:
            st.markdown("""
            <div class="info-box">
                <div class="info-box-title">📊 What are Descriptive Statistics?</div>
                <div class="info-box-content">
                    Descriptive statistics summarize and describe the main features of a dataset. They include 
                    measures of central tendency (mean, median, mode), dispersion (variance, standard deviation, 
                    range, IQR), and shape (skewness, kurtosis). These statistics help you understand the 
                    distribution and characteristics of your data before conducting inferential analysis.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if len(numeric_cols) > 0:
                # Variable Selection
                selected_vars = st.multiselect(
                    "Select Variables for Analysis",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))],
                    help="Choose one or more numeric variables to analyze"
                )

                if selected_vars:
                    # Basic Statistics
                    st.markdown("### 📈 Basic Statistics")

                    stats_df = df[selected_vars].describe().T
                    stats_df['variance'] = df[selected_vars].var()
                    stats_df['skewness'] = df[selected_vars].skew()
                    stats_df['kurtosis'] = df[selected_vars].kurtosis()
                    stats_df['iqr'] = stats_df['75%'] - stats_df['25%']
                    stats_df['range'] = stats_df['max'] - stats_df['min']
                    stats_df['cv'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)

                    # Reorder columns
                    stats_df = stats_df[
                        ['count', 'mean', 'std', 'variance', 'min', '25%', '50%', '75%', 'max', 'range', 'iqr',
                         'skewness', 'kurtosis', 'cv']]
                    stats_df.columns = ['Count', 'Mean', 'Std Dev', 'Variance', 'Min', 'Q1 (25%)', 'Median (50%)',
                                        'Q3 (75%)', 'Max', 'Range', 'IQR', 'Skewness', 'Kurtosis', 'CV (%)']

                    st.dataframe(stats_df.round(decimal_places), use_container_width=True)

                    # Statistics Explanation
                    with st.expander("📚 Understanding the Statistics"):
                        st.markdown("""
                        **Measures of Central Tendency:**
                        - **Mean**: The arithmetic average of all values
                        - **Median (50%)**: The middle value when data is sorted; robust to outliers

                        **Measures of Dispersion:**
                        - **Standard Deviation (Std Dev)**: Average distance from the mean
                        - **Variance**: Square of standard deviation; measures data spread
                        - **Range**: Difference between maximum and minimum values
                        - **IQR (Interquartile Range)**: Range of the middle 50% of data; robust to outliers
                        - **CV (Coefficient of Variation)**: Relative variability as percentage of mean

                        **Measures of Shape:**
                        - **Skewness**: Measures asymmetry (0 = symmetric, + = right-skewed, - = left-skewed)
                        - **Kurtosis**: Measures tail heaviness (0 = normal, + = heavy tails, - = light tails)

                        **Quartiles:**
                        - **Q1 (25%)**: 25% of data falls below this value
                        - **Q3 (75%)**: 75% of data falls below this value
                        """)

                    # Download Statistics
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        stats_df.to_excel(writer, sheet_name='Descriptive_Statistics')
                    output.seek(0)

                    st.download_button(
                        label="📥 Download Statistics Report",
                        data=output,
                        file_name="descriptive_statistics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    # Outlier Detection
                    st.markdown("### 🎯 Outlier Detection (IQR Method)")

                    outlier_info = []
                    for col in selected_vars:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                        outlier_info.append({
                            'Variable': col,
                            'Lower Bound': round(lower_bound, decimal_places),
                            'Upper Bound': round(upper_bound, decimal_places),
                            'Outliers Count': len(outliers),
                            'Outliers %': round(len(outliers) / len(df) * 100, 2)
                        })

                    outlier_df = pd.DataFrame(outlier_info)
                    st.dataframe(outlier_df.round(decimal_places), use_container_width=True)

                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ About Outliers:</strong> Outliers are detected using the IQR method (1.5 × IQR rule). 
                        Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are flagged as outliers. These may represent 
                        genuine extreme values or data errors that require investigation.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No numeric variables found in the dataset.")

        # =============================================================================
        # TAB 3: DATA TRANSFORMATIONS
        # =============================================================================

        with tab3:
            st.markdown("""
            <div class="info-box">
                <div class="info-box-title">🔄 What are Data Transformations?</div>
                <div class="info-box-content">
                    Data transformations modify variable distributions to meet statistical assumptions or improve 
                    model performance. Common transformations include logarithmic (for right-skewed data), 
                    standardization (zero mean, unit variance), and power transformations (Box-Cox, Yeo-Johnson) 
                    which automatically find optimal transformations to achieve normality.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if len(numeric_cols) > 0:
                col1, col2 = st.columns([1, 1])

                with col1:
                    transform_var = st.selectbox(
                        "Select Variable to Transform",
                        numeric_cols,
                        help="Choose a numeric variable to apply transformations"
                    )

                with col2:
                    transform_type = st.selectbox(
                        "Select Transformation Type",
                        [
                            "Log Transformation (ln)",
                            "Log10 Transformation",
                            "Square Root Transformation",
                            "Cube Root Transformation",
                            "Reciprocal Transformation (1/x)",
                            "Square Transformation (x²)",
                            "Standardization (Z-score)",
                            "Min-Max Normalization (0-1)",
                            "Robust Scaling",
                            "Box-Cox Transformation",
                            "Yeo-Johnson Transformation",
                            "Quantile Transformation (Normal)",
                            "Quantile Transformation (Uniform)"
                        ],
                        help="Choose the transformation method to apply"
                    )

                if st.button("🔄 Apply Transformation", type="primary"):
                    original_data = df[transform_var].dropna()
                    transformed_data = None
                    transform_success = True
                    error_msg = ""

                    try:
                        if transform_type == "Log Transformation (ln)":
                            if (original_data <= 0).any():
                                shift = abs(original_data.min()) + 1
                                transformed_data = np.log(original_data + shift)
                                st.info(f"ℹ️ Data shifted by {shift:.4f} to handle non-positive values")
                            else:
                                transformed_data = np.log(original_data)

                        elif transform_type == "Log10 Transformation":
                            if (original_data <= 0).any():
                                shift = abs(original_data.min()) + 1
                                transformed_data = np.log10(original_data + shift)
                                st.info(f"ℹ️ Data shifted by {shift:.4f} to handle non-positive values")
                            else:
                                transformed_data = np.log10(original_data)

                        elif transform_type == "Square Root Transformation":
                            if (original_data < 0).any():
                                shift = abs(original_data.min())
                                transformed_data = np.sqrt(original_data + shift)
                                st.info(f"ℹ️ Data shifted by {shift:.4f} to handle negative values")
                            else:
                                transformed_data = np.sqrt(original_data)

                        elif transform_type == "Cube Root Transformation":
                            transformed_data = np.cbrt(original_data)

                        elif transform_type == "Reciprocal Transformation (1/x)":
                            if (original_data == 0).any():
                                transformed_data = 1 / (original_data + 0.001)
                                st.info("ℹ️ Added small constant (0.001) to avoid division by zero")
                            else:
                                transformed_data = 1 / original_data

                        elif transform_type == "Square Transformation (x²)":
                            transformed_data = original_data ** 2

                        elif transform_type == "Standardization (Z-score)":
                            scaler = StandardScaler()
                            transformed_data = pd.Series(
                                scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten(),
                                index=original_data.index
                            )

                        elif transform_type == "Min-Max Normalization (0-1)":
                            scaler = MinMaxScaler()
                            transformed_data = pd.Series(
                                scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten(),
                                index=original_data.index
                            )

                        elif transform_type == "Robust Scaling":
                            scaler = RobustScaler()
                            transformed_data = pd.Series(
                                scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten(),
                                index=original_data.index
                            )

                        elif transform_type == "Box-Cox Transformation":
                            if (original_data <= 0).any():
                                shift = abs(original_data.min()) + 1
                                pt = PowerTransformer(method='box-cox', standardize=True)
                                transformed_data = pd.Series(
                                    pt.fit_transform((original_data + shift).values.reshape(-1, 1)).flatten(),
                                    index=original_data.index
                                )
                                st.info(
                                    f"ℹ️ Data shifted by {shift:.4f} for Box-Cox (requires positive values). Lambda = {pt.lambdas_[0]:.4f}")
                            else:
                                pt = PowerTransformer(method='box-cox', standardize=True)
                                transformed_data = pd.Series(
                                    pt.fit_transform(original_data.values.reshape(-1, 1)).flatten(),
                                    index=original_data.index
                                )
                                st.info(f"ℹ️ Box-Cox Lambda = {pt.lambdas_[0]:.4f}")

                        elif transform_type == "Yeo-Johnson Transformation":
                            pt = PowerTransformer(method='yeo-johnson', standardize=True)
                            transformed_data = pd.Series(
                                pt.fit_transform(original_data.values.reshape(-1, 1)).flatten(),
                                index=original_data.index
                            )
                            st.info(f"ℹ️ Yeo-Johnson Lambda = {pt.lambdas_[0]:.4f}")

                        elif transform_type == "Quantile Transformation (Normal)":
                            qt = QuantileTransformer(output_distribution='normal', random_state=42)
                            transformed_data = pd.Series(
                                qt.fit_transform(original_data.values.reshape(-1, 1)).flatten(),
                                index=original_data.index
                            )

                        elif transform_type == "Quantile Transformation (Uniform)":
                            qt = QuantileTransformer(output_distribution='uniform', random_state=42)
                            transformed_data = pd.Series(
                                qt.fit_transform(original_data.values.reshape(-1, 1)).flatten(),
                                index=original_data.index
                            )

                    except Exception as e:
                        transform_success = False
                        error_msg = str(e)

                    if transform_success and transformed_data is not None:
                        # Comparison Visualization
                        fig = make_subplots(rows=2, cols=2,
                                            subplot_titles=['Original Distribution', 'Transformed Distribution',
                                                            'Original Q-Q Plot', 'Transformed Q-Q Plot'])

                        # Original Histogram
                        fig.add_trace(
                            go.Histogram(x=original_data, name='Original', marker_color='#8b5cf6', opacity=0.7),
                            row=1, col=1
                        )

                        # Transformed Histogram
                        fig.add_trace(
                            go.Histogram(x=transformed_data, name='Transformed', marker_color='#10b981', opacity=0.7),
                            row=1, col=2
                        )

                        # Q-Q Plots
                        # Original Q-Q
                        orig_sorted = np.sort(original_data)
                        orig_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(orig_sorted)))
                        fig.add_trace(
                            go.Scatter(x=orig_theoretical, y=orig_sorted, mode='markers',
                                       name='Original Q-Q', marker=dict(color='#8b5cf6', size=5)),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=[orig_theoretical.min(), orig_theoretical.max()],
                                       y=[orig_sorted.min(), orig_sorted.max()],
                                       mode='lines', name='Reference', line=dict(color='red', dash='dash')),
                            row=2, col=1
                        )

                        # Transformed Q-Q
                        trans_sorted = np.sort(transformed_data)
                        trans_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(trans_sorted)))
                        fig.add_trace(
                            go.Scatter(x=trans_theoretical, y=trans_sorted, mode='markers',
                                       name='Transformed Q-Q', marker=dict(color='#4caf50', size=5)),
                            row=2, col=2
                        )
                        fig.add_trace(
                            go.Scatter(x=[trans_theoretical.min(), trans_theoretical.max()],
                                       y=[trans_sorted.min(), trans_sorted.max()],
                                       mode='lines', name='Reference', line=dict(color='red', dash='dash')),
                            row=2, col=2
                        )

                        fig.update_layout(
                            height=700,
                            showlegend=False,
                            title_text=f"Transformation Comparison: {transform_type}",
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Statistics Comparison
                        st.markdown("### 📊 Statistics Comparison")

                        comparison_stats = pd.DataFrame({
                            'Statistic': ['Mean', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max'],
                            'Original': [
                                original_data.mean(),
                                original_data.std(),
                                original_data.skew(),
                                original_data.kurtosis(),
                                original_data.min(),
                                original_data.max()
                            ],
                            'Transformed': [
                                transformed_data.mean(),
                                transformed_data.std(),
                                transformed_data.skew(),
                                transformed_data.kurtosis(),
                                transformed_data.min(),
                                transformed_data.max()
                            ]
                        })
                        comparison_stats['Change'] = comparison_stats['Transformed'] - comparison_stats['Original']

                        st.dataframe(comparison_stats.round(decimal_places), use_container_width=True)

                        # Normality Test Comparison
                        st.markdown("### 🧪 Normality Test Comparison")

                        _, orig_p = shapiro(original_data[:5000] if len(original_data) > 5000 else original_data)
                        _, trans_p = shapiro(
                            transformed_data[:5000] if len(transformed_data) > 5000 else transformed_data)

                        norm_comparison = pd.DataFrame({
                            'Data': ['Original', 'Transformed'],
                            'Shapiro-Wilk p-value': [orig_p, trans_p],
                            'Normal?': ['Yes ✅' if p > 0.05 else 'No ❌' for p in [orig_p, trans_p]]
                        })

                        st.dataframe(norm_comparison.round(decimal_places), use_container_width=True)

                        # Save Transformed Data
                        df[f'{transform_var}_transformed'] = np.nan
                        df.loc[original_data.index, f'{transform_var}_transformed'] = transformed_data.values

                        st.success(f"✅ Transformed variable saved as '{transform_var}_transformed'")

                        # Download transformed data
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        output.seek(0)

                        st.download_button(
                            label="📥 Download Data with Transformed Variable",
                            data=output,
                            file_name="data_with_transformations.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error(f"❌ Transformation failed: {error_msg}")

                # Transformation Guide
                with st.expander("📚 Transformation Guide"):
                    st.markdown("""
                    **When to Use Each Transformation:**

                    | Transformation | Use When | Effect |
                    |----------------|----------|--------|
                    | **Log** | Right-skewed data, multiplicative relationships | Reduces right skew, stabilizes variance |
                    | **Square Root** | Count data, moderate right skew | Milder than log, handles zeros |
                    | **Box-Cox** | Unknown optimal transformation (positive data) | Automatically finds best power parameter |
                    | **Yeo-Johnson** | Unknown optimal transformation (any data) | Like Box-Cox but handles negative values |
                    | **Z-score** | Features with different scales | Mean=0, Std=1 |
                    | **Min-Max** | Need bounded [0,1] range | Preserves relative distances |
                    | **Robust** | Data with outliers | Uses median and IQR |
                    | **Quantile (Normal)** | Force normal distribution | Maps to normal distribution |

                    **Box-Cox Formula**: y(λ) = (x^λ - 1) / λ when λ ≠ 0, or ln(x) when λ = 0

                    **Yeo-Johnson Formula**: Extends Box-Cox to handle zero and negative values
                    """)
            else:
                st.warning("⚠️ No numeric variables found for transformation.")

        # =============================================================================
        # TAB 4: NORMALITY TESTS
        # =============================================================================

        with tab4:
            st.markdown("""
            <div class="info-box">
                <div class="info-box-title">📈 What are Normality Tests?</div>
                <div class="info-box-content">
                    Normality tests assess whether a sample comes from a normally distributed population. This is 
                    important because many statistical tests (t-tests, ANOVA, regression) assume normality. 
                    Multiple tests are provided as each has different strengths: Shapiro-Wilk is powerful for small 
                    samples, while Anderson-Darling and Jarque-Bera focus on different aspects of the distribution.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if len(numeric_cols) > 0:
                norm_vars = st.multiselect(
                    "Select Variables for Normality Testing",
                    numeric_cols,
                    default=[numeric_cols[0]] if numeric_cols else [],
                    help="Select one or more variables to test for normality"
                )

                if norm_vars and st.button("🧪 Run Normality Tests", type="primary"):
                    results = []

                    for var in norm_vars:
                        data = df[var].dropna()
                        sample = data.values[:5000] if len(data) > 5000 else data.values

                        st.markdown(f"### 📊 Normality Tests for: **{var}**")

                        # 1. Shapiro-Wilk Test
                        try:
                            stat_sw, p_sw = shapiro(sample)
                            results.append(
                                {'Variable': var, 'Test': 'Shapiro-Wilk', 'Statistic': stat_sw, 'p-value': p_sw})
                            st.markdown(get_normality_interpretation(stat_sw, p_sw, "Shapiro-Wilk Test"))
                        except:
                            st.warning("Shapiro-Wilk test could not be computed.")

                        # 2. D'Agostino-Pearson Test
                        if len(sample) >= 20:
                            try:
                                stat_dp, p_dp = normaltest(sample)
                                results.append({'Variable': var, 'Test': "D'Agostino-Pearson", 'Statistic': stat_dp,
                                                'p-value': p_dp})
                                st.markdown(get_normality_interpretation(stat_dp, p_dp, "D'Agostino-Pearson Test"))
                            except:
                                st.warning("D'Agostino-Pearson test could not be computed.")

                        # 3. Jarque-Bera Test
                        try:
                            stat_jb, p_jb = jarque_bera(sample)
                            results.append(
                                {'Variable': var, 'Test': 'Jarque-Bera', 'Statistic': stat_jb, 'p-value': p_jb})
                            st.markdown(get_normality_interpretation(stat_jb, p_jb, "Jarque-Bera Test"))
                        except:
                            st.warning("Jarque-Bera test could not be computed.")

                        # 4. Anderson-Darling Test
                        try:
                            result_ad = anderson(sample, dist='norm')
                            ad_stat = result_ad.statistic
                            ad_critical = result_ad.critical_values[2]  # 5% significance level
                            ad_conclusion = "Normal" if ad_stat < ad_critical else "Not Normal"
                            results.append(
                                {'Variable': var, 'Test': 'Anderson-Darling', 'Statistic': ad_stat, 'p-value': np.nan})
                            st.markdown(
                                f"🔵 **Anderson-Darling Test**: Statistic = {ad_stat:.4f}, Critical Value (5%) = {ad_critical:.4f}\n\n*Conclusion*: {ad_conclusion} (reject H₀ if statistic > critical value)")
                        except:
                            st.warning("Anderson-Darling test could not be computed.")

                        # 5. Lilliefors Test
                        try:
                            stat_lf, p_lf = lilliefors(sample)
                            results.append(
                                {'Variable': var, 'Test': 'Lilliefors', 'Statistic': stat_lf, 'p-value': p_lf})
                            st.markdown(get_normality_interpretation(stat_lf, p_lf, "Lilliefors Test (Modified K-S)"))
                        except:
                            st.warning("Lilliefors test could not be computed.")

                        # 6. Kolmogorov-Smirnov Test
                        try:
                            # Standardize for K-S test
                            standardized = (sample - np.mean(sample)) / np.std(sample)
                            stat_ks, p_ks = kstest(standardized, 'norm')
                            results.append(
                                {'Variable': var, 'Test': 'Kolmogorov-Smirnov', 'Statistic': stat_ks, 'p-value': p_ks})
                            st.markdown(get_normality_interpretation(stat_ks, p_ks, "Kolmogorov-Smirnov Test"))
                        except:
                            st.warning("Kolmogorov-Smirnov test could not be computed.")

                        st.markdown("---")

                        # Visual Check
                        st.markdown("### 📉 Visual Normality Assessment")

                        fig = make_subplots(rows=1, cols=3,
                                            subplot_titles=['Histogram with Normal Curve', 'Q-Q Plot', 'Box Plot'])

                        # Histogram
                        fig.add_trace(
                            go.Histogram(x=data, name='Data', marker_color='#8b5cf6', opacity=0.7,
                                         histnorm='probability density'),
                            row=1, col=1
                        )

                        # Overlay normal curve
                        x_range = np.linspace(data.min(), data.max(), 100)
                        normal_curve = stats.norm.pdf(x_range, data.mean(), data.std())
                        fig.add_trace(
                            go.Scatter(x=x_range, y=normal_curve, mode='lines',
                                       name='Normal Curve', line=dict(color='red', width=2)),
                            row=1, col=1
                        )

                        # Q-Q Plot
                        sorted_data = np.sort(data)
                        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
                        fig.add_trace(
                            go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers',
                                       name='Q-Q', marker=dict(color='#8b5cf6', size=5)),
                            row=1, col=2
                        )
                        fig.add_trace(
                            go.Scatter(x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                                       y=[sorted_data.min(), sorted_data.max()],
                                       mode='lines', name='45° Line', line=dict(color='red', dash='dash')),
                            row=1, col=2
                        )

                        # Box Plot
                        fig.add_trace(
                            go.Box(y=data, name=var, marker_color='#8b5cf6'),
                            row=1, col=3
                        )

                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Summary Table
                    if results:
                        st.markdown("### 📋 Complete Results Summary")
                        results_df = pd.DataFrame(results)
                        results_df['Conclusion'] = results_df['p-value'].apply(
                            lambda p: 'Normal ✅' if p > significance_level else 'Not Normal ❌' if pd.notna(
                                p) else 'See Critical Value'
                        )
                        st.dataframe(results_df.round(decimal_places), use_container_width=True)

                        # Download Results
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results_df.to_excel(writer, sheet_name='Normality_Tests', index=False)
                        output.seek(0)

                        st.download_button(
                            label="📥 Download Normality Test Results",
                            data=output,
                            file_name="normality_tests.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                # Test Explanations
                with st.expander("📚 Understanding Normality Tests"):
                    st.markdown("""
                    **Overview of Normality Tests:**

                    | Test | Best For | Null Hypothesis |
                    |------|----------|-----------------|
                    | **Shapiro-Wilk** | Small to medium samples (n < 5000) | Data is normally distributed |
                    | **D'Agostino-Pearson** | Larger samples (n ≥ 20) | Data is normally distributed |
                    | **Jarque-Bera** | Large samples; tests skewness & kurtosis | Skewness=0, Kurtosis=3 |
                    | **Anderson-Darling** | All sample sizes; sensitive to tails | Data follows specified distribution |
                    | **Lilliefors** | When mean/variance are estimated | Data is normally distributed |
                    | **Kolmogorov-Smirnov** | Comparing to any distribution | Sample matches reference distribution |

                    **Interpretation Guidelines:**
                    - **p-value > α**: Fail to reject H₀ → Data appears normal
                    - **p-value ≤ α**: Reject H₀ → Data significantly deviates from normality

                    **Important Notes:**
                    - Visual inspection (Q-Q plots, histograms) should complement statistical tests
                    - Large samples may show statistical significance even with minor deviations
                    - Use multiple tests for robust conclusions
                    - Consider transformations if normality is violated
                    """)
            else:
                st.warning("⚠️ No numeric variables found for normality testing.")

        # =============================================================================
        # TAB 5: CORRELATION ANALYSIS
        # =============================================================================

        with tab5:
            st.markdown("""
            <div class="info-box">
                <div class="info-box-title">🔗 What is Correlation Analysis?</div>
                <div class="info-box-content">
                    Correlation measures the strength and direction of relationships between variables. 
                    Pearson correlation assesses linear relationships, Spearman measures monotonic relationships 
                    (ranks), and Kendall's Tau is robust to outliers and ties. Values range from -1 (perfect 
                    negative) to +1 (perfect positive), with 0 indicating no relationship.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if len(numeric_cols) >= 2:
                st.markdown("### 🎯 Correlation Settings")

                col1, col2 = st.columns(2)

                with col1:
                    corr_vars = st.multiselect(
                        "Select Variables for Correlation Matrix",
                        numeric_cols,
                        default=numeric_cols[:min(8, len(numeric_cols))],
                        help="Select variables to include in the correlation analysis"
                    )

                with col2:
                    corr_method = st.selectbox(
                        "Correlation Method",
                        ["Pearson", "Spearman", "Kendall"],
                        help="Pearson (linear), Spearman (monotonic/ranks), Kendall (robust)"
                    )

                if len(corr_vars) >= 2:
                    # Calculate Correlation Matrix
                    corr_matrix = df[corr_vars].corr(method=corr_method.lower())

                    # Heatmap
                    st.markdown("### 🗺️ Correlation Heatmap")

                    fig = px.imshow(
                        corr_matrix,
                        labels=dict(x="Variable", y="Variable", color="Correlation"),
                        x=corr_vars,
                        y=corr_vars,
                        color_continuous_scale='RdBu_r',
                        zmin=-1,
                        zmax=1,
                        aspect='auto'
                    )

                    # Add correlation values as text
                    fig.update_traces(
                        text=corr_matrix.round(2).values,
                        texttemplate='%{text}',
                        textfont={"size": 10}
                    )

                    fig.update_layout(
                        height=600,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family='Source Sans Pro')
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Correlation Table
                    st.markdown("### 📊 Correlation Matrix")
                    st.dataframe(corr_matrix.round(decimal_places), use_container_width=True)

                    # Pairwise Correlation with p-values
                    st.markdown("### 🔬 Detailed Pairwise Correlations with Significance")

                    pairwise_results = []
                    for i, var1 in enumerate(corr_vars):
                        for j, var2 in enumerate(corr_vars):
                            if i < j:
                                valid_data = df[[var1, var2]].dropna()
                                if len(valid_data) > 2:
                                    if corr_method == "Pearson":
                                        corr, p = pearsonr(valid_data[var1], valid_data[var2])
                                    elif corr_method == "Spearman":
                                        corr, p = spearmanr(valid_data[var1], valid_data[var2])
                                    else:
                                        corr, p = kendalltau(valid_data[var1], valid_data[var2])

                                    # Interpret strength
                                    abs_corr = abs(corr)
                                    if abs_corr >= 0.8:
                                        strength = "Very Strong"
                                    elif abs_corr >= 0.6:
                                        strength = "Strong"
                                    elif abs_corr >= 0.4:
                                        strength = "Moderate"
                                    elif abs_corr >= 0.2:
                                        strength = "Weak"
                                    else:
                                        strength = "Very Weak"

                                    direction = "Positive" if corr > 0 else "Negative" if corr < 0 else "None"
                                    sig = "Yes ✅" if p < significance_level else "No ❌"

                                    pairwise_results.append({
                                        'Variable 1': var1,
                                        'Variable 2': var2,
                                        'Correlation': corr,
                                        'p-value': p,
                                        'Significant?': sig,
                                        'Strength': strength,
                                        'Direction': direction
                                    })

                    if pairwise_results:
                        pairwise_df = pd.DataFrame(pairwise_results)
                        pairwise_df = pairwise_df.sort_values('Correlation', key=abs, ascending=False)
                        st.dataframe(pairwise_df.round(decimal_places), use_container_width=True)

                        # Scatter Plot Matrix for Top Correlations
                        st.markdown("### 📈 Scatter Plot Matrix")

                        if len(corr_vars) <= 6:
                            fig = px.scatter_matrix(
                                df[corr_vars].dropna(),
                                dimensions=corr_vars,
                                color_discrete_sequence=['#2d5a87']
                            )
                            fig.update_traces(diagonal_visible=False, showupperhalf=False)
                            fig.update_layout(
                                height=700,
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(
                                "📊 Scatter matrix is best viewed with 6 or fewer variables. Select fewer variables for visualization.")

                        # Download Results
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
                            pairwise_df.to_excel(writer, sheet_name='Pairwise_Correlations', index=False)
                        output.seek(0)

                        st.download_button(
                            label="📥 Download Correlation Analysis",
                            data=output,
                            file_name="correlation_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                # Correlation Guide
                with st.expander("📚 Correlation Interpretation Guide"):
                    st.markdown("""
                    **Correlation Coefficient Interpretation:**

                    | Range | Strength |
                    |-------|----------|
                    | 0.80 - 1.00 | Very Strong |
                    | 0.60 - 0.79 | Strong |
                    | 0.40 - 0.59 | Moderate |
                    | 0.20 - 0.39 | Weak |
                    | 0.00 - 0.19 | Very Weak / None |

                    **Method Comparison:**

                    | Method | Type | Best For |
                    |--------|------|----------|
                    | **Pearson** | Parametric | Linear relationships, normally distributed data |
                    | **Spearman** | Non-parametric | Monotonic relationships, ordinal data |
                    | **Kendall** | Non-parametric | Small samples, many tied ranks |

                    **Important Notes:**
                    - Correlation ≠ Causation
                    - Always visualize relationships with scatter plots
                    - Consider spurious correlations and confounding variables
                    - Check for non-linear patterns that correlation may miss
                    """)
            else:
                st.warning("⚠️ At least 2 numeric variables are required for correlation analysis.")

        # =============================================================================
        # TAB 6: HYPOTHESIS TESTS
        # =============================================================================

        with tab6:
            st.markdown("""
            <div class="info-box">
                <div class="info-box-title">🧪 What is Hypothesis Testing?</div>
                <div class="info-box-content">
                    Hypothesis testing is a statistical method for making decisions based on data. We test a null 
                    hypothesis (H₀) against an alternative hypothesis (H₁). If the p-value is less than the 
                    significance level (α), we reject H₀. Different tests are designed for comparing means, 
                    variances, distributions, or testing for independence.
                </div>
            </div>
            """, unsafe_allow_html=True)

            test_category = st.selectbox(
                "Select Test Category",
                ["Parametric Tests (Means)", "Non-Parametric Tests", "Variance Tests", "Time Series Tests"],
                help="Choose the category of statistical tests"
            )

            if test_category == "Parametric Tests (Means)":
                st.markdown("### 📊 Parametric Tests for Comparing Means")

                param_test = st.selectbox(
                    "Select Test",
                    ["One-Sample t-Test", "Independent Two-Sample t-Test", "Paired t-Test", "One-Way ANOVA"],
                    help="Choose the specific test to perform"
                )

                if param_test == "One-Sample t-Test":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>One-Sample t-Test:</strong> Tests if the mean of a sample differs significantly from 
                        a hypothesized population mean. Assumes approximately normal distribution.
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        test_var = st.selectbox("Select Variable", numeric_cols)
                    with col2:
                        pop_mean = st.number_input("Hypothesized Population Mean", value=0.0)

                    if st.button("Run One-Sample t-Test", type="primary"):
                        data = df[test_var].dropna()
                        stat, p = ttest_1samp(data, pop_mean)

                        st.markdown(f"""
                        ### Results

                        **Test Statistic (t):** {stat:.{decimal_places}f}

                        **p-value:** {p:.{decimal_places}f}

                        **Sample Mean:** {data.mean():.{decimal_places}f}

                        **Sample Std Dev:** {data.std():.{decimal_places}f}

                        **Sample Size:** {len(data)}
                        """)

                        interp, emoji = interpret_pvalue(p, significance_level)
                        if p < significance_level:
                            st.error(
                                f"{emoji} **Conclusion:** Reject H₀. The sample mean ({data.mean():.4f}) is significantly different from {pop_mean}. ({interp})")
                        else:
                            st.success(
                                f"{emoji} **Conclusion:** Fail to reject H₀. No significant difference from {pop_mean}. ({interp})")

                elif param_test == "Independent Two-Sample t-Test":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Independent Two-Sample t-Test:</strong> Compares means of two independent groups. 
                        Assumes normality and (typically) equal variances.
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        group_var = st.selectbox("Grouping Variable",
                                                 categorical_cols if categorical_cols else ['No categorical variables'])
                    with col2:
                        test_var = st.selectbox("Test Variable", numeric_cols)
                    with col3:
                        equal_var = st.checkbox("Assume Equal Variances", value=True)

                    if group_var != 'No categorical variables' and st.button("Run Two-Sample t-Test", type="primary"):
                        groups = df[group_var].dropna().unique()[:2]
                        if len(groups) >= 2:
                            group1 = df[df[group_var] == groups[0]][test_var].dropna()
                            group2 = df[df[group_var] == groups[1]][test_var].dropna()

                            stat, p = ttest_ind(group1, group2, equal_var=equal_var)

                            st.markdown(f"""
                            ### Results

                            **Test Statistic (t):** {stat:.{decimal_places}f}

                            **p-value:** {p:.{decimal_places}f}

                            | Group | n | Mean | Std Dev |
                            |-------|---|------|---------|
                            | {groups[0]} | {len(group1)} | {group1.mean():.4f} | {group1.std():.4f} |
                            | {groups[1]} | {len(group2)} | {group2.mean():.4f} | {group2.std():.4f} |
                            """)

                            interp, emoji = interpret_pvalue(p, significance_level)
                            if p < significance_level:
                                st.error(
                                    f"{emoji} **Conclusion:** Reject H₀. The means are significantly different. ({interp})")
                            else:
                                st.success(
                                    f"{emoji} **Conclusion:** Fail to reject H₀. No significant difference between means. ({interp})")

                elif param_test == "Paired t-Test":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Paired t-Test:</strong> Compares means of two related measurements (e.g., before/after). 
                        Each pair should be from the same subject or matched cases.
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        var1 = st.selectbox("First Variable (e.g., Before)", numeric_cols, key='paired1')
                    with col2:
                        var2 = st.selectbox("Second Variable (e.g., After)", numeric_cols, key='paired2')

                    if st.button("Run Paired t-Test", type="primary"):
                        valid_data = df[[var1, var2]].dropna()
                        stat, p = ttest_rel(valid_data[var1], valid_data[var2])

                        diff = valid_data[var1] - valid_data[var2]

                        st.markdown(f"""
                        ### Results

                        **Test Statistic (t):** {stat:.{decimal_places}f}

                        **p-value:** {p:.{decimal_places}f}

                        **Mean Difference:** {diff.mean():.{decimal_places}f}

                        **Std Dev of Differences:** {diff.std():.{decimal_places}f}

                        **Number of Pairs:** {len(valid_data)}
                        """)

                        interp, emoji = interpret_pvalue(p, significance_level)
                        if p < significance_level:
                            st.error(
                                f"{emoji} **Conclusion:** Reject H₀. Significant difference between paired observations. ({interp})")
                        else:
                            st.success(
                                f"{emoji} **Conclusion:** Fail to reject H₀. No significant difference between pairs. ({interp})")

                elif param_test == "One-Way ANOVA":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>One-Way ANOVA:</strong> Tests if means differ across three or more groups. 
                        Assumes normality and equal variances (homoscedasticity).
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        group_var = st.selectbox("Grouping Variable",
                                                 categorical_cols if categorical_cols else ['No categorical variables'],
                                                 key='anova_group')
                    with col2:
                        test_var = st.selectbox("Test Variable", numeric_cols, key='anova_test')

                    if group_var != 'No categorical variables' and st.button("Run One-Way ANOVA", type="primary"):
                        groups = [df[df[group_var] == g][test_var].dropna() for g in df[group_var].dropna().unique()]
                        groups = [g for g in groups if len(g) > 0]

                        if len(groups) >= 2:
                            stat, p = f_oneway(*groups)

                            # Group statistics
                            group_stats = []
                            for g_name, g_data in zip(df[group_var].dropna().unique(), groups):
                                group_stats.append({
                                    'Group': g_name,
                                    'n': len(g_data),
                                    'Mean': g_data.mean(),
                                    'Std Dev': g_data.std()
                                })

                            st.markdown(f"""
                            ### Results

                            **F-Statistic:** {stat:.{decimal_places}f}

                            **p-value:** {p:.{decimal_places}f}
                            """)

                            st.markdown("**Group Statistics:**")
                            st.dataframe(pd.DataFrame(group_stats).round(decimal_places), use_container_width=True)

                            interp, emoji = interpret_pvalue(p, significance_level)
                            if p < significance_level:
                                st.error(
                                    f"{emoji} **Conclusion:** Reject H₀. At least one group mean differs significantly. ({interp})")
                            else:
                                st.success(
                                    f"{emoji} **Conclusion:** Fail to reject H₀. No significant difference among group means. ({interp})")

            elif test_category == "Non-Parametric Tests":
                st.markdown("### 📊 Non-Parametric Tests")

                np_test = st.selectbox(
                    "Select Test",
                    ["Mann-Whitney U Test", "Wilcoxon Signed-Rank Test", "Kruskal-Wallis H Test"],
                    help="Non-parametric alternatives to t-tests and ANOVA"
                )

                if np_test == "Mann-Whitney U Test":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Mann-Whitney U Test:</strong> Non-parametric alternative to independent t-test. 
                        Compares distributions of two independent groups. Does not assume normality.
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        group_var = st.selectbox("Grouping Variable",
                                                 categorical_cols if categorical_cols else ['No categorical variables'],
                                                 key='mw_group')
                    with col2:
                        test_var = st.selectbox("Test Variable", numeric_cols, key='mw_test')

                    if group_var != 'No categorical variables' and st.button("Run Mann-Whitney U Test", type="primary"):
                        groups = df[group_var].dropna().unique()[:2]
                        if len(groups) >= 2:
                            group1 = df[df[group_var] == groups[0]][test_var].dropna()
                            group2 = df[df[group_var] == groups[1]][test_var].dropna()

                            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')

                            st.markdown(f"""
                            ### Results

                            **U-Statistic:** {stat:.{decimal_places}f}

                            **p-value:** {p:.{decimal_places}f}

                            | Group | n | Median | IQR |
                            |-------|---|--------|-----|
                            | {groups[0]} | {len(group1)} | {group1.median():.4f} | {group1.quantile(0.75) - group1.quantile(0.25):.4f} |
                            | {groups[1]} | {len(group2)} | {group2.median():.4f} | {group2.quantile(0.75) - group2.quantile(0.25):.4f} |
                            """)

                            interp, emoji = interpret_pvalue(p, significance_level)
                            if p < significance_level:
                                st.error(
                                    f"{emoji} **Conclusion:** Reject H₀. Distributions differ significantly. ({interp})")
                            else:
                                st.success(
                                    f"{emoji} **Conclusion:** Fail to reject H₀. No significant difference between distributions. ({interp})")

                elif np_test == "Wilcoxon Signed-Rank Test":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Wilcoxon Signed-Rank Test:</strong> Non-parametric alternative to paired t-test. 
                        Tests if paired observations come from the same distribution.
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        var1 = st.selectbox("First Variable", numeric_cols, key='wilcox1')
                    with col2:
                        var2 = st.selectbox("Second Variable", numeric_cols, key='wilcox2')

                    if st.button("Run Wilcoxon Signed-Rank Test", type="primary"):
                        valid_data = df[[var1, var2]].dropna()
                        stat, p = wilcoxon(valid_data[var1], valid_data[var2])

                        st.markdown(f"""
                        ### Results

                        **W-Statistic:** {stat:.{decimal_places}f}

                        **p-value:** {p:.{decimal_places}f}

                        **Number of Pairs:** {len(valid_data)}
                        """)

                        interp, emoji = interpret_pvalue(p, significance_level)
                        if p < significance_level:
                            st.error(
                                f"{emoji} **Conclusion:** Reject H₀. Significant difference between paired observations. ({interp})")
                        else:
                            st.success(
                                f"{emoji} **Conclusion:** Fail to reject H₀. No significant difference between pairs. ({interp})")

                elif np_test == "Kruskal-Wallis H Test":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Kruskal-Wallis H Test:</strong> Non-parametric alternative to one-way ANOVA. 
                        Tests if samples come from the same distribution across multiple groups.
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        group_var = st.selectbox("Grouping Variable",
                                                 categorical_cols if categorical_cols else ['No categorical variables'],
                                                 key='kw_group')
                    with col2:
                        test_var = st.selectbox("Test Variable", numeric_cols, key='kw_test')

                    if group_var != 'No categorical variables' and st.button("Run Kruskal-Wallis Test", type="primary"):
                        groups = [df[df[group_var] == g][test_var].dropna().values for g in
                                  df[group_var].dropna().unique()]
                        groups = [g for g in groups if len(g) > 0]

                        if len(groups) >= 2:
                            stat, p = kruskal(*groups)

                            st.markdown(f"""
                            ### Results

                            **H-Statistic:** {stat:.{decimal_places}f}

                            **p-value:** {p:.{decimal_places}f}

                            **Number of Groups:** {len(groups)}
                            """)

                            interp, emoji = interpret_pvalue(p, significance_level)
                            if p < significance_level:
                                st.error(
                                    f"{emoji} **Conclusion:** Reject H₀. At least one group distribution differs. ({interp})")
                            else:
                                st.success(
                                    f"{emoji} **Conclusion:** Fail to reject H₀. No significant difference among groups. ({interp})")

            elif test_category == "Variance Tests":
                st.markdown("### 📊 Tests for Homogeneity of Variance")

                var_test = st.selectbox(
                    "Select Test",
                    ["Levene's Test", "Bartlett's Test"],
                    help="Tests for equality of variances across groups"
                )

                st.markdown(f"""
                <div class="warning-box">
                    <strong>{var_test}:</strong> Tests whether variances are equal across groups. 
                    {'Robust to departures from normality.' if var_test == "Levene's Test" else 'Assumes normality.'}
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    group_var = st.selectbox("Grouping Variable",
                                             categorical_cols if categorical_cols else ['No categorical variables'],
                                             key='var_group')
                with col2:
                    test_var = st.selectbox("Test Variable", numeric_cols, key='var_test')

                if group_var != 'No categorical variables' and st.button("Run Variance Test", type="primary"):
                    groups = [df[df[group_var] == g][test_var].dropna().values for g in df[group_var].dropna().unique()]
                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) >= 2:
                        if var_test == "Levene's Test":
                            stat, p = levene(*groups)
                        else:
                            stat, p = bartlett(*groups)

                        st.markdown(f"""
                        ### Results

                        **Test Statistic:** {stat:.{decimal_places}f}

                        **p-value:** {p:.{decimal_places}f}
                        """)

                        # Group variances
                        var_stats = []
                        for g_name, g_data in zip(df[group_var].dropna().unique(), groups):
                            var_stats.append({
                                'Group': g_name,
                                'n': len(g_data),
                                'Variance': np.var(g_data, ddof=1),
                                'Std Dev': np.std(g_data, ddof=1)
                            })

                        st.dataframe(pd.DataFrame(var_stats).round(decimal_places), use_container_width=True)

                        interp, emoji = interpret_pvalue(p, significance_level)
                        if p < significance_level:
                            st.error(
                                f"{emoji} **Conclusion:** Reject H₀. Variances are significantly different. ({interp})")
                        else:
                            st.success(
                                f"{emoji} **Conclusion:** Fail to reject H₀. Variances appear equal (homoscedastic). ({interp})")

            elif test_category == "Time Series Tests":
                st.markdown("### 📊 Time Series Diagnostic Tests")

                ts_test = st.selectbox(
                    "Select Test",
                    ["Augmented Dickey-Fuller (ADF) Test", "KPSS Test", "Durbin-Watson Test"],
                    help="Tests for stationarity and autocorrelation"
                )

                test_var = st.selectbox("Select Variable", numeric_cols, key='ts_test_var')

                if ts_test == "Augmented Dickey-Fuller (ADF) Test":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>ADF Test:</strong> Tests for unit root (non-stationarity). 
                        H₀: Series has a unit root (non-stationary). Rejecting H₀ suggests stationarity.
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Run ADF Test", type="primary"):
                        data = df[test_var].dropna()
                        result = adfuller(data, autolag='AIC')

                        st.markdown(f"""
                        ### Results

                        **ADF Statistic:** {result[0]:.{decimal_places}f}

                        **p-value:** {result[1]:.{decimal_places}f}

                        **Lags Used:** {result[2]}

                        **Number of Observations:** {result[3]}

                        **Critical Values:**
                        - 1%: {result[4]['1%']:.{decimal_places}f}
                        - 5%: {result[4]['5%']:.{decimal_places}f}
                        - 10%: {result[4]['10%']:.{decimal_places}f}
                        """)

                        interp, emoji = interpret_pvalue(result[1], significance_level)
                        if result[1] < significance_level:
                            st.success(f"{emoji} **Conclusion:** Reject H₀. Series appears stationary. ({interp})")
                        else:
                            st.error(
                                f"{emoji} **Conclusion:** Fail to reject H₀. Series appears non-stationary (has unit root). ({interp})")

                elif ts_test == "KPSS Test":
                    st.markdown("""
                    <div class="warning-box">
                        <strong>KPSS Test:</strong> Tests for stationarity (opposite of ADF). 
                        H₀: Series is stationary. Rejecting H₀ suggests non-stationarity.
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Run KPSS Test", type="primary"):
                        data = df[test_var].dropna()
                        stat, p, lags, crit = kpss(data, regression='c')

                        st.markdown(f"""
                        ### Results

                        **KPSS Statistic:** {stat:.{decimal_places}f}

                        **p-value:** {p:.{decimal_places}f}

                        **Lags Used:** {lags}

                        **Critical Values:**
                        - 10%: {crit['10%']:.{decimal_places}f}
                        - 5%: {crit['5%']:.{decimal_places}f}
                        - 2.5%: {crit['2.5%']:.{decimal_places}f}
                        - 1%: {crit['1%']:.{decimal_places}f}
                        """)

                        interp, emoji = interpret_pvalue(p, significance_level)
                        if p < significance_level:
                            st.error(f"{emoji} **Conclusion:** Reject H₀. Series appears non-stationary. ({interp})")
                        else:
                            st.success(
                                f"{emoji} **Conclusion:** Fail to reject H₀. Series appears stationary. ({interp})")

        # =============================================================================
        # TAB 7: VISUALIZATIONS
        # =============================================================================

        with tab7:
            st.markdown("""
            <div class="info-box">
                <div class="info-box-title">🎨 Data Visualization</div>
                <div class="info-box-content">
                    Effective visualization is crucial for understanding data patterns, distributions, and relationships. 
                    This section provides various plot types using Plotly for interactive, publication-quality graphics. 
                    Choose single or multiple plots based on your analysis needs.
                </div>
            </div>
            """, unsafe_allow_html=True)

            viz_mode = st.radio(
                "Visualization Mode",
                ["Single Plot", "Multiple Plots (Grid)"],
                horizontal=True,
                help="Choose to create one detailed plot or multiple plots in a grid"
            )

            if viz_mode == "Single Plot":
                plot_type = st.selectbox(
                    "Select Plot Type",
                    [
                        "Histogram",
                        "Box Plot",
                        "Violin Plot",
                        "Scatter Plot",
                        "Line Plot",
                        "Bar Chart",
                        "Area Chart",
                        "Density Plot (KDE)",
                        "ECDF Plot",
                        "Pair Plot",
                        "Heatmap",
                        "3D Scatter Plot",
                        "Sunburst Chart",
                        "Treemap"
                    ],
                    help="Select the type of visualization to create"
                )

                # Dynamic options based on plot type
                if plot_type == "Histogram":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        hist_var = st.selectbox("Variable", numeric_cols, key='hist_var')
                    with col2:
                        n_bins = st.slider("Number of Bins", 10, 100, 30)
                    with col3:
                        hist_norm = st.selectbox("Normalization", ['count', 'probability', 'density', 'percent'])

                    show_kde = st.checkbox("Show KDE Curve", value=True)

                    if st.button("Generate Plot", type="primary"):
                        fig = px.histogram(
                            df, x=hist_var, nbins=n_bins,
                            histnorm=hist_norm if hist_norm != 'count' else None,
                            color_discrete_sequence=[px.colors.qualitative.Bold[0]],
                            title=f"Distribution of {hist_var}"
                        )

                        if show_kde:
                            data = df[hist_var].dropna()
                            x_range = np.linspace(data.min(), data.max(), 100)
                            kde = stats.gaussian_kde(data)

                            # Scale KDE to match histogram
                            if hist_norm == 'density':
                                y_kde = kde(x_range)
                            elif hist_norm == 'probability':
                                y_kde = kde(x_range) * (data.max() - data.min()) / n_bins
                            else:
                                y_kde = kde(x_range) * len(data) * (data.max() - data.min()) / n_bins

                            fig.add_trace(go.Scatter(x=x_range, y=y_kde, mode='lines',
                                                     name='KDE', line=dict(color='red', width=2)))

                        fig.update_layout(
                            plot_bgcolor='white', paper_bgcolor='white',
                            font=dict(family='Source Sans Pro')
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Box Plot":
                    col1, col2 = st.columns(2)
                    with col1:
                        box_var = st.selectbox("Variable", numeric_cols, key='box_var')
                    with col2:
                        group_var = st.selectbox("Group By (optional)", ['None'] + categorical_cols, key='box_group')

                    show_points = st.checkbox("Show Data Points", value=False)

                    if st.button("Generate Plot", type="primary"):
                        fig = px.box(
                            df, y=box_var,
                            x=group_var if group_var != 'None' else None,
                            points='all' if show_points else 'outliers',
                            color=group_var if group_var != 'None' else None,
                            title=f"Box Plot of {box_var}"
                        )
                        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Violin Plot":
                    col1, col2 = st.columns(2)
                    with col1:
                        violin_var = st.selectbox("Variable", numeric_cols, key='violin_var')
                    with col2:
                        group_var = st.selectbox("Group By (optional)", ['None'] + categorical_cols, key='violin_group')

                    show_box = st.checkbox("Show Box Plot Inside", value=True)

                    if st.button("Generate Plot", type="primary"):
                        fig = px.violin(
                            df, y=violin_var,
                            x=group_var if group_var != 'None' else None,
                            box=show_box,
                            color=group_var if group_var != 'None' else None,
                            title=f"Violin Plot of {violin_var}"
                        )
                        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Scatter Plot":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_var = st.selectbox("X Variable", numeric_cols, key='scatter_x')
                    with col2:
                        y_var = st.selectbox("Y Variable", numeric_cols, key='scatter_y')
                    with col3:
                        color_var = st.selectbox("Color By (optional)", ['None'] + categorical_cols + numeric_cols,
                                                 key='scatter_color')

                    add_trendline = st.checkbox("Add Trendline", value=False)

                    if st.button("Generate Plot", type="primary"):
                        fig = px.scatter(
                            df, x=x_var, y=y_var,
                            color=color_var if color_var != 'None' else None,
                            trendline='ols' if add_trendline else None,
                            title=f"Scatter Plot: {x_var} vs {y_var}"
                        )
                        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)

                        if add_trendline:
                            # Show regression statistics
                            valid_data = df[[x_var, y_var]].dropna()
                            corr, p = pearsonr(valid_data[x_var], valid_data[y_var])
                            st.info(f"📊 Pearson Correlation: {corr:.4f} (p-value: {p:.4f})")

                elif plot_type == "Line Plot":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox("X Variable", df.columns.tolist(), key='line_x')
                    with col2:
                        y_vars = st.multiselect("Y Variable(s)", numeric_cols, default=[numeric_cols[0]], key='line_y')

                    if st.button("Generate Plot", type="primary") and y_vars:
                        fig = go.Figure()
                        for y_var in y_vars:
                            fig.add_trace(go.Scatter(x=df[x_var], y=df[y_var], mode='lines+markers', name=y_var))

                        fig.update_layout(
                            title=f"Line Plot",
                            xaxis_title=x_var,
                            yaxis_title="Values",
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Bar Chart":
                    col1, col2 = st.columns(2)
                    with col1:
                        cat_var = st.selectbox("Category Variable",
                                               categorical_cols if categorical_cols else df.columns.tolist(),
                                               key='bar_cat')
                    with col2:
                        val_var = st.selectbox("Value Variable", ['Count'] + numeric_cols, key='bar_val')

                    if st.button("Generate Plot", type="primary"):
                        if val_var == 'Count':
                            counts = df[cat_var].value_counts().reset_index()
                            counts.columns = [cat_var, 'Count']
                            fig = px.bar(counts, x=cat_var, y='Count', title=f"Count by {cat_var}")
                        else:
                            agg_data = df.groupby(cat_var)[val_var].mean().reset_index()
                            fig = px.bar(agg_data, x=cat_var, y=val_var, title=f"Mean {val_var} by {cat_var}")

                        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Density Plot (KDE)":
                    kde_vars = st.multiselect("Select Variable(s)", numeric_cols, default=[numeric_cols[0]],
                                              key='kde_vars')

                    if st.button("Generate Plot", type="primary") and kde_vars:
                        fig = go.Figure()

                        for var in kde_vars:
                            data = df[var].dropna()
                            kde = stats.gaussian_kde(data)
                            x_range = np.linspace(data.min(), data.max(), 200)
                            fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', name=var, fill='tozeroy'))

                        fig.update_layout(
                            title="Density Plot (KDE)",
                            xaxis_title="Value",
                            yaxis_title="Density",
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "ECDF Plot":
                    ecdf_vars = st.multiselect("Select Variable(s)", numeric_cols, default=[numeric_cols[0]],
                                               key='ecdf_vars')

                    if st.button("Generate Plot", type="primary") and ecdf_vars:
                        fig = px.ecdf(df, x=ecdf_vars, title="Empirical Cumulative Distribution Function")
                        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Pair Plot":
                    pair_vars = st.multiselect("Select Variables (2-6 recommended)", numeric_cols,
                                               default=numeric_cols[:min(4, len(numeric_cols))], key='pair_vars')

                    color_var = st.selectbox("Color By (optional)", ['None'] + categorical_cols, key='pair_color')

                    if st.button("Generate Plot", type="primary") and len(pair_vars) >= 2:
                        fig = px.scatter_matrix(
                            df[pair_vars + ([color_var] if color_var != 'None' else [])].dropna(),
                            dimensions=pair_vars,
                            color=color_var if color_var != 'None' else None,
                            title="Pair Plot Matrix"
                        )
                        fig.update_layout(height=700, plot_bgcolor='white', paper_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Heatmap":
                    heat_vars = st.multiselect("Select Variables", numeric_cols,
                                               default=numeric_cols[:min(10, len(numeric_cols))], key='heat_vars')

                    if st.button("Generate Plot", type="primary") and len(heat_vars) >= 2:
                        corr_matrix = df[heat_vars].corr()
                        fig = px.imshow(
                            corr_matrix,
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1,
                            title="Correlation Heatmap"
                        )
                        fig.update_traces(text=corr_matrix.round(2).values, texttemplate='%{text}')
                        fig.update_layout(height=600, plot_bgcolor='white', paper_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "3D Scatter Plot":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_3d = st.selectbox("X Variable", numeric_cols, key='3d_x')
                    with col2:
                        y_3d = st.selectbox("Y Variable", numeric_cols, key='3d_y')
                    with col3:
                        z_3d = st.selectbox("Z Variable", numeric_cols, key='3d_z')

                    color_3d = st.selectbox("Color By (optional)", ['None'] + categorical_cols + numeric_cols,
                                            key='3d_color')

                    if st.button("Generate Plot", type="primary"):
                        fig = px.scatter_3d(
                            df, x=x_3d, y=y_3d, z=z_3d,
                            color=color_3d if color_3d != 'None' else None,
                            title=f"3D Scatter Plot"
                        )
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == "Sunburst Chart":
                    if len(categorical_cols) >= 1:
                        path_vars = st.multiselect("Select Hierarchy (order matters)", categorical_cols,
                                                   default=[categorical_cols[0]], key='sun_path')
                        value_var = st.selectbox("Value Variable (optional)", ['Count'] + numeric_cols, key='sun_value')

                        if st.button("Generate Plot", type="primary") and path_vars:
                            if value_var == 'Count':
                                fig = px.sunburst(df, path=path_vars, title="Sunburst Chart")
                            else:
                                fig = px.sunburst(df, path=path_vars, values=value_var, title="Sunburst Chart")
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Sunburst charts require at least one categorical variable.")

                elif plot_type == "Treemap":
                    if len(categorical_cols) >= 1:
                        path_vars = st.multiselect("Select Hierarchy", categorical_cols,
                                                   default=[categorical_cols[0]], key='tree_path')
                        value_var = st.selectbox("Value Variable (optional)", ['Count'] + numeric_cols,
                                                 key='tree_value')

                        if st.button("Generate Plot", type="primary") and path_vars:
                            if value_var == 'Count':
                                fig = px.treemap(df, path=path_vars, title="Treemap")
                            else:
                                fig = px.treemap(df, path=path_vars, values=value_var, title="Treemap")
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Treemaps require at least one categorical variable.")

            else:  # Multiple Plots (Grid)
                st.markdown("### 🖼️ Create Multiple Plots")

                n_plots = st.slider("Number of Plots", 2, 6, 4)
                n_cols = st.slider("Columns", 1, 3, 2)

                plot_configs = []

                for i in range(n_plots):
                    with st.expander(f"📊 Plot {i + 1} Configuration", expanded=i == 0):
                        col1, col2 = st.columns(2)
                        with col1:
                            plot_type = st.selectbox(
                                "Plot Type",
                                ["Histogram", "Box Plot", "Scatter", "Line", "Violin", "Bar"],
                                key=f'multi_type_{i}'
                            )
                        with col2:
                            var1 = st.selectbox("Variable 1", numeric_cols, key=f'multi_var1_{i}')

                        var2 = None
                        if plot_type in ["Scatter", "Line"]:
                            var2 = st.selectbox("Variable 2", numeric_cols, key=f'multi_var2_{i}')

                        plot_configs.append({
                            'type': plot_type,
                            'var1': var1,
                            'var2': var2
                        })

                if st.button("Generate All Plots", type="primary"):
                    n_rows = (n_plots + n_cols - 1) // n_cols

                    fig = make_subplots(
                        rows=n_rows, cols=n_cols,
                        subplot_titles=[f"{cfg['type']}: {cfg['var1']}" for cfg in plot_configs]
                    )

                    for i, cfg in enumerate(plot_configs):
                        row = i // n_cols + 1
                        col = i % n_cols + 1

                        if cfg['type'] == "Histogram":
                            fig.add_trace(
                                go.Histogram(x=df[cfg['var1']].dropna(), name=cfg['var1'], marker_color='#8b5cf6'),
                                row=row, col=col
                            )
                        elif cfg['type'] == "Box Plot":
                            fig.add_trace(
                                go.Box(y=df[cfg['var1']].dropna(), name=cfg['var1'], marker_color='#8b5cf6'),
                                row=row, col=col
                            )
                        elif cfg['type'] == "Violin":
                            fig.add_trace(
                                go.Violin(y=df[cfg['var1']].dropna(), name=cfg['var1'], line_color='#8b5cf6'),
                                row=row, col=col
                            )
                        elif cfg['type'] == "Scatter":
                            fig.add_trace(
                                go.Scatter(x=df[cfg['var1']], y=df[cfg['var2']], mode='markers',
                                           name=f"{cfg['var1']} vs {cfg['var2']}", marker=dict(color='#8b5cf6')),
                                row=row, col=col
                            )
                        elif cfg['type'] == "Line":
                            fig.add_trace(
                                go.Scatter(x=df.index, y=df[cfg['var1']], mode='lines',
                                           name=cfg['var1'], line=dict(color='#8b5cf6')),
                                row=row, col=col
                            )
                        elif cfg['type'] == "Bar":
                            if cfg['var1'] in categorical_cols:
                                counts = df[cfg['var1']].value_counts()
                                fig.add_trace(
                                    go.Bar(x=counts.index, y=counts.values, name=cfg['var1'], marker_color='#8b5cf6'),
                                    row=row, col=col
                                )
                            else:
                                fig.add_trace(
                                    go.Bar(x=df.index[:20], y=df[cfg['var1']].head(20), name=cfg['var1'],
                                           marker_color='#8b5cf6'),
                                    row=row, col=col
                                )

                    fig.update_layout(
                        height=350 * n_rows,
                        showlegend=False,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        title_text="Multiple Plots Dashboard"
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # Download Plot Options
            st.markdown("---")
            st.markdown("### 💾 Export Options")
            st.info(
                "📌 To save individual plots, hover over the plot and click the camera icon in the top-right corner to download as PNG.")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("Please ensure your file is a valid Excel file (.xlsx or .xls) with properly formatted data.")

else:
    # Welcome Screen when no file is uploaded
    st.markdown("""
    <div class="section-card" style="text-align: center; padding: 3rem;">
        <h2 style="background: linear-gradient(135deg, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Poppins', sans-serif; font-size: 2rem;">📁 Upload Your Data to Begin</h2>
        <p style="color: #64748b; font-size: 1.1rem; margin: 1.5rem 0;">
            Use the sidebar to upload an Excel file (.xlsx) and unlock powerful data analysis tools.
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 2rem;">
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #ecfeff 0%, #f0f9ff 100%); border-radius: 16px; border-top: 4px solid #06b6d4;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <h4 style="color: #0e7490;">Descriptive Statistics</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Mean, median, variance, skewness, kurtosis, and more</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); border-radius: 16px; border-top: 4px solid #8b5cf6;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔄</div>
                <h4 style="color: #6d28d9;">Data Transformations</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Log, Box-Cox, Yeo-Johnson, standardization, and more</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%); border-radius: 16px; border-top: 4px solid #ec4899;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
                <h4 style="color: #be185d;">Normality Tests</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Shapiro-Wilk, Jarque-Bera, Anderson-Darling, and more</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border-radius: 16px; border-top: 4px solid #f97316;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔗</div>
                <h4 style="color: #c2410c;">Correlation Analysis</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Pearson, Spearman, Kendall with significance testing</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 16px; border-top: 4px solid #10b981;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧪</div>
                <h4 style="color: #047857;">Hypothesis Tests</h4>
                <p style="color: #64748b; font-size: 0.9rem;">t-tests, ANOVA, Mann-Whitney, Kruskal-Wallis, and more</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%); border-radius: 16px; border-top: 4px solid #eab308;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎨</div>
                <h4 style="color: #a16207;">Visualizations</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Interactive plots: histograms, scatter, 3D, heatmaps, and more</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div class="footer">
    <p style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">
        📊 Comprehensive Data Analysis & Visualization Platform
    </p>
    <p style="font-size: 1rem; color: rgba(255,255,255,0.9);">
        Developed by <strong style="color: #fef08a; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">Dr. Merwan Roudane</strong><br>
        Econometrician | Time Series Analyst | Statistical Software Developer
    </p>
    <p style="font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-top: 1rem;">
        Built with Streamlit, Plotly, Scipy, Statsmodels, and Scikit-learn<br>
        © 2025 All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)