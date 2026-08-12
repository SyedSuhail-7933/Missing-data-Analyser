import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Missing Data Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    .auth-box {
        max-width: 420px;
        margin: 60px auto;
        padding: 40px;
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
    }
    .auth-title {
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .auth-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 28px;
    }
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        padding: 14px 16px !important;
        font-size: 14px !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(59, 130, 246, 0.35) !important;
    }
    .main-header {
        text-align: center;
        padding: 10px 0 25px 0;
    }
    .main-title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 22px 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(59, 130, 246, 0.25);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 34px;
        font-weight: 700;
        color: #60a5fa;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 6px;
    }
    .section-header {
        font-size: 22px;
        font-weight: 600;
        color: #e2e8f0;
        margin: 28px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(59, 130, 246, 0.25);
    }
    .stDataFrame td, .stDataFrame th {
        background: rgba(15, 23, 42, 0.5) !important;
        color: #e2e8f0 !important;
        border-color: rgba(148, 163, 184, 0.08) !important;
        font-size: 13px !important;
    }
    .stDataFrame th {
        background: rgba(59, 130, 246, 0.15) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 11px !important;
        letter-spacing: 0.5px;
    }
    .stSidebar {
        background: rgba(15, 23, 42, 0.95) !important;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #64748b;
        font-size: 12px;
        margin-top: 40px;
        border-top: 1px solid rgba(148, 163, 184, 0.08);
    }
    div[role="radiogroup"] {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 4px;
    }
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    .stFileUploader > div > button {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 2px dashed rgba(59, 130, 246, 0.3) !important;
        border-radius: 16px !important;
        color: #94a3b8 !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #60a5fa !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    defaults = {
        'authenticated': False,
        'user': None,
        'users_db': {'admin': 'admin123', 'demo': 'demo123'},
        'df': None,
        'df_cleaned': None,
        'missing_stats': None,
        'custom_fill_value': '0',
        'login_error': None,
        'signup_error': None,
        'signup_success': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================================================
# AUTH
# =============================================================================
def login_user(username, password):
    users_db = st.session_state.get('users_db', {})
    if username in users_db and users_db[username] == password:
        st.session_state.authenticated = True
        st.session_state.user = username
        st.session_state.login_error = None
        return True
    st.session_state.login_error = "Invalid username or password"
    return False

def signup_user(username, password, confirm_password):
    users_db = st.session_state.get('users_db', {})
    if username in users_db:
        st.session_state.signup_error = "Username already exists"
        return False
    if len(username) < 3:
        st.session_state.signup_error = "Username must be at least 3 characters"
        return False
    if len(password) < 6:
        st.session_state.signup_error = "Password must be at least 6 characters"
        return False
    if password != confirm_password:
        st.session_state.signup_error = "Passwords do not match"
        return False
    users_db[username] = password
    st.session_state.users_db = users_db
    st.session_state.signup_error = None
    st.session_state.signup_success = "Account created! Please login."
    return True

def logout():
    keys_to_clear = ['authenticated', 'user', 'df', 'df_cleaned', 'missing_stats',
                     'login_error', 'signup_error', 'signup_success']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()

# =============================================================================
# LOGIN PAGE
# =============================================================================
def show_login_page():
    st.markdown('<div class="auth-box">', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <div style="font-size: 44px; margin-bottom: 8px;">🔍</div>
            <div class="auth-title">Missing Data Analyzer</div>
            <div class="auth-subtitle">Professional Data Quality Assessment</div>
        </div>
    """, unsafe_allow_html=True)

    tab = st.radio("", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed", key="auth_tab_radio")
    st.markdown("<br>", unsafe_allow_html=True)

    if tab == "Login":
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")
            submitted = st.form_submit_button("Sign In")

            if submitted:
                if not username or not password:
                    st.error("⚠️ Please fill in all fields")
                elif login_user(username, password):
                    st.success("✅ Welcome back!")
                    st.balloons()
                    st.experimental_rerun()
                else:
                    st.error(f"❌ {st.session_state.get('login_error', 'Login failed')}")
    else:
        with st.form("signup_form", clear_on_submit=False):
            new_username = st.text_input("Username", placeholder="Choose a username (min 3 chars)", key="signup_user")
            new_password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="signup_pass")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="signup_confirm")
            submitted = st.form_submit_button("Create Account")

            if submitted:
                if not new_username or not new_password or not confirm_password:
                    st.error("⚠️ Please fill in all fields")
                elif signup_user(new_username, new_password, confirm_password):
                    st.success(f"✅ {st.session_state.get('signup_success', 'Account created!')}")
                else:
                    st.error(f"❌ {st.session_state.get('signup_error', 'Signup failed')}")

    st.markdown("""
        <div style="margin-top: 20px; padding: 14px; background: rgba(59, 130, 246, 0.08); border-radius: 12px; border-left: 3px solid #3b82f6;">
            <p style="margin: 0; font-size: 12px; color: #94a3b8;">
                <strong style="color: #60a5fa;">Demo:</strong> 
                username <code style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px;">admin</code> / 
                password <code style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px;">admin123</code>
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# DATA ANALYSIS
# =============================================================================
def analyze_missing_data(df):
    missing_counts = df.isnull().sum()
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    total_missing = int(missing_counts.sum())
    cols_with_missing = int((missing_counts > 0).sum())
    return {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'total_missing': total_missing,
        'cols_with_missing': cols_with_missing,
        'missing_counts': missing_counts,
        'missing_percentages': missing_percentages,
        'complete_rows': df.dropna().shape[0]
    }

def get_severity(pct):
    if pct == 0: return "CLEAN", "#10b981"
    elif pct < 10: return "LOW", "#10b981"
    elif pct < 30: return "MEDIUM", "#f59e0b"
    else: return "HIGH", "#ef4444"

def clean_data(df, method, custom_value='0'):
    df_clean = df.copy()
    if method == "Drop rows with any missing values":
        df_clean = df_clean.dropna()
    elif method == "Drop rows with all missing values":
        df_clean = df_clean.dropna(how='all')
    elif method == "Drop columns with missing values":
        df_clean = df_clean.dropna(axis=1)
    elif method == "Fill with Mean (Numeric only)":
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif method == "Fill with Median (Numeric only)":
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif method == "Fill with Mode":
        for col in df_clean.columns:
            if df_clean[col].isnull().any() and not df_clean[col].mode().empty:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    elif method == "Fill with Forward Fill":
        df_clean = df_clean.ffill()
    elif method == "Fill with Backward Fill":
        df_clean = df_clean.bfill()
    elif method == "Fill with Custom Value":
        df_clean = df_clean.fillna(custom_value)
    elif method == "Interpolate":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate()
    return df_clean

# =============================================================================
# DASHBOARD
# =============================================================================
def show_dashboard():
    with st.sidebar:
        user = st.session_state.get('user', 'User')
        initial = user[0].upper() if user else 'U'
        st.markdown(f"""
            <div style="padding: 20px; background: rgba(30, 41, 59, 0.6); border-radius: 16px; margin: 10px; text-align: center;">
                <div style="width: 56px; height: 56px; border-radius: 50%; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto; font-size: 22px; font-weight: 700; color: white;">{initial}</div>
                <div style="font-weight: 600; font-size: 15px;">{user}</div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">Data Analyst</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            logout()
            st.experimental_rerun()

    st.markdown("""
        <div class="main-header">
            <div class="main-title">🔍 Missing Data Analyzer</div>
            <p style="color: #94a3b8; font-size: 15px; margin-top: 6px;">Upload your dataset to identify, analyze, and clean missing values</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        key="file_uploader"
    )

    use_demo = st.checkbox("Use Demo Dataset (Titanic-style with missing values)", value=False, key="demo_checkbox")

    if use_demo and st.session_state.get('df') is None:
        np.random.seed(42)
        demo_data = {
            'PassengerId': range(1, 101),
            'Name': [f'Passenger_{i}' for i in range(1, 101)],
            'Age': np.where(np.random.random(100) > 0.8, np.nan, np.random.normal(30, 12, 100)),
            'Fare': np.where(np.random.random(100) > 0.85, np.nan, np.random.exponential(30, 100)),
            'Embarked': np.where(np.random.random(100) > 0.9, np.nan, np.random.choice(['S', 'C', 'Q'], 100)),
            'Survived': np.random.choice([0, 1], 100),
            'Pclass': np.random.choice([1, 2, 3], 100),
            'Cabin': np.where(np.random.random(100) > 0.7, np.nan, np.random.choice(['A1', 'B2', 'C3', 'D4'], 100))
        }
        st.session_state.df = pd.DataFrame(demo_data)
        st.session_state.missing_stats = analyze_missing_data(st.session_state.df)
        st.success("✅ Demo dataset loaded with intentional missing values!")

    if uploaded_file is not None and st.session_state.get('df') is None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.session_state.missing_stats = analyze_missing_data(st.session_state.df)
            st.success(f"✅ Loaded: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    if st.session_state.get('df') is not None:
        c1, c2 = st.columns([1, 8])
        with c1:
            if st.button("🔄 Reset", key="reset_btn"):
                st.session_state.df = None
                st.session_state.df_cleaned = None
                st.session_state.missing_stats = None
                st.experimental_rerun()

    if st.session_state.get('df') is not None:
        df = st.session_state.df
        stats = st.session_state.missing_stats

        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

        completeness = round((stats['complete_rows'] / stats['total_rows']) * 100, 1) if stats['total_rows'] > 0 else 0
        comp_color = "#10b981" if completeness > 90 else "#f59e0b" if completeness > 70 else "#ef4444"
        miss_color = "#ef4444" if stats['total_missing'] > 0 else "#10b981"

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["total_rows"]:,}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["total_cols"]}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {miss_color};">{stats["total_missing"]:,}</div><div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["cols_with_missing"]}</div><div class="metric-label">Affected Cols</div></div>', unsafe_allow_html=True)
        with m5: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {comp_color};">{completeness}%</div><div class="metric-label">Completeness</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">🔎 Missing Values by Column</div>', unsafe_allow_html=True)

        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(dt) for dt in df.dtypes.values],
            'Missing Count': stats['missing_counts'].values,
            'Missing %': stats['missing_percentages'].values.round(2),
            'Non-Null Count': df.count().values
        })
        missing_df['Severity'] = missing_df['Missing %'].apply(lambda x: get_severity(x)[0])

        def color_severity(val):
            cmap = {
                'CLEAN': 'background-color: rgba(16, 185, 129, 0.15); color: #10b981;',
                'LOW': 'background-color: rgba(16, 185, 129, 0.15); color: #10b981;',
                'MEDIUM': 'background-color: rgba(245, 158, 11, 0.15); color: #f59e0b;',
                'HIGH': 'background-color: rgba(239, 68, 68, 0.15); color: #ef4444;'
            }
            return cmap.get(val, '')

        st.dataframe(missing_df.style.applymap(color_severity, subset=['Severity']), use_container_width=True, height=320)

        st.markdown('<div class="section-header">📈 Visual Analysis</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)

        with v1:
            missing_cols = stats['missing_counts'][stats['missing_counts'] > 0]
            if len(missing_cols) > 0:
                fig = px.bar(
                    x=missing_cols.index, y=missing_cols.values,
                    labels={'x': 'Column', 'y': 'Missing Count'},
                    title="Missing Values by Column",
                    color=missing_cols.values,
                    color_continuous_scale=['#10b981', '#f59e0b', '#ef4444']
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', title_x=0.5, margin=dict(t=40, b=40))
                fig.update_xaxes(gridcolor='rgba(148,163,184,0.1)')
                fig.update_yaxes(gridcolor='rgba(148,163,184,0.1)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("✨ Your dataset is completely clean — no missing values!")

        with v2:
            labels = ['Complete Rows', 'Rows with Missing']
            values = [stats['complete_rows'], stats['total_rows'] - stats['complete_rows']]
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.65, marker_colors=['#10b981', '#ef4444'], textinfo='percent', textfont_size=14)])
            fig.update_layout(title="Completeness Distribution", paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', title_x=0.5, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15), margin=dict(t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

        if stats['cols_with_missing'] > 0:
            st.markdown('<div class="section-header">🌡️ Missing Value Heatmap</div>', unsafe_allow_html=True)
            missing_matrix = df.isnull().astype(int)
            missing_cols_names = stats['missing_counts'][stats['missing_counts'] > 0].index.tolist()
            if missing_cols_names:
                subset = missing_matrix[missing_cols_names].head(100)
                fig = px.imshow(subset, color_continuous_scale=['rgba(30,41,59,0.8)', '#ef4444'], title="Pattern (First 100 Rows)", aspect='auto')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', title_x=0.5, coloraxis_showscale=False, margin=dict(t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">🧹 Data Cleaning</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 3])

        with c1:
            methods = [
                "Drop rows with any missing values",
                "Drop rows with all missing values",
                "Drop columns with missing values",
                "Fill with Mean (Numeric only)",
                "Fill with Median (Numeric only)",
                "Fill with Mode",
                "Fill with Forward Fill",
                "Fill with Backward Fill",
                "Fill with Custom Value",
                "Interpolate"
            ]
            method = st.selectbox("Cleaning Method", methods, key="clean_method")

            custom_val = '0'
            if method == "Fill with Custom Value":
                custom_val = st.text_input("Custom value:", value=st.session_state.get('custom_fill_value', '0'), key="custom_val")
                st.session_state.custom_fill_value = custom_val

            if st.button("🚀 Apply Cleaning", use_container_width=True, key="apply_clean"):
                with st.spinner("Processing..."):
                    cv = st.session_state.get('custom_fill_value', '0') if method == "Fill with Custom Value" else '0'
                    st.session_state.df_cleaned = clean_data(df, method, cv)
                    st.success("✅ Cleaning applied!")

        with c2:
            if st.session_state.get('df_cleaned') is not None:
                cd = st.session_state.df_cleaned
                orig_miss = stats['total_missing']
                new_miss = int(cd.isnull().sum().sum())
                removed = orig_miss - new_miss
                st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.08); padding: 20px; border-radius: 16px; border: 1px solid rgba(16, 185, 129, 0.25);">
                        <h4 style="margin-top: 0; color: #10b981; font-size: 16px;">✨ Cleaning Results</h4>
                        <p style="color: #94a3b8; font-size: 13px; margin: 6px 0;">Original missing: <strong style="color: #ef4444;">{orig_miss:,}</strong></p>
                        <p style="color: #94a3b8; font-size: 13px; margin: 6px 0;">Remaining missing: <strong style="color: {'#10b981' if new_miss == 0 else '#f59e0b'};">{new_miss:,}</strong></p>
                        <p style="color: #94a3b8; font-size: 13px; margin: 6px 0;">Values handled: <strong style="color: #60a5fa;">{removed:,}</strong></p>
                        <p style="color: #94a3b8; font-size: 13px; margin: 6px 0;">Rows after cleaning: <strong>{len(cd):,}</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Select a method and click Apply Cleaning to see results")

        if st.session_state.get('df_cleaned') is not None:
            st.markdown('<div class="section-header">✅ Cleaned Dataset</div>', unsafe_allow_html=True)
            t1, t2 = st.tabs(["📋 Preview", "📥 Export"])

            with t1:
                st.dataframe(st.session_state.df_cleaned.head(100), use_container_width=True)
                comp_data = pd.DataFrame({
                    'Metric': ['Missing (Before)', 'Missing (After)', 'Rows (Before)', 'Rows (After)'],
                    'Count': [stats['total_missing'], st.session_state.df_cleaned.isnull().sum().sum(), len(df), len(st.session_state.df_cleaned)],
                    'Type': ['Before', 'After', 'Before', 'After']
                })
                fig = px.bar(comp_data, x='Metric', y='Count', color='Type', color_discrete_map={'Before': '#ef4444', 'After': '#10b981'}, title="Before vs After Cleaning", barmode='group')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', title_x=0.5, margin=dict(t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                fmt = st.radio("Format", ["CSV", "JSON"], horizontal=True, key="export_fmt")
                if fmt == "CSV":
                    csv = st.session_state.df_cleaned.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, "cleaned_dataset.csv", "text/csv", use_container_width=True)
                else:
                    json_str = st.session_state.df_cleaned.to_json(orient='records', indent=2)
                    st.download_button("📥 Download JSON", json_str, "cleaned_dataset.json", "application/json", use_container_width=True)

    st.markdown('<div class="footer"><p>🔍 Missing Data Analyzer v1.0 | Professional Data Quality Tool</p></div>', unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    load_css()
    init_session_state()
    if not st.session_state.get('authenticated', False):
        show_login_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()