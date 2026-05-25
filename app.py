"""
app.py
-----------------
Main Streamlit application. Ties together all modules:
    loader.py -> file upload and validation
    analyzer.py -> statistics, correlations, anomalies
    visualizer.py -> interactive Plotly charts
    forecaster.py -> time-series forecasting
    agent.py -> natural-language Q&A via Gemini

Tab structure:
    Overview -> schema table + line charts (time series & trends)
    Charts -> bar charts + scatter + custom builder
    Advanced -> distributions, correlation, missing values, anomalies
    Forecast -> time-series forecasting
    Ask -> natural language Q&A

Run locally:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Data_Talk",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.loader import load_file, schema_dataframe
from modules.analyzer import analyze
from modules.visualizer import (
    auto_charts,
    plot_distribution,
    plot_categorical_bar,
    plot_scatter,
    plot_time_series,
    plot_correlation_heatmap,
    plot_missing_values,
)
from modules.forecaster import forecast, get_forecastable_pairs
from modules.agent import (
    build_context,
    ask,
    check_limit,
    suggest_questions,
    ConversationHistory,
    QUESTION_LIMIT,
)


# --------------------
#  Load external CSS
# --------------------

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --------------------
#  Session state
# --------------------

def _init_session() -> None:
    defaults = {
        "dataset": None,
        "report": None,
        "context": None,
        "history": ConversationHistory(),
        "chat_messages": [],
        "data_talk_question_count": 0,
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session()


# --------------------
#  Sidebar
# --------------------

with st.sidebar:
    st.markdown("## Data_Talk")
    st.caption("Upload your data. Ask anything.")
    st.divider()

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Max recommended size: 50 MB",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded:
        if (
            st.session_state.dataset is None
            or st.session_state.dataset.meta.get("filename") != uploaded.name
        ):
            with st.spinner("Loading and analysing…"):
                dataset = load_file(uploaded)

                if not dataset.ok:
                    st.error(dataset.error)
                    st.stop()

                report  = analyze(dataset)
                context = build_context(dataset, report)

                st.session_state.dataset = dataset
                st.session_state.report = report
                st.session_state.context = context
                st.session_state.history = ConversationHistory()
                st.session_state.chat_messages = []
                st.session_state.datatalk_question_count = 0

            st.success(f"✓ {uploaded.name} loaded")

    if st.session_state.dataset:
        meta = st.session_state.dataset.meta
        st.divider()
        st.markdown("**Dataset info**")
        col1, col2 = st.columns(2)
        col1.metric("Rows", f"{meta['n_rows']:,}")
        col2.metric("Columns", meta['n_cols'])
        col1.metric("Missing", f"{meta['pct_missing']}%")
        col2.metric("Size", f"{meta['memory_kb']} KB")

        if st.session_state.context:
            mode = st.session_state.context.data_mode
            mode_labels = {
                "full": "Full data sent to AI",
                "sample": "Stratified sample sent",
                "stats_only": "Large file — sample only",
            }
            st.caption(mode_labels.get(mode, ""))

        st.divider()

        if st.button("Clear & upload new file", use_container_width=True):
            for key in ["dataset", "report", "context"]:
                st.session_state[key] = None
            st.session_state.chat_messages = []
            st.session_state.history = ConversationHistory()
            st.session_state.datatalk_question_count = 0
            st.session_state.uploader_key += 1
            st.rerun()


# ----------------------------------
#  Landing page — no file uploaded
# ----------------------------------

if st.session_state.dataset is None:
    st.markdown("# Data_Talk")
    st.markdown("### Talk to your data in plain language.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Upload**")
        st.markdown("CSV or Excel file, any size. Data_Talk adapts automatically.")
    with col2:
        st.markdown("**Explore**")
        st.markdown("Automatic charts, statistics, anomaly detection.")
    with col3:
        st.markdown("**Ask**")
        st.markdown("Ask questions in any language. Get answers from your data.")

    st.info("Upload a file from the sidebar to get started.")
    st.stop()


# --------------------------------------------
#  Guard — safety check before rendering tabs
# --------------------------------------------

dataset = st.session_state.dataset
report  = st.session_state.report
context = st.session_state.context

if dataset is None or report is None or context is None:
    st.info("Upload a file from the sidebar to get started.")
    st.stop()


# -------
#  Tabs
# -------

tab_overview, tab_charts, tab_advanced, tab_forecast, tab_chat = st.tabs([
    "Overview",
    "Charts",
    "Advanced",
    "Forecast",
    "Ask Data_Talk",
])


# =============================================
#  TAB 1 — Overview
#  Schema table + key insights + line charts
# =============================================

with tab_overview:

    # -- Schema --
    st.markdown('<p class="section-title">Schema</p>', unsafe_allow_html=True)
    st.dataframe(schema_dataframe(dataset), use_container_width=True, height=280)

    st.divider()

    # -- Key insights --
    st.markdown('<p class="section-title">Key Insights</p>', unsafe_allow_html=True)
    for insight in report.insights:
        st.markdown(f'<div class="insight-pill">{insight}</div>', unsafe_allow_html=True)

    st.divider()

    # -- Line charts: time series for every datetime × numeric pair --
    if dataset.datetime_cols and dataset.numeric_cols:
        st.markdown('<p class="section-title">Trends over time</p>', unsafe_allow_html=True)
        date_col = dataset.datetime_cols[0]
        for i, val_col in enumerate(dataset.numeric_cols[:4]):
            fig = plot_time_series(dataset.df, date_col, val_col)
            st.plotly_chart(fig, use_container_width=True, key=f"ov_ts_{i}")

    elif dataset.numeric_cols:
        # No datetime column — show line chart of numeric columns by index
        st.markdown('<p class="section-title">Numeric trends (by row index)</p>', unsafe_allow_html=True)
        import plotly.graph_objects as go
        for i, col in enumerate(dataset.numeric_cols[:4]):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=dataset.df[col].values,
                mode="lines",
                line=dict(color="#00c9a7", width=2),
                name=col,
                hovertemplate=f"Row %{{x}}<br>{col}: %{{y:.2f}}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"Trend — {col}", font=dict(size=15)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", size=12),
                margin=dict(l=40, r=40, t=50, b=40),
            )
            fig.update_xaxes(title_text="Row index")
            fig.update_yaxes(title_text=col)
            st.plotly_chart(fig, use_container_width=True, key=f"ov_line_{i}")
    else:
        st.info("No numeric columns available for line charts.")

    # -- Raw data preview --
    st.divider()
    with st.expander("Raw data preview (first 100 rows)"):
        st.dataframe(dataset.df.head(100), use_container_width=True)


# =============================================
#  TAB 2 — Charts
#  Bar charts + scatter + custom builder
# =============================================

with tab_charts:

    # -- Categorical bar charts (auto) --
    if dataset.categorical_cols:
        st.markdown('<p class="section-title">Category Distribution</p>', unsafe_allow_html=True)
        for i, col in enumerate(dataset.categorical_cols[:4]):
            fig = plot_categorical_bar(dataset.df, col)
            st.plotly_chart(fig, use_container_width=True, key=f"ch_bar_{i}")
            st.divider()

    # -- Scatter plots (auto — first 2 numeric pairs) --
    if len(dataset.numeric_cols) >= 2:
        st.markdown('<p class="section-title">Relationships</p>', unsafe_allow_html=True)
        pairs_shown = 0
        for i in range(min(3, len(dataset.numeric_cols) - 1)):
            x_col = dataset.numeric_cols[i]
            y_col = dataset.numeric_cols[i + 1]
            color = dataset.categorical_cols[0] if dataset.categorical_cols else None
            fig = plot_scatter(dataset.df, x_col, y_col, color_col=color)
            st.plotly_chart(fig, use_container_width=True, key=f"ch_sc_{i}")
            pairs_shown += 1
            if pairs_shown >= 2:
                break
        st.divider()

    # -- Custom chart builder --
    st.markdown('<p class="section-title">Custom Chart</p>', unsafe_allow_html=True)
    chart_type = st.selectbox(
        "Chart type",
        ["Bar — category counts", "Scatter plot", "Line — time series"],
        key="chart_type_sel",
    )

    if chart_type == "Bar — category counts":
        if dataset.categorical_cols:
            col   = st.selectbox("Column", dataset.categorical_cols, key="cust_bar_col")
            top_n = st.slider("Show top N", 5, 30, 15, key="cust_bar_n")
            st.plotly_chart(
                plot_categorical_bar(dataset.df, col, top_n),
                use_container_width=True, key="cust_bar_fig"
            )
        else:
            st.info("No categorical columns found.")

    elif chart_type == "Scatter plot":
        if len(dataset.numeric_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            x_col  = c1.selectbox("X axis",    dataset.numeric_cols, key="cust_sc_x")
            y_col  = c2.selectbox("Y axis",    dataset.numeric_cols, index=1, key="cust_sc_y")
            color  = c3.selectbox("Colour by", ["None"] + dataset.categorical_cols, key="cust_sc_c")
            st.plotly_chart(
                plot_scatter(dataset.df, x_col, y_col,
                             color_col=None if color == "None" else color),
                use_container_width=True, key="cust_sc_fig"
            )
        else:
            st.info("Need at least 2 numeric columns for a scatter plot.")

    elif chart_type == "Line — time series":
        if dataset.datetime_cols and dataset.numeric_cols:
            c1, c2 = st.columns(2)
            date_col = c1.selectbox("Date column",  dataset.datetime_cols, key="cust_ts_d")
            val_col  = c2.selectbox("Value column", dataset.numeric_cols,  key="cust_ts_v")
            st.plotly_chart(
                plot_time_series(dataset.df, date_col, val_col),
                use_container_width=True, key="cust_ts_fig"
            )
        else:
            st.info("Need at least one datetime and one numeric column.")


# =================================================
#  TAB 3 — Advanced
#  Distributions, correlation, missing, anomalies
# =================================================

with tab_advanced:

    # -- Descriptive statistics --
    st.markdown('<p class="section-title">Descriptive Statistics</p>', unsafe_allow_html=True)
    if not report.summary_df.empty:
        st.dataframe(report.summary_df, use_container_width=True)
    else:
        st.info("No numeric columns found.")

    # -- Distributions --
    if dataset.numeric_cols:
        st.divider()
        st.markdown('<p class="section-title">Distributions</p>', unsafe_allow_html=True)
        for i, col in enumerate(dataset.numeric_cols[:6]):
            fig = plot_distribution(dataset.df, col)
            st.plotly_chart(fig, use_container_width=True, key=f"adv_dist_{i}")

    # -- Correlation heatmap --
    if report.correlation_df is not None:
        st.divider()
        st.markdown('<p class="section-title">Correlation Matrix</p>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_correlation_heatmap(report.correlation_df),
            use_container_width=True, key="adv_corr"
        )

    # -- Missing values --
    if not report.missing_df.empty:
        st.divider()
        st.markdown('<p class="section-title">Missing Values</p>', unsafe_allow_html=True)
        missing_fig = plot_missing_values(report.missing_df)
        if missing_fig:
            st.plotly_chart(missing_fig, use_container_width=True, key="adv_miss")
        st.dataframe(report.missing_df, use_container_width=True)
    else:
        st.divider()
        st.success("✓ No missing values detected.")

    # -- Anomalies --
    st.divider()
    st.markdown('<p class="section-title">Anomaly Detection (IQR)</p>', unsafe_allow_html=True)
    if not report.anomalies_df.empty:
        n   = len(report.anomalies_df)
        pct = round(n / len(dataset.df) * 100, 1)
        st.markdown(
            f'<div class="anomaly-warning">{n} rows flagged as anomalies ({pct}% of data). '
            f'The <b>anomaly_in</b> column shows which field triggered the flag.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(report.anomalies_df, use_container_width=True, height=280)
    else:
        st.success("✓ No anomalies detected.")


# ====================
#  TAB 4 — Forecast
# ====================

with tab_forecast:
    pairs = get_forecastable_pairs(dataset)

    if not pairs:
        st.info(
            "No forecastable column pairs found. "
            "Data_Talk needs at least one datetime column and one numeric column."
        )
    else:
        st.markdown('<p class="section-title">Time-Series Forecast</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        date_cols = list({p[0] for p in pairs})
        val_cols = list({p[1] for p in pairs})

        date_col = col1.selectbox("Date column",  date_cols, key="fc_date")
        value_col = col2.selectbox("Value column", val_cols,  key="fc_val")
        periods = col3.slider("Periods to forecast", 7, 365, 30, key="fc_periods")

        if st.button("▶ Run Forecast", type="primary", key="fc_btn"):
            with st.spinner("Forecasting…"):
                result = forecast(dataset, date_col, value_col, periods)

            if result.ok:
                model_label = "Prophet" if result.model_used == "prophet" else "Linear Regression"
                st.caption(f"Model used: **{model_label}**")
                st.plotly_chart(result.fig, use_container_width=True, key="fc_fig")
                with st.expander("Forecast table"):
                    st.dataframe(result.forecast_df, use_container_width=True)
            else:
                st.error(result.error)


# ================
#  TAB 5 — Chat
# ================

with tab_chat:
    allowed, used, remaining = check_limit(st.session_state)
    col1, col2 = st.columns([3, 1])
    col1.markdown('<p class="section-title">Ask Data_Talk</p>', unsafe_allow_html=True)
    col2.markdown(
        f'<div style="text-align:right;padding-top:0.5rem">'
        f'<span class="limit-badge">{remaining}/{QUESTION_LIMIT} questions left</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_messages:
        st.markdown("**Quick start — click a question:**")
        suggestions = suggest_questions(dataset, report)
        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["_pending_question"] = q
                st.rerun()

    for role, text in st.session_state.chat_messages:
        css_class = "chat-user" if role == "user" else "chat-assistant"
        prefix = "You" if role == "user" else "Data_Talk"
        st.markdown(
            f'<div class="{css_class}"><b>{prefix}</b><br>{text}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    prefill = st.session_state.get("_pending_question", "")
    if "_pending_question" in st.session_state:
        del st.session_state["_pending_question"]

    question = st.chat_input(placeholder="Ask anything about your data…") or prefill

    if question:
        if not allowed:
            st.warning(
                f"You have reached the session limit of **{QUESTION_LIMIT} questions**. "
                "Reload the page to start a new session."
            )
        else:
            st.session_state.chat_messages.append(("user", question))
            with st.spinner("Data_Talk is thinking…"):
                answer = ask(
                    question = question,
                    context = context,
                    history = st.session_state.history,
                    session_state = st.session_state,
                )
            st.session_state.chat_messages.append(("assistant", answer))
            st.rerun()