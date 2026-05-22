"""
AI Code Review Agent — Streamlit Dashboard
Main entry point for the interactive code review application.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from core.orchestrator import CodeReviewAgent, PipelineError
from core.models import ReviewReport, ReviewComment
from utils.export import export_markdown, export_csv, export_json
from config.settings import CATEGORIES, SEVERITIES, THEME, PROVIDERS

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Code Review Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ───────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def render_hero():
    """Render the hero section with gradient title."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔍 AI Code Review Agent</div>
        <div class="hero-subtitle">
            Autonomous code analysis powered by AST parsing & Groq AI.
            Paste a GitHub repo URL and get confidence-rated review comments in seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stat_cards(report: ReviewReport):
    """Render the stats overview cards."""
    stats = report.summary_stats()
    cols = st.columns(5)

    card_data = [
        ("📂", str(stats["files_analyzed"]), "Files Analyzed"),
        ("🔎", str(stats["total_elements"]), "Code Elements"),
        ("💬", str(stats["total_comments"]), "Issues Found"),
        ("🔴", str(stats["critical_issues"]), "Critical"),
        ("📊", f"{stats['avg_confidence']:.0f}%", "Avg Confidence"),
    ]

    for col, (icon, value, label) in zip(cols, card_data):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">{icon}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_severity_badge(severity: str) -> str:
    """Return an HTML severity badge."""
    return f'<span class="badge badge-{severity}">{severity.upper()}</span>'


def render_confidence_meter(confidence: int) -> str:
    """Return an HTML confidence meter with bar and percentage."""
    if confidence >= 70:
        bucket = "high"
    elif confidence >= 50:
        bucket = "medium"
    else:
        bucket = "low"

    color_map = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}
    color = color_map[bucket]

    return f"""
    <div class="confidence-meter confidence-{bucket}">
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width: {confidence}%; background: {color};"></div>
        </div>
        <span style="color: {color}; font-weight: 600;">{confidence}%</span>
    </div>
    """


def render_comment_card(comment: ReviewComment, show_verify: bool = False):
    """Render a single review comment as a styled card."""
    verify_class = "needs-verify" if show_verify else ""
    verify_badge = '<span class="badge badge-verify">⚠️ VERIFY THIS</span> ' if show_verify else ""

    category_display = comment.category.replace("_", " ").title()

    suggestion_html = ""
    if comment.suggestion:
        suggestion_html = f"""
        <div class="code-suggestion">
            💡 <strong>Suggestion:</strong> {comment.suggestion}
        </div>
        """

    st.markdown(f"""
    <div class="review-card severity-{comment.severity} {verify_class} animate-in">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem;">
            <div>
                {verify_badge}{render_severity_badge(comment.severity)}
                <span class="category-pill" style="margin-left: 0.4rem;">{category_display}</span>
            </div>
            {render_confidence_meter(comment.confidence)}
        </div>
        <h4 style="margin: 0 0 0.5rem; color: #f1f5f9; font-size: 1.05rem; font-weight: 600;">
            {comment.title}
        </h4>
        <div class="file-path" style="margin-bottom: 0.75rem;">
            📄 {comment.file_path} : L{comment.line_start}–{comment.line_end}
        </div>
        <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6; margin: 0;">
            {comment.description}
        </p>
        {suggestion_html}
    </div>
    """, unsafe_allow_html=True)


def render_confidence_chart(report: ReviewReport):
    """Render a confidence distribution chart."""
    if not report.comments:
        return

    high = len(report.high_confidence_comments)
    medium = len(report.medium_confidence_comments)
    low = len(report.low_confidence_comments)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[high, medium, low],
        y=["High (≥70%)", "Medium (50-69%)", "Low (<50%)"],
        orientation="h",
        marker=dict(
            color=["#22c55e", "#f59e0b", "#ef4444"],
            line=dict(width=0),
        ),
        text=[high, medium, low],
        textposition="auto",
        textfont=dict(color="white", size=14, family="Inter"),
    ))

    fig.update_layout(
        title=None,
        xaxis_title="Number of Comments",
        yaxis_title=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
        ),
        yaxis=dict(
            showgrid=False,
        ),
        height=200,
        margin=dict(l=10, r=10, t=10, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_category_chart(report: ReviewReport):
    """Render a category distribution donut chart."""
    if not report.comments:
        return

    categories = report.category_distribution
    if not categories:
        return

    labels = [k.replace("_", " ").title() for k in categories.keys()]
    values = list(categories.values())

    colors = ["#667eea", "#764ba2", "#f59e0b", "#22c55e", "#3b82f6", "#ec4899", "#8b5cf6"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors[:len(labels)]),
        textfont=dict(color="white", size=12, family="Inter"),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    )])

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        showlegend=False,
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def filter_comments(
    comments: list[ReviewComment],
    categories: list[str],
    severities: list[str],
    confidence_range: tuple[int, int],
    file_filter: str,
) -> list[ReviewComment]:
    """Apply filters to the comment list."""
    filtered = comments

    if categories:
        filtered = [c for c in filtered if c.category in categories]

    if severities:
        filtered = [c for c in filtered if c.severity in severities]

    filtered = [
        c for c in filtered
        if confidence_range[0] <= c.confidence <= confidence_range[1]
    ]

    if file_filter and file_filter != "All Files":
        filtered = [c for c in filtered if c.file_path == file_filter]

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar with input and filters."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0 1rem;">
            <div style="font-size: 1.3rem; font-weight: 700;
                 background: linear-gradient(135deg, #667eea, #764ba2);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🔍 Code Review Agent
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">Repository</div>', unsafe_allow_html=True)

        repo_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/owner/repo",
            label_visibility="collapsed",
            key="repo_url_input",
        )

        st.markdown('<div class="sidebar-title">LLM Provider</div>', unsafe_allow_html=True)
        provider_options = list(PROVIDERS.keys())
        selected_provider = st.selectbox(
            "Provider",
            options=provider_options,
            format_func=lambda x: PROVIDERS[x]["name"],
            index=0,
            key="provider_select",
            label_visibility="collapsed",
        )

        provider_info = PROVIDERS[selected_provider]

        st.markdown('<div class="sidebar-title">Model</div>', unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Model",
            options=provider_info["models"],
            index=0,
            key="model_select",
            label_visibility="collapsed",
        )

        # API Key input (if not in env)
        env_var_name = provider_info["env_key"]
        api_key_input = st.text_input(
            f"{provider_info['name']} API Key",
            type="password",
            placeholder=f"Enter key (or set {env_var_name} in .env)",
            key="api_key_input_val",
            help=f"Required for AI review using {provider_info['name']}. Set via .env file or enter here.",
        )

        # Help Card for Free API platform education
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 0.75rem; border-radius: 8px; font-size: 0.8rem; margin-top: 0.5rem; margin-bottom: 0.8rem; border-left: 3px solid #8b5cf6;">
            <strong>🎁 Tier Info:</strong> {provider_info['free_tier']}<br>
            👉 <a href="{provider_info['docs_url']}" target="_blank" style="color: #667eea; text-decoration: underline; font-weight: 500;">Get {provider_info['name']} Key</a>
        </div>
        """, unsafe_allow_html=True)

        analyze_btn = st.button(
            "🚀 Analyze Repository",
            type="primary",
            use_container_width=True,
            key="analyze_btn",
        )

        st.divider()

        # ── Filters (only if we have results) ─────────────────────────
        st.markdown('<div class="sidebar-title">Filters</div>', unsafe_allow_html=True)

        category_filter = st.multiselect(
            "Category",
            options=CATEGORIES,
            format_func=lambda x: x.replace("_", " ").title(),
            key="category_filter",
        )

        severity_filter = st.multiselect(
            "Severity",
            options=SEVERITIES,
            format_func=lambda x: x.capitalize(),
            key="severity_filter",
        )

        confidence_range = st.slider(
            "Confidence Range",
            min_value=0,
            max_value=100,
            value=(0, 100),
            key="confidence_slider",
        )

        # File filter — populated dynamically after analysis
        file_options = ["All Files"]
        if "report" in st.session_state and st.session_state.report:
            files = sorted(set(c.file_path for c in st.session_state.report.comments))
            file_options.extend(files)

        file_filter = st.selectbox(
            "File",
            options=file_options,
            key="file_filter",
        )

        st.divider()

        # ── Export Section ─────────────────────────────────────────────
        st.markdown('<div class="sidebar-title">Export</div>', unsafe_allow_html=True)

        if "report" in st.session_state and st.session_state.report:
            report = st.session_state.report

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Markdown",
                    data=export_markdown(report),
                    file_name=f"review_{report.repo_name.replace('/', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_md",
                )
            with col2:
                st.download_button(
                    "📊 CSV",
                    data=export_csv(report),
                    file_name=f"review_{report.repo_name.replace('/', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_csv",
                )

            st.download_button(
                "📦 JSON",
                data=export_json(report),
                file_name=f"review_{report.repo_name.replace('/', '_')}.json",
                mime="application/json",
                use_container_width=True,
                key="dl_json",
            )

        st.divider()

        # Footer
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; color: #64748b; font-size: 0.75rem;">
            Built with ❤️ using Streamlit + {provider_info['name']}<br>
            <a href="https://github.com" style="color: #667eea;">View on GitHub</a>
        </div>
        """, unsafe_allow_html=True)

    return repo_url, selected_provider, selected_model, api_key_input, analyze_btn, category_filter, severity_filter, confidence_range, file_filter


# ═══════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""

    # Initialize session state
    if "report" not in st.session_state:
        st.session_state.report = None
    if "logs" not in st.session_state:
        st.session_state.logs = []

    # Render sidebar
    repo_url, provider, model, api_key, analyze_btn, cat_filter, sev_filter, conf_range, file_filter = render_sidebar()

    # Render hero
    render_hero()

    # ── Run Analysis ──────────────────────────────────────────────────
    if analyze_btn:
        if not repo_url or not repo_url.strip():
            st.error("⚠️ Please enter a GitHub repository URL.")
            return

        st.session_state.report = None
        st.session_state.logs = []

        # Create the agent
        try:
            agent = CodeReviewAgent(provider=provider, api_key=api_key, model=model)
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {e}")
            return

        # Run the pipeline with live progress
        log_container = st.container()
        progress_bar = st.progress(0, text="Starting analysis...")

        log_messages: list[str] = []

        def on_progress(msg: str):
            log_messages.append(msg)
            with log_container:
                with st.expander("📋 Pipeline Logs", expanded=True):
                    st.markdown("\n\n".join(log_messages))

        def on_step(step_name: str, fraction: float):
            progress_bar.progress(min(fraction, 1.0), text=f"{step_name}...")

        try:
            with st.spinner("🤖 AI agent is analyzing the repository..."):
                report = agent.run(
                    repo_url=repo_url.strip(),
                    progress_callback=on_progress,
                    step_callback=on_step,
                )
            st.session_state.report = report
            st.session_state.logs = log_messages
            progress_bar.progress(1.0, text="✅ Analysis complete!")
            st.rerun()

        except PipelineError as e:
            progress_bar.empty()
            st.error(f"❌ Pipeline Error: {e}")
            return
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Unexpected error: {e}")
            return

    # ── Display Results ───────────────────────────────────────────────
    report = st.session_state.report

    if report is None:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🏗️</div>
            <div class="empty-state-text">
                Enter a GitHub repository URL in the sidebar and click
                <strong>Analyze Repository</strong> to get started.
            </div>
            <div style="color: #64748b; font-size: 0.85rem; margin-top: 1rem;">
                The agent will clone the repo, parse code via AST, and generate
                AI-powered review comments with confidence scores.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Show past logs
    if st.session_state.logs:
        with st.expander("📋 Pipeline Logs", expanded=False):
            st.markdown("\n\n".join(st.session_state.logs))

    # Stats cards
    render_stat_cards(report)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    if report.comments:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown(
                '<div class="section-header">📊 Confidence Distribution</div>',
                unsafe_allow_html=True,
            )
            render_confidence_chart(report)

        with chart_col2:
            st.markdown(
                '<div class="section-header">🏷️ Categories Breakdown</div>',
                unsafe_allow_html=True,
            )
            render_category_chart(report)

    # ── Apply Filters ─────────────────────────────────────────────────
    filtered = filter_comments(
        report.comments,
        cat_filter,
        sev_filter,
        conf_range,
        file_filter,
    )

    if not filtered and report.comments:
        st.info("🔍 No comments match the current filters. Try adjusting them in the sidebar.")
        return

    if not filtered:
        st.success("✨ No issues found in this repository. The code looks clean!")
        return

    # ── High Confidence Findings ──────────────────────────────────────
    high_conf = [c for c in filtered if c.confidence >= 70]
    if high_conf:
        st.markdown(
            '<div class="section-header">✅ High Confidence Findings</div>'
            '<div class="section-description">'
            'These issues have been identified with high confidence (≥70%) and are very likely genuine.'
            '</div>',
            unsafe_allow_html=True,
        )
        for comment in sorted(high_conf, key=lambda c: (
            {"critical": 0, "warning": 1, "info": 2, "suggestion": 3}.get(c.severity, 4),
            -c.confidence
        )):
            render_comment_card(comment)

    # ── Medium Confidence ─────────────────────────────────────────────
    med_conf = [c for c in filtered if 50 <= c.confidence < 70]
    if med_conf:
        st.markdown(
            '<div class="section-header">🟡 Medium Confidence</div>'
            '<div class="section-description">'
            'These are probable issues. Consider reviewing them in context.'
            '</div>',
            unsafe_allow_html=True,
        )
        for comment in sorted(med_conf, key=lambda c: -c.confidence):
            render_comment_card(comment)

    # ── Low Confidence (Needs Verification) ───────────────────────────
    low_conf = [c for c in filtered if c.confidence < 50]
    if low_conf:
        st.markdown(
            '<div class="section-header">⚠️ Needs Verification</div>'
            '<div class="section-description">'
            'These findings have lower confidence scores (&lt;50%) and may need human review. '
            'They could be false positives or context-dependent issues.'
            '</div>',
            unsafe_allow_html=True,
        )
        for comment in sorted(low_conf, key=lambda c: -c.confidence):
            render_comment_card(comment, show_verify=True)


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
