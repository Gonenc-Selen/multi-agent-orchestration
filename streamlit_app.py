from pathlib import Path

import streamlit as st

from streamlit_components import renderers
from streamlit_components.loaders import filter_runs_by_mode, list_all_runs

st.set_page_config(
    page_title="Agentic Energy CPR",
    page_icon="⚡",
    layout="wide",
)

RUNS_DIR = Path(__file__).parent / "runs"


@st.cache_resource
def _load_all():
    return list_all_runs(RUNS_DIR)


all_runs = _load_all()
v1_runs = filter_runs_by_mode(all_runs, "v1")
v2_runs = filter_runs_by_mode(all_runs, "v2")
v3_runs = filter_runs_by_mode(all_runs, "v3")
v4_runs = filter_runs_by_mode(all_runs, "v4")


def _run_label(r) -> str:
    return f"{r.timestamp_str}  ({r.scenario_name})"


with st.sidebar:
    st.title("⚡ Koşu Seçimi")
    st.markdown("---")

    st.markdown("**V1 — Sessiz Sistem**")
    v1_idx = st.selectbox(
        "V1 koşusu",
        range(len(v1_runs)),
        format_func=lambda i: _run_label(v1_runs[i]),
        key="sel_v1",
        disabled=not v1_runs,
        label_visibility="collapsed",
    )

    st.markdown("**V2 — Niyet Bildirimli**")
    v2_idx = st.selectbox(
        "V2 koşusu",
        range(len(v2_runs)),
        format_func=lambda i: _run_label(v2_runs[i]),
        key="sel_v2",
        disabled=not v2_runs,
        label_visibility="collapsed",
    )

    st.markdown("**V3 — Müzakere Katmanlı**")
    v3_idx = st.selectbox(
        "V3 koşusu",
        range(len(v3_runs)),
        format_func=lambda i: _run_label(v3_runs[i]),
        key="sel_v3",
        disabled=not v3_runs,
        label_visibility="collapsed",
    )

    st.markdown("**V4 — Stres Testi**")
    v4_idx = st.selectbox(
        "V4 koşusu",
        range(len(v4_runs)),
        format_func=lambda i: _run_label(v4_runs[i]),
        key="sel_v4",
        disabled=not v4_runs,
        label_visibility="collapsed",
    )

    st.markdown("---")
    if v1_runs:
        m = v1_runs[v1_idx].metrics
        st.caption(f"V1 ihlal: {m.get('capacity_violation_count', 0)} | welfare: {m.get('total_welfare', 0):.1f}")
    if v2_runs:
        m = v2_runs[v2_idx].metrics
        st.caption(f"V2 ihlal: {m.get('capacity_violation_count', 0)} | welfare: {m.get('total_welfare', 0):.1f}")
    if v3_runs:
        m = v3_runs[v3_idx].metrics
        st.caption(f"V3 ihlal: {m.get('capacity_violation_count', 0)} | welfare: {m.get('total_welfare', 0):.1f}")
    if v4_runs:
        m = v4_runs[v4_idx].metrics
        st.caption(f"V4 ihlal: {m.get('capacity_violation_count', 0)} | welfare: {m.get('total_welfare', 0):.1f}")


v1_data = v1_runs[v1_idx] if v1_runs else None
v2_data = v2_runs[v2_idx] if v2_runs else None
v3_data = v3_runs[v3_idx] if v3_runs else None
v4_data = v4_runs[v4_idx] if v4_runs else None

tab0, tab1, tab2, tab3 = st.tabs(
    [
        "Veri Seti",
        "Yöntem",
        "Canlı Simülasyon",
        "Karşılaştırma",
    ]
)

with tab0:
    renderers.render_dataset_tab()

with tab1:
    renderers.render_method_tab()

with tab2:
    renderers.render_live_simulation_tab()

with tab3:
    renderers.render_comparison_tab()
