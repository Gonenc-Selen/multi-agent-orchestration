from __future__ import annotations

import datetime
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_components.loaders import RunData, get_cumulative, is_promise_kept, load_run_by_id

AGENT_IDS = ["H1", "H6", "H8"]
AGENT_COLORS = {"H1": "#E63946", "H6": "#06A77D", "H8": "#4361EE"}
AGENT_ICONS = {"H1": "🔌", "H6": "🌿", "H8": "⚡"}
AGENT_LABELS = {
    "H1": "H1 — Geniş Aile",
    "H6": "H6 — Küçük Hane",
    "H8": "H8 — Büyük Hane",
}
AGENT_PERSONAS = {
    "H1": "4-5 kişilik kalabalık aile. Orta-yüksek tüketim, iklim kontrolü yoğun. 5 m² panel, pil kapasitesi 10 kWh.",
    "H6": "1-2 kişilik küçük ve verimli hane. Düşük ve kararlı tüketim. 3 m² panel, pil kapasitesi 6 kWh.",
    "H8": "Büyük hane, üçün en büyük paneli (8 m²). Yüksek tüketim, stratejik ve öz-çıkar odaklı. Pil kapasitesi 15 kWh.",
}
VERSION_COLORS = {"v1": "#E63946", "v2": "#06A77D", "v3": "#4361EE"}

COMPARISON_RUN_IDS: dict[str, str] = {
    "v1": "20260530T163112",
    "v2": "20260530T165456",
    "v3": "20260530T183953",
    "v4": "20260530T201035",
}
_COMP_RUNS_DIR = Path(__file__).parent.parent / "runs"


def _bg_color(hex_color: str, alpha: float = 0.08) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _card(agent_id: str, body_html: str) -> None:
    color = AGENT_COLORS.get(agent_id, "#888")
    bg = _bg_color(color)
    icon = AGENT_ICONS.get(agent_id, "")
    label = AGENT_LABELS.get(agent_id, agent_id)
    st.markdown(
        f'<div style="border-left:5px solid {color};padding:12px 16px;'
        f'margin:4px 0;background:{bg};border-radius:4px">'
        f"<b style='color:{color}'>{icon} {label}</b>"
        f"{body_html}</div>",
        unsafe_allow_html=True,
    )


def render_intent_card(agent_id: str, intent: dict) -> None:
    draw = intent.get("intent_draw_kwh", 0)
    offer = intent.get("intent_offer_kwh", 0)
    message = intent.get("message", "")
    body = (
        f"<br>Niyet Çekiş: <b>{draw:.1f} kWh</b> &nbsp;"
        f"Niyet Teklif: <b>{offer:.1f} kWh</b>"
        f"<hr style='border-color:rgba(128,128,128,0.3);margin:8px 0'>"
        f"<small><i>{message}</i></small>"
    )
    _card(agent_id, body)


def render_action_card(
    agent_id: str,
    action: dict,
    intent: dict | None = None,
    tolerance: float = 0.5,
) -> None:
    draw = action.get("draw_kwh", 0)
    offer = action.get("offer_kwh", 0)
    store = action.get("store_kwh", 0)
    reasoning = action.get("reasoning", "")

    promise_html = ""
    if intent is not None:
        kept = is_promise_kept(intent, action, tolerance)
        icon, text, pcolor = (
            ("✓", "Söz tutuldu", "green") if kept else ("✗", "Söz tutulmadı", "#cc3333")
        )
        promise_html = (
            f'&nbsp;&nbsp;<span style="color:{pcolor};font-weight:bold">'
            f"{icon} {text}</span>"
        )

    body = (
        f"<br>Çekiş: <b>{draw:.1f} kWh</b> &nbsp;"
        f"Teklif: <b>{offer:.1f} kWh</b> &nbsp;"
        f"Depo: <b>{store:.1f} kWh</b>"
        f"{promise_html}"
        f"<hr style='border-color:rgba(128,128,128,0.3);margin:8px 0'>"
        f"<small><i>{reasoning}</i></small>"
    )
    _card(agent_id, body)


def render_round_result(result: dict) -> None:
    total_draw = result.get("total_draw_kwh", 0.0)
    capacity = result.get("capacity_kwh", 5.5)
    overflow = result.get("overflow_kwh", 0.0)
    usage_pct = min(100.0, total_draw / capacity * 100) if capacity > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Çekiş", f"{total_draw:.2f} kWh")
    c2.metric("Kapasite", f"{capacity:.1f} kWh")
    c3.metric("Aşım", f"{overflow:.2f} kWh")
    c4.metric("Kullanım Oranı", f"%{usage_pct:.0f}")

    if overflow > 0:
        st.error(f"⚠️ Kapasite aşıldı — fazla çekiş: {overflow:.2f} kWh")
    else:
        st.success("✓ Kapasite sınırı korundu")

    payoffs = result.get("payoffs", [])
    if payoffs:
        rows = [
            {
                "Ajan": p.get("agent_id", ""),
                "Çekiş (kWh)": round(p.get("draw_kwh", 0), 2),
                "Teklif (kWh)": round(p.get("offer_kwh", 0), 2),
                "Net Kazanç": round(p.get("net_payoff", 0), 2),
                "Ceza": round(p.get("penalty", 0), 2),
            }
            for p in payoffs
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


CATEGORY_COLORS: dict[str, str] = {
    "coordination": "#4361EE",
    "agreement": "#06A77D",
    "offer_proposal": "#06b6d4",
    "offer_request": "#22d3ee",
    "warning": "#F77F00",
    "rejection": "#E63946",
    "other": "#6B7280",
}

_TR_MONTHS = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
    5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
    9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}


def _promise_symbols(states: list[str]) -> str:
    parts = []
    for s in states:
        if s == "kept":
            parts.append('<span style="color:#06A77D;font-size:1em">✓</span>')
        elif s == "broken":
            parts.append('<span style="color:#E63946;font-size:1em">✗</span>')
        elif s == "future":
            parts.append('<span style="color:#bbb;font-size:0.85em">⏳</span>')
        else:
            parts.append('<span style="color:#ccc">·</span>')
    return "".join(parts)


def _render_agent_status_card(
    aid: str,
    current_round: int,
    bundle: "RoundBundle | None",
    csv_vals: dict,
    cum: dict,
    mode: str,
    tolerance: float,
) -> None:
    from streamlit_components.loaders import RoundBundle  # local to avoid circular at module level

    color = AGENT_COLORS[aid]
    icon = AGENT_ICONS[aid]
    bg = _bg_color(color, 0.06)

    consumption = csv_vals.get("consumption", 0.0)
    pv = csv_vals.get("pv", 0.0)
    net_gap = max(0.0, consumption - pv)

    action = (bundle.actions.get(aid, {}) if bundle else {}) or {}
    intent = (bundle.intents.get(aid, {}) if bundle else {}) or {}

    draw_kwh = action.get("draw_kwh", 0.0)
    capacity_ref = 5.5
    draw_pct = min(draw_kwh / capacity_ref * 100, 100.0)

    # Niyet satırı (V2+)
    intent_html = ""
    if mode != "v1" and intent:
        intent_draw = intent.get("intent_draw_kwh", 0.0)
        if action:
            kept = is_promise_kept(intent, action, tolerance)
            badge_color = "#06A77D" if kept else "#E63946"
            badge_text = "✓ tutuldu" if kept else "✗ tutulmadı"
            badge = (
                f'<span style="background:{badge_color};color:white;'
                f'border-radius:3px;padding:1px 5px;font-size:0.75em;margin-left:4px">'
                f'{badge_text}</span>'
            )
        else:
            badge = ""
        intent_html = (
            f"<div style='font-size:0.83em;margin:2px 0'>"
            f"<span style='color:#888'>💭 Niyet:</span> <b>{intent_draw:.2f} kWh</b>{badge}</div>"
        )

    # Söz tutma sembolü satırı (V2+)
    promise_html = ""
    states = cum.get("promise_states", [])
    if mode != "v1" and any(s != "no_intent" for s in states):
        promise_html = (
            f"<div style='font-size:0.83em;margin:2px 0'>"
            f"<span style='color:#888'>Söz:</span> {_promise_symbols(states)}</div>"
        )

    # Müzakere sayısı (V3/V4)
    neg_html = ""
    neg_count = cum.get("neg_message_count", 0)
    if mode in ("v3", "v4") and neg_count > 0:
        neg_html = (
            f"<div style='font-size:0.83em;margin:2px 0'>"
            f"<span style='color:#888'>Müzakere:</span> <b>{neg_count} mesaj</b></div>"
        )

    st.markdown(
        f'<div style="border-left:5px solid {color};padding:14px 16px;'
        f'background:{bg};border-radius:4px;margin-bottom:4px">'
        f'<div style="font-weight:700;color:{color};margin-bottom:10px">{icon} {AGENT_LABELS[aid]}</div>'
        f'<div style="font-size:0.75em;font-weight:700;color:#555;letter-spacing:0.5px;margin-bottom:6px">⚡ BU TURUN DURUMU</div>'
        f"<div style='font-size:0.83em;margin:2px 0'><span style='color:#888'>Tüketim:</span> <b>{consumption:.2f} kWh</b></div>"
        f"<div style='font-size:0.83em;margin:2px 0'><span style='color:#888'>PV üretimi:</span> <b>{pv:.2f} kWh</b></div>"
        f"<div style='font-size:0.83em;margin:2px 0'><span style='color:#888'>Net açık:</span> <b>{net_gap:.2f} kWh</b></div>"
        f"{intent_html}"
        f"<div style='font-size:0.83em;margin:2px 0'><span style='color:#888'>⚡ Karar (çekiş):</span> <b>{draw_kwh:.2f} kWh</b></div>"
        f"<div style='font-size:0.83em;margin:2px 0'><span style='color:#888'>Kapasite payı:</span> <b>%{draw_pct:.0f}</b></div>"
        f'<hr style="border:none;border-top:1px solid rgba(128,128,128,0.2);margin:10px 0">'
        f'<div style="font-size:0.75em;font-weight:700;color:#555;letter-spacing:0.5px;margin-bottom:6px">📊 KÜMÜLATİF (Tur 1→{current_round})</div>'
        f"<div style='font-size:0.83em;margin:2px 0'><span style='color:#888'>Toplam çekiş:</span> <b>{cum.get('total_draw', 0):.2f} kWh</b></div>"
        f"<div style='font-size:0.83em;margin:2px 0'><span style='color:#888'>Toplam ceza:</span> <b>{cum.get('total_penalty', 0):.2f}</b></div>"
        f"<div style='font-size:0.83em;margin:2px 0'><span style='color:#888'>Net puan:</span> <b>{cum.get('total_net_payoff', 0):.2f}</b></div>"
        f"{promise_html}"
        f"{neg_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_scene(bundle: "RoundBundle | None", mode: str, tolerance: float) -> None:
    if bundle is None:
        st.info("Bu tur için log verisi bulunamadı.")
        return

    # V2+ → Niyet aşaması
    if mode in ("v2", "v3", "v4"):
        st.markdown("##### 💭 Niyet Aşaması")
        if bundle.intents:
            cols = st.columns(3)
            for col, aid in zip(cols, ["H1", "H6", "H8"]):
                with col:
                    intent = bundle.intents.get(aid)
                    if intent:
                        render_intent_card(aid, intent)
                    else:
                        st.caption(f"{aid}: niyet verisi yok")
        else:
            st.caption("Bu turda niyet verisi yok.")

    # V3/V4 → Müzakere
    if mode in ("v3", "v4"):
        st.markdown("##### 💬 Müzakere")
        if bundle.neg_rounds:
            total_neg = max(bundle.neg_rounds.keys(), default=1)
            for neg_r in sorted(bundle.neg_rounds.keys()):
                msgs = bundle.neg_rounds[neg_r]
                st.markdown(
                    f'<div style="font-size:0.8em;color:#888;margin:10px 0 4px 0">'
                    f'─── Müzakere Turu {neg_r}/{total_neg} ───</div>',
                    unsafe_allow_html=True,
                )
                for msg in msgs:
                    from_agent = msg.get("from_agent", "?")
                    to_agent = msg.get("to_agent", "?")
                    category = msg.get("category", "other")
                    message = msg.get("message", "")
                    from_color = AGENT_COLORS.get(from_agent, "#888")
                    to_color = AGENT_COLORS.get(to_agent, "#888")
                    cat_color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"])
                    st.markdown(
                        f'<div style="border-left:4px solid {from_color};padding:8px 12px;'
                        f'margin:4px 0;background:rgba(0,0,0,0.02);border-radius:0 4px 4px 0">'
                        f'<b style="color:{from_color}">{from_agent}</b>'
                        f' → <b style="color:{to_color}">{to_agent}</b>'
                        f' <span style="background:{cat_color};color:white;border-radius:3px;'
                        f'padding:1px 6px;font-size:0.73em;margin-left:6px">{category}</span>'
                        f'<div style="margin-top:5px;font-size:0.88em;color:#333">{message}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("Bu turda müzakere mesajı yok.")

    # Kararlar (tüm modlar)
    st.markdown("##### ⚡ Kararlar")
    cols = st.columns(3)
    for col, aid in zip(cols, ["H1", "H6", "H8"]):
        with col:
            action = bundle.actions.get(aid)
            intent = bundle.intents.get(aid) if mode != "v1" else None
            if action:
                render_action_card(aid, action, intent=intent, tolerance=tolerance)
            else:
                st.info(f"{aid}: karar verisi yok")


def _render_round_result_bar(bundle: "RoundBundle | None", run: RunData) -> None:
    if bundle is None or bundle.result is None:
        return
    result = bundle.result
    total_draw = result.get("total_draw_kwh", 0.0)
    capacity = (
        result.get("capacity_kwh")
        or run.config.get("scenario", {}).get("grid_capacity_kwh", 5.5)
    )
    overflow = result.get("overflow_kwh", 0.0)
    usage_pct = min(total_draw / capacity * 100, 100.0) if capacity > 0 else 0.0
    bar_color = "#E63946" if overflow > 0 else "#06A77D"

    status_html = (
        f'<span style="color:#E63946">⚠️ Kapasite ihlali: +{overflow:.2f} kWh</span>'
        if overflow > 0
        else '<span style="color:#06A77D">✓ Kapasite sınırı korundu</span>'
    )
    st.markdown(
        f'<div style="margin:4px 0">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.85em;margin-bottom:4px">'
        f'<span>Toplam çekiş: <b>{total_draw:.2f} kWh</b></span>'
        f'<span>Kapasite: <b>{capacity:.1f} kWh</b></span></div>'
        f'<div style="background:#e9ecef;border-radius:4px;height:16px;overflow:hidden">'
        f'<div style="background:{bar_color};width:{usage_pct:.1f}%;height:100%;border-radius:4px"></div></div>'
        f'<div style="margin-top:6px;font-size:0.88em">{status_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

_MERMAID_DIAGRAM = """
graph TD
    subgraph L1["Katman 1 — Fiziksel Dünya"]
        H1["H1 ev + panel + batarya"]
        H6["H6 ev + panel + batarya"]
        H8["H8 ev + panel + batarya"]
    end

    subgraph L2["Katman 2 — Veri Toplama"]
        D1["Tüketim sayacı"]
        D2["PV ölçümü"]
        D3["Batarya durumu"]
    end

    subgraph L3["Katman 3 — Agentic AI İşleme"]
        A1["H1 LLM ajanı"]
        A6["H6 LLM ajanı"]
        A8["H8 LLM ajanı"]
        FLOW["Niyet → Müzakere → Karar"]
    end

    subgraph L4["Katman 4 — Karar & Geri Bildirim"]
        R1["Hakem skoru"]
        R2["Kapasite kontrolü"]
        R3["Ev sahibi bildirimi"]
    end

    H1 --> D1
    H6 --> D2
    H8 --> D3
    D1 --> A1
    D2 --> A6
    D3 --> A8
    A1 --> FLOW
    A6 --> FLOW
    A8 --> FLOW
    FLOW --> R1
    R1 --> R2
    R2 --> R3
"""

_VERSION_CARDS = [
    {
        "label": "V1",
        "name": "Sessiz Sistem",
        "color": "#888780",
        "metrics": [
            ("İletişim", "yok"),
            ("Persona", "heterojen"),
            ("Müzakere", "—"),
        ],
        "quote": "Ajanlar konuşmaz, sadece geçmiş eylemleri gözlemler.",
    },
    {
        "label": "V2",
        "name": "Niyet Paylaşımı",
        "color": "#4361EE",
        "metrics": [
            ("İletişim", "tek yönlü açıklama"),
            ("Persona", "heterojen"),
            ("Müzakere", "—"),
        ],
        "quote": "Ajanlar planlarını paylaşır ama bağlayıcı değildir (cheap talk).",
    },
    {
        "label": "V3",
        "name": "Müzakere Masası",
        "color": "#06A77D",
        "metrics": [
            ("İletişim", "ikili mesajlaşma"),
            ("Persona", "heterojen"),
            ("Müzakere", "3 tur"),
        ],
        "quote": "Ajanlar hedefli mesajlaşır, pazarlık eder.",
    },
    {
        "label": "V4",
        "name": "Stres Testi",
        "color": "#F77F00",
        "metrics": [
            ("İletişim", "ikili mesajlaşma"),
            ("Persona", "homojen egoist"),
            ("Müzakere", "7 tur"),
        ],
        "quote": "Sistem sınırlarını test eder: tüm ajanlar egoist, uzun müzakere.",
    },
]


def render_method_tab() -> None:
    # --- Section 1: System description ---
    st.title("Yöntem")
    st.markdown(
        "Her hane bağımsız bir LLM ajanıdır (Gemini 2.5 Pro, Vertex AI). Her turda ajanlar "
        "günlük tüketim ve PV üretim verilerini alır, paylaşılan şebeke kapasitesi (5.5 kWh) "
        "altında çekiş/paylaşım/depolama kararı verir. Hakem deterministik formülle skorlar; "
        "kapasite aşımı durumunda ceza orantılı dağıtılır. Sistem dört farklı iletişim "
        "seviyesinde test edilmiştir."
    )

    # --- Section 2: Architecture table (pure HTML, no CDN) ---
    st.markdown("#### IoT Veri Akışı — 4 Katmanlı Mimari")

    def _cell(text: str, color: str) -> str:
        return (
            f'<td style="padding:10px 14px;text-align:center;vertical-align:middle">'
            f'<div style="background:white;border:1.5px solid {color};border-radius:6px;'
            f'padding:7px 10px;font-size:0.85em;line-height:1.4">{text}</div></td>'
        )

    def _arrow_row() -> str:
        return (
            '<tr><td></td>'
            + ''.join(
                '<td style="text-align:center;font-size:1.3em;color:#bbb;padding:2px">↓</td>'
                for _ in range(3)
            )
            + '</tr>'
        )

    def _layer_header(color: str, label: str) -> str:
        return (
            f'<td style="padding:8px 12px;white-space:nowrap;vertical-align:middle">'
            f'<div style="border-left:4px solid {color};padding-left:10px;'
            f'color:{color};font-weight:700;font-size:0.82em;line-height:1.5">{label}</div></td>'
        )

    c1, c2, c3, c4 = "#4A90D9", "#7B68EE", "#06A77D", "#E63946"

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-family:sans-serif">
      <thead>
        <tr style="background:#f0f0f0">
          <th style="padding:8px 12px;text-align:left;font-size:0.8em;color:#666;width:22%">Katman</th>
          <th style="padding:8px 12px;text-align:center;font-size:0.8em;color:{AGENT_COLORS['H1']}">🔌 H1 — Geniş Aile</th>
          <th style="padding:8px 12px;text-align:center;font-size:0.8em;color:{AGENT_COLORS['H6']}">🌿 H6 — Küçük Hane</th>
          <th style="padding:8px 12px;text-align:center;font-size:0.8em;color:{AGENT_COLORS['H8']}">⚡ H8 — Büyük Hane</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background:{_bg_color(c1, 0.06)}">
          {_layer_header(c1, "Katman 1<br>Fiziksel Dünya")}
          {_cell("Ev + güneş paneli<br>+ batarya", c1)}
          {_cell("Ev + güneş paneli<br>+ batarya", c1)}
          {_cell("Ev + güneş paneli<br>+ batarya", c1)}
        </tr>
        {_arrow_row()}
        <tr style="background:{_bg_color(c2, 0.06)}">
          {_layer_header(c2, "Katman 2<br>Veri Toplama")}
          {_cell("Tüketim sayacı<br>(CSV verisi)", c2)}
          {_cell("PV ölçümü<br>(solar_w_m²)", c2)}
          {_cell("Batarya durumu<br>(SoC %)", c2)}
        </tr>
        {_arrow_row()}
        <tr style="background:{_bg_color(c3, 0.06)}">
          {_layer_header(c3, "Katman 3<br>Agentic AI")}
          {_cell("H1 LLM ajanı<br><i>Niyet → Karar</i>", c3)}
          {_cell("H6 LLM ajanı<br><i>Niyet → Karar</i>", c3)}
          {_cell("H8 LLM ajanı<br><i>Niyet → Karar</i>", c3)}
        </tr>
        {_arrow_row()}
        <tr style="background:{_bg_color(c4, 0.06)}">
          {_layer_header(c4, "Katman 4<br>Geri Bildirim")}
          {_cell("Hakem skoru<br>+ net kazanç", c4)}
          {_cell("Kapasite<br>kontrolü", c4)}
          {_cell("Ev sahibi<br>bildirimi", c4)}
        </tr>
      </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown(
        "_Bu çalışmadaki simülasyon Katman 2 ve 3'ün bir prototipidir; "
        "sayaçlar CSV verisi, ajanlar Vertex AI ile temsil edilmiştir._"
    )

    st.markdown("---")

    # --- Section 3: Version cards ---
    st.markdown("#### Sistem Versiyonları")
    cols = st.columns(4)
    for col, card in zip(cols, _VERSION_CARDS):
        color = card["color"]
        bg = _bg_color(color, 0.07)
        metrics_html = "".join(
            f"<div style='font-size:0.85em;margin:3px 0'>"
            f"<span style='color:#666'>{k}:</span> <b>{v}</b></div>"
            for k, v in card["metrics"]
        )
        with col:
            st.markdown(
                f'<div style="border-left:5px solid {color};padding:14px 16px;'
                f'background:{bg};border-radius:4px;height:100%">'
                f"<div style='font-size:0.75em;color:{color};font-weight:700;letter-spacing:1px'>"
                f"{card['label']}</div>"
                f"<div style='font-size:1.05em;font-weight:700;margin:4px 0 10px 0'>"
                f"{card['name']}</div>"
                f"<hr style='border:none;border-top:1px solid rgba(128,128,128,0.2);margin:8px 0'>"
                f"{metrics_html}"
                f"<hr style='border:none;border-top:1px solid rgba(128,128,128,0.2);margin:10px 0 8px 0'>"
                f"<div style='font-size:0.85em;color:#444;font-style:italic'>"
                f"&ldquo;{card['quote']}&rdquo;</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "community_window.csv"


@st.cache_data
def _load_community_window() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def render_dataset_tab() -> None:
    st.title("Veri Seti — Puebla, Meksika")
    st.markdown(
        "Mendeley Data üzerinden erişilebilen, Puebla (Meksika) bölgesinden gerçek ev tipi "
        "tüketim ve PV üretim verisi. 2022–2023 yılları. "
        "DOI: [10.17632/vsjtbzjttb.4](https://doi.org/10.17632/vsjtbzjttb.4)"
    )

    st.subheader("Haneler")
    agent_details = [
        (
            "H1",
            "4–5 kişilik geniş bir aile. Ortalama günlük tüketim ~3.9 kWh; "
            "iklim kontrolü (klima, ısıtıcı) ağır basıyor. "
            "5 m² panel ile kısmi PV desteği, 10 kWh batarya.",
        ),
        (
            "H6",
            "1–2 kişilik küçük ve verimli hane. Günlük tüketim ~1.6 kWh ile "
            "topluluğun en düşük tüketicisi. "
            "3 m² panel, 6 kWh batarya; dengeli ve öngörülü kullanım.",
        ),
        (
            "H8",
            "Kalabalık hane; günlük ~4.3 kWh ile en yüksek tüketici. "
            "8 m² panel en büyük PV kapasitesini sağlar, fakat yüksek tüketim baskısı sürer. "
            "15 kWh batarya ile stratejik depolama.",
        ),
    ]
    cols = st.columns(3)
    for col, (aid, persona_text) in zip(cols, agent_details):
        color = AGENT_COLORS[aid]
        bg = _bg_color(color, 0.07)
        icon = AGENT_ICONS[aid]
        with col:
            st.markdown(
                f'<div style="border-left:5px solid {color};padding:14px 16px;'
                f'background:{bg};border-radius:4px;margin-bottom:8px">'
                f"<b style='color:{color}'>{icon} {AGENT_LABELS[aid]}</b>"
                f"<p style='margin:8px 0 0 0;font-size:0.92em'>{persona_text}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.info(
        "**Kış Senaryosu — Aralık 2022**\n\n"
        "- **Tarih:** 1–10 Aralık 2022\n"
        "- **Sıcaklık:** gece ~6 °C, gündüz ~21 °C\n"
        "- **Solar radyasyon:** ~124 W/m²\n"
        "- **Karakter:** Düşük PV + yüksek tüketim → kıtlık rejimi"
    )

    # Charts
    try:
        df = _load_community_window()
        winter = df[
            (df["date"] >= "2022-12-01") & (df["date"] <= "2022-12-10")
        ].copy()

        date_labels = winter["date"].dt.strftime("%d Ara")

        c_left, c_right = st.columns(2)

        with c_left:
            fig_cons = go.Figure()
            for aid, col_name in [("H1", "h1_kwh"), ("H6", "h6_kwh"), ("H8", "h8_kwh")]:
                fig_cons.add_trace(
                    go.Scatter(
                        x=date_labels,
                        y=winter[col_name],
                        name=AGENT_LABELS[aid],
                        mode="lines+markers",
                        line=dict(color=AGENT_COLORS[aid], width=2),
                        marker=dict(size=6),
                    )
                )
            fig_cons.update_layout(
                title="Günlük Tüketim (kWh)",
                yaxis_title="kWh",
                height=360,
                margin=dict(t=40, b=20),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_cons, use_container_width=True)

        with c_right:
            fig_pv = go.Figure()
            for aid, col_name in [("H1", "h1_pv_kwh"), ("H6", "h6_pv_kwh"), ("H8", "h8_pv_kwh")]:
                fig_pv.add_trace(
                    go.Scatter(
                        x=date_labels,
                        y=winter[col_name],
                        name=AGENT_LABELS[aid],
                        mode="lines+markers",
                        line=dict(color=AGENT_COLORS[aid], width=2, dash="dot"),
                        marker=dict(size=6),
                    )
                )
            fig_pv.update_layout(
                title="Günlük PV Üretimi (kWh)",
                yaxis_title="kWh",
                height=360,
                margin=dict(t=40, b=20),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_pv, use_container_width=True)

    except FileNotFoundError:
        st.warning(
            f"`{_DATA_PATH}` bulunamadı. "
            "`uv run python scripts/prepare_data.py` komutunu çalıştırın."
        )

    st.markdown(
        "_Bu kıtlık manzarası, ajanların paylaşılan kapasiteyi nasıl yönettiğinin testidir. "
        "Aynı problem agentic sistem olmadan çözülmek istense, her hanenin tüketim/üretim açığını "
        "önceden hesaplaması ve manuel paylaşım planı kurması gerekirdi._"
    )


def render_overview_tab() -> None:
    st.title("Agentic Energy CPR")
    st.markdown(
        """
        **Agentic Energy CPR** (Common-Pool Resource), 3 LLM ajanının paylaşılan bir
        şebeke kapasitesi (5.5 kWh/gün) üzerinde 10 tur boyunca karar aldığı bir
        çok-etmenli simülasyondur. Her hane bir **Gemini 2.5 Pro** ajanıyla temsil edilir.
        Kıtlık koşullarında ajanlar arasındaki çelişen çıkarlar Ostrom (1990)
        Common-Pool Resource teorisiyle analiz edilmektedir. Üç versiyon kademeli
        iletişim katmanları ekler: sessiz sistemden tam müzakereye.
        """
    )

    st.subheader("Haneler")
    cols = st.columns(3)
    for col, agent_id in zip(cols, ["H1", "H6", "H8"]):
        color = AGENT_COLORS[agent_id]
        icon = AGENT_ICONS[agent_id]
        with col:
            st.markdown(
                f'<div style="border-top:6px solid {color};padding:16px;'
                f'border-radius:4px;background:{_bg_color(color, 0.06)}">'
                f"<h4 style='color:{color}'>{icon} {AGENT_LABELS[agent_id]}</h4>"
                f"<p>{AGENT_PERSONAS[agent_id]}</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Sürümler")
    v_cols = st.columns(3)
    versions = [
        (
            "V1 — Sessiz Sistem",
            "Ajanlar direkt karar verir. Niyet beyanı veya iletişim yoktur. "
            "Sadece kendi tüketim ve PV verilerini bilir.",
        ),
        (
            "V2 — Niyet Bildirimli",
            "Her tur başında ajanlar niyetlerini ve kısa açıklamalarını diğerleriyle "
            "paylaşır. Pazarlık yoktur; ancak sosyal baskı ve gözlem oluşabilir.",
        ),
        (
            "V3 — Müzakere Katmanlı",
            "Niyet beyanından sonra 3 tur yapılandırılmış müzakere gerçekleşir. "
            "Ajanlar koordinasyon, teklif, uyarı ve onay mesajları gönderir.",
        ),
    ]
    for col, (title, desc) in zip(v_cols, versions):
        with col:
            st.info(f"**{title}**\n\n{desc}")


def render_live_simulation_tab() -> None:
    _MODE_OPTIONS = [
        "V1 — Sessiz Sistem",
        "V2 — Niyet Paylaşımı",
        "V3 — Müzakere Masası",
        "V4 — Stres Testi",
    ]
    _MODE_KEY = {
        "V1 — Sessiz Sistem": "v1",
        "V2 — Niyet Paylaşımı": "v2",
        "V3 — Müzakere Masası": "v3",
        "V4 — Stres Testi": "v4",
    }

    mode_label = st.radio(
        "İletişim Modu",
        _MODE_OPTIONS,
        horizontal=True,
        key="live_mode",
    )
    mode_key = _MODE_KEY[mode_label]
    run = load_run_by_id(COMPARISON_RUN_IDS[mode_key], _COMP_RUNS_DIR)

    if run is None:
        st.info(f"'{mode_label}' için henüz koşu verisi yok — senaryoyu çalıştırın.")
        return

    # Round selector
    if "current_round" not in st.session_state:
        st.session_state.current_round = 1

    n_rounds = max(run.rounds.keys(), default=10)
    btn_cols = st.columns(n_rounds)
    for i, col in enumerate(btn_cols):
        rn = i + 1
        btn_type = "primary" if st.session_state.current_round == rn else "secondary"
        if col.button(f"Tur {rn}", key=f"live_rnd_{rn}", type=btn_type, width="stretch"):
            st.session_state.current_round = rn
            st.rerun()

    current_round = st.session_state.current_round

    # Date header — avoid datetime.fromisoformat entirely
    _raw = run.config.get("scenario", {}).get("start_date", "2022-12-01")
    # hasattr check: date/datetime objects have .isoformat(); plain strings don't
    _raw_str: str = _raw.isoformat()[:10] if hasattr(_raw, "isoformat") else str(_raw)[:10]
    round_ts = pd.Timestamp(_raw_str) + pd.Timedelta(days=current_round - 1)
    tr_date = f"{round_ts.day} {_TR_MONTHS[round_ts.month]} {round_ts.year}"
    st.markdown(f"### 📅 {tr_date} — Tur {current_round}/{n_rounds}")

    # CSV row for this round
    df = _load_community_window()
    csv_row = df[df["date"] == round_ts]
    _col_map = {
        "H1": ("h1_kwh", "h1_pv_kwh"),
        "H6": ("h6_kwh", "h6_pv_kwh"),
        "H8": ("h8_kwh", "h8_pv_kwh"),
    }
    csv_data: dict[str, dict] = {}
    for aid, (c_col, pv_col) in _col_map.items():
        if not csv_row.empty and c_col in csv_row.columns:
            csv_data[aid] = {
                "consumption": float(csv_row[c_col].iloc[0]),
                "pv": float(csv_row[pv_col].iloc[0]),
            }
        else:
            csv_data[aid] = {"consumption": 0.0, "pv": 0.0}

    # Tolerance (priority: metrics.json → config → 0.5)
    tolerance: float = run.metrics.get("tolerance_kwh") or run.tolerance_kwh or 0.5

    # Cumulative stats
    cumulative = get_cumulative(run, current_round)

    # --- Agent cards ---
    agent_cols = st.columns(3)
    for col, aid in zip(agent_cols, ["H1", "H6", "H8"]):
        with col:
            _render_agent_status_card(
                aid, current_round,
                run.rounds.get(current_round),
                csv_data[aid],
                cumulative[aid],
                mode_key,
                tolerance,
            )

    st.markdown("---")

    # --- Scene area ---
    _render_scene(run.rounds.get(current_round), mode_key, tolerance)

    st.markdown("---")

    # --- Round result bar ---
    st.markdown("##### Tur Sonucu")
    _render_round_result_bar(run.rounds.get(current_round), run)


# ---------------------------------------------------------------------------
# Comparison tab helpers
# ---------------------------------------------------------------------------

def _violation_summary(run: RunData) -> dict:
    count = 0
    total_ov = 0.0
    per_round = [0.0] * 10
    for rn, bundle in run.rounds.items():
        if bundle.result:
            ov = float(bundle.result.get("overflow_kwh", 0.0))
            if ov > 0:
                count += 1
                total_ov += ov
            if 1 <= rn <= 10:
                per_round[rn - 1] = ov
    return {"count": count, "total_overflow": total_ov, "per_round": per_round}


def _promise_rates(run: RunData, tol: float) -> dict[str, float]:
    rates: dict[str, float] = {}
    for aid in AGENT_IDS:
        kept = total = 0
        for bundle in run.rounds.values():
            intent = bundle.intents.get(aid)
            action = bundle.actions.get(aid)
            if intent and action:
                total += 1
                if is_promise_kept(intent, action, tol):
                    kept += 1
        rates[aid] = kept / total if total > 0 else 0.0
    return rates


def _neg_category_counts(run: RunData) -> Counter:
    counts: Counter = Counter()
    for e in run.events:
        if e.get("event") == "negotiation_message":
            counts[e.get("category", "other")] += 1
    return counts


def _comp_leader(run: RunData) -> str:
    nets: dict[str, float] = {aid: 0.0 for aid in AGENT_IDS}
    for bundle in run.rounds.values():
        if bundle.result:
            for p in bundle.result.get("payoffs", []):
                aid = p.get("agent_id", "")
                if aid in nets:
                    nets[aid] += p.get("net_payoff", 0.0)
    return max(nets, key=lambda k: nets[k]) if nets else "—"


def render_comparison_tab() -> None:
    # Load the 4 fixed runs
    comp_runs: dict[str, RunData | None] = {}
    for vkey, run_id in COMPARISON_RUN_IDS.items():
        r = load_run_by_id(run_id, _COMP_RUNS_DIR)
        if r is None:
            st.warning(f"{vkey.upper()} run bulunamadı: {run_id}")
        comp_runs[vkey] = r

    v1r = comp_runs["v1"]
    v2r = comp_runs["v2"]
    v3r = comp_runs["v3"]
    v4r = comp_runs["v4"]

    def _tol(r: RunData | None) -> float:
        if r is None:
            return 0.5
        return r.metrics.get("tolerance_kwh") or r.tolerance_kwh or 0.5

    # --- Section 1: Title + Ana Bulgu ---
    st.title("Karşılaştırma — Dört Sistem Yan Yana")
    st.info(
        "**Ana Bulgu:** İletişim katmanları kademeli artırıldığında kapasite ihlalleri "
        "V1'deki 4'ten V3'teki 1'e düştü. Ancak V4'te tüm ajanlar egoist persona ile "
        "çalıştığında, 7 turluk müzakereye rağmen ihlal 5'e yükseldi — V1'in bile üzerinde. "
        "**İletişim altyapısı tek başına yeterli değil, ajanların niyeti de kritik.**"
    )

    # Precompute summaries
    empty_viol = {"count": 0, "total_overflow": 0.0, "per_round": [0.0] * 10}
    viol = {k: (_violation_summary(r) if r else empty_viol) for k, r in comp_runs.items()}

    # --- Section 2: 4 version cards ---
    _VCARD = {
        "v1": {"label": "V1", "name": "Sessiz Sistem",    "color": "#888780", "badge": ""},
        "v2": {"label": "V2", "name": "Niyet Paylaşımı",  "color": "#4361EE", "badge": ""},
        "v3": {"label": "V3", "name": "Müzakere Masası",  "color": "#06A77D", "badge": " ✅"},
        "v4": {"label": "V4", "name": "Stres Testi",      "color": "#F77F00", "badge": " ⚠"},
    }

    vcols = st.columns(4)
    for col, vkey in zip(vcols, ["v1", "v2", "v3", "v4"]):
        meta = _VCARD[vkey]
        color = meta["color"]
        bg = _bg_color(color, 0.07)
        r = comp_runs[vkey]
        vc = viol[vkey]
        with col:
            if r is None:
                st.warning(f"{meta['label']} yok")
                continue
            leader = _comp_leader(r)
            neg_count = sum(1 for e in r.events if e.get("event") == "negotiation_message")
            neg_str = str(neg_count) if neg_count > 0 else "—"
            if r.mode in ("v2", "v3", "v4"):
                pr = _promise_rates(r, _tol(r))
                h1_str = f"%{int(pr.get('H1', 0) * 100)}"
            else:
                h1_str = "—"
            st.markdown(
                f'<div style="border-left:5px solid {color};padding:14px 16px;'
                f'background:{bg};border-radius:4px">'
                f'<div style="font-size:0.75em;font-weight:700;color:{color};letter-spacing:1px">{meta["label"]}</div>'
                f'<div style="font-size:1.05em;font-weight:700;margin:4px 0 10px 0">{meta["name"]}{meta["badge"]}</div>'
                f'<hr style="border:none;border-top:1px solid rgba(128,128,128,0.2);margin:8px 0">'
                f'<div style="font-size:0.84em;margin:3px 0"><span style="color:#666">İhlal:</span> <b>{vc["count"]}</b></div>'
                f'<div style="font-size:0.84em;margin:3px 0"><span style="color:#666">Aşım:</span> <b>{vc["total_overflow"]:.2f} kWh</b></div>'
                f'<div style="font-size:0.84em;margin:3px 0"><span style="color:#666">Lider:</span> <b>{leader}</b></div>'
                f'<div style="font-size:0.84em;margin:3px 0"><span style="color:#666">Müzakere:</span> <b>{neg_str}</b></div>'
                f'<div style="font-size:0.84em;margin:3px 0"><span style="color:#666">H1 söz:</span> <b>{h1_str}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --- Section 3: Violation bar chart ---
    v_labels = ["V1 Sessiz Sistem", "V2 Niyet Paylaşımı", "V3 Müzakere Masası", "V4 Stres Testi"]
    v_counts = [viol[k]["count"] for k in ["v1", "v2", "v3", "v4"]]
    fig_bar = go.Figure(go.Bar(
        x=v_labels,
        y=v_counts,
        marker_color=["#888780", "#4361EE", "#06A77D", "#F77F00"],
        text=v_counts,
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Kapasite İhlal Sayısı (10 tur içinde)",
        height=350,
        showlegend=False,
        margin=dict(t=50, b=20),
        yaxis=dict(range=[0, max(v_counts, default=0) + 2]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Section 4: Per-round heatmap ---
    z_matrix, text_matrix = [], []
    for vkey in ["v1", "v2", "v3", "v4"]:
        row_z, row_t = [], []
        for ov in viol[vkey]["per_round"]:
            row_z.append(1 if ov > 0 else 0)
            row_t.append(f"+{ov:.2f}" if ov > 0 else "✓")
        z_matrix.append(row_z)
        text_matrix.append(row_t)

    fig_heat = go.Figure(go.Heatmap(
        z=z_matrix,
        x=[f"T{i + 1}" for i in range(10)],
        y=["V1", "V2", "V3", "V4"],
        colorscale=[[0, "#06A77D"], [1, "#E63946"]],
        text=text_matrix,
        texttemplate="%{text}",
        showscale=False,
        xgap=3,
        ygap=3,
    ))
    fig_heat.update_layout(
        title="Hangi turda ihlal? (kırmızı = ihlal, yeşil = OK)",
        height=300,
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Section 5: Promise keeping grouped bar ---
    fig_promise = go.Figure()
    for aid, color in [("H1", AGENT_COLORS["H1"]), ("H6", AGENT_COLORS["H6"]), ("H8", AGENT_COLORS["H8"])]:
        y_vals = []
        for r in [v2r, v3r, v4r]:
            if r is None:
                y_vals.append(0.0)
            else:
                pr = _promise_rates(r, _tol(r))
                y_vals.append(round(pr.get(aid, 0.0) * 100, 1))
        fig_promise.add_trace(go.Bar(
            name=aid,
            x=["V2", "V3", "V4"],
            y=y_vals,
            marker_color=color,
            text=[f"%{v:.0f}" for v in y_vals],
            textposition="outside",
        ))
    fig_promise.update_layout(
        barmode="group",
        title="Söz Tutma Oranı (%) — V2 / V3 / V4",
        yaxis=dict(range=[0, 115], title="%"),
        height=380,
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_promise, use_container_width=True)

    # --- Section 6: Negotiation category V3 vs V4 ---
    st.markdown("---")
    _cat_colors = {
        "coordination": "#4361EE", "agreement": "#06A77D",
        "offer_proposal": "#06b6d4", "offer_request": "#22d3ee",
        "warning": "#F77F00", "rejection": "#E63946", "other": "#6B7280",
    }
    col3, col4 = st.columns(2)
    for col, r, title in [(col3, v3r, "V3 Müzakere Masası"), (col4, v4r, "V4 Stres Testi")]:
        with col:
            st.subheader(title)
            if r is None:
                st.warning("Run yok")
                continue
            cat_counts = _neg_category_counts(r)
            if cat_counts:
                labels = list(cat_counts.keys())
                values = list(cat_counts.values())
                fig_pie = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=[_cat_colors.get(lb, "#6B7280") for lb in labels],
                    hole=0.35,
                    textinfo="label+percent",
                ))
                fig_pie.update_layout(
                    height=320,
                    margin=dict(t=20, b=20, l=20, r=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Müzakere verisi yok.")

    # --- Section 7: Bulgular ---
    st.markdown("---")
    st.markdown("""
### 🎯 Ana Bulgular

1. **İletişim katmanı ihlal düşürür:** V1'de 4 olan ihlal, V3'te 1'e indi (%75 azalma).
2. **V3 koordinasyon zirvesi:** Tek küçük ihlal (0.27 kWh), H1 sözünde mükemmel (%100).
3. **Egoist persona sistemi yener:** V4'te 7 müzakere turu olmasına rağmen 5 ihlal — V1'den de fazla.
4. **Çatışma sayısallaştı:** V3 → V4 geçişinde warning mesajı oranı dramatik biçimde arttı.
5. **H8 sürekli lider:** Stratejik persona 4/4 versiyonda en yüksek net puanı aldı.
""")
