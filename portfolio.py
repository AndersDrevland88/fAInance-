"""
CALMA HOLDING AS – Porteføljedashboard
Kjør med: streamlit run portfolio.py
Installer: pip install streamlit yfinance pandas plotly
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json
import os
import math

st.set_page_config(
    page_title="CALMA HOLDING – Portefølje",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DATA ─────────────────────────────────────────────────────────────────────
DATA_FILE = "portfolio_data.json"

DEFAULT_HOLDINGS = []

def load_data():
    if os.path.exists(DATA_FILE):
        data = json.load(open(DATA_FILE, "r", encoding="utf-8"))
        for h in data:
            h.setdefault("own_target", None)
            h.setdefault("dcf_params", {})
            h.setdefault("catalysts", [])
        return data
    return DEFAULT_HOLDINGS

def save_data(holdings):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(holdings, f, ensure_ascii=False, indent=2)

if "holdings" not in st.session_state:
    st.session_state.holdings = load_data()

# ─── FETCH ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_prices(tickers: tuple):
    prices = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            price = info.last_price
            prev  = info.previous_close
            prices[ticker] = {"price": price, "prev_close": prev,
                              "change_pct": ((price - prev) / prev * 100) if prev else 0}
        except Exception:
            prices[ticker] = None
    return prices

@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        return {
            "eps":           info.get("trailingEps"),
            "forward_eps":   info.get("forwardEps"),
            "book_value":    info.get("bookValue"),
            "pe":            info.get("trailingPE"),
            "forward_pe":    info.get("forwardPE"),
            "peg":           info.get("pegRatio"),
            "pb":            info.get("priceToBook"),
            "fcf":           info.get("freeCashflow"),
            "shares_out":    info.get("sharesOutstanding"),
            "market_cap":    info.get("marketCap"),
            "beta":          info.get("beta"),
            "debt_equity":   info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "roe":           info.get("returnOnEquity"),
            "gross_margin":  info.get("grossMargins"),
            "op_margin":     info.get("operatingMargins"),
            "net_margin":    info.get("profitMargins"),
            "div_yield":     info.get("dividendYield"),
            "target_mean":   info.get("targetMeanPrice"),
            "target_high":   info.get("targetHighPrice"),
            "target_low":    info.get("targetLowPrice"),
            "target_median": info.get("targetMedianPrice"),
            "num_analysts":  info.get("numberOfAnalystOpinions"),
            "rec_key":       info.get("recommendationKey"),
            "rec_mean":      info.get("recommendationMean"),
            "rev_growth":    info.get("revenueGrowth"),
            "earn_growth":   info.get("earningsGrowth"),
            "hi52":          info.get("fiftyTwoWeekHigh"),
            "lo52":          info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def fetch_calendar(ticker: str):
    try:
        return yf.Ticker(ticker).calendar
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_history(tickers: tuple, period: str):
    dfs = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period=period)["Close"].reset_index()
            hist["Ticker"] = t
            dfs.append(hist)
        except Exception:
            pass
    return pd.concat(dfs) if dfs else pd.DataFrame()

# ─── VALUATION ────────────────────────────────────────────────────────────────
def graham_number(eps, book_value):
    if eps and book_value and eps > 0 and book_value > 0:
        return math.sqrt(22.5 * eps * book_value)
    return None

def dcf_value(base_fcf, growth, discount, terminal_growth, years, shares_out):
    if not base_fcf or not shares_out or shares_out == 0:
        return None
    if discount <= terminal_growth:
        return None
    fcf = base_fcf
    total_pv = 0.0
    for t in range(1, int(years) + 1):
        fcf *= (1 + growth)
        total_pv += fcf / (1 + discount) ** t
    terminal_pv = (fcf * (1 + terminal_growth) / (discount - terminal_growth)) / (1 + discount) ** years
    return (total_pv + terminal_pv) / shares_out

def mos(intrinsic, price):
    if intrinsic and intrinsic > 0:
        return (intrinsic - price) / intrinsic * 100
    return None

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def get_price(h, prices):
    if h["type"] == "unlisted":
        return h["manual_price"] or h["avg_cost"]
    p = prices.get(h["ticker"])
    return p["price"] if p else h["avg_cost"]

def fmt_nok(v):
    if v is None: return "–"
    return f"{v:,.0f} kr".replace(",", " ")

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "–"
    return f"{'+'if v>=0 else''}{v:.1f}%"

def fmt_n(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "–"
    return f"{v:.{d}f}"

def rec_label(key):
    return {"strong_buy": "🟢 Sterkt kjøp", "buy": "🟢 Kjøp",
            "hold": "🟡 Hold", "underperform": "🔴 Underperform",
            "sell": "🔴 Selg"}.get(key, key or "–")

def color_val(v):
    if isinstance(v, (int, float)) and not math.isnan(v):
        return "color: #22c55e" if v >= 0 else "color: #ef4444"
    return ""

def compute_portfolio(holdings, prices):
    rows = []
    for h in holdings:
        price   = get_price(h, prices)
        cost    = h["avg_cost"] * h["shares"]
        mkt_val = price * h["shares"]
        gain    = mkt_val - cost
        p       = prices.get(h.get("ticker"))
        rows.append({**h, "price": price, "cost": cost, "mkt_val": mkt_val,
                     "gain": gain, "gain_pct": (gain/cost*100) if cost else 0,
                     "day_chg": p["change_pct"] if p else None})
    return pd.DataFrame(rows)

# ─── INIT ─────────────────────────────────────────────────────────────────────
listed_tickers = tuple(h["ticker"] for h in st.session_state.holdings if h["type"] == "listed" and h["ticker"])

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 CALMA HOLDING AS")
    st.markdown("---")
    if st.button("🔄 Oppdater kurser", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("### ＋ Legg til posisjon")
    with st.expander("Ny aksje"):
        nn   = st.text_input("Selskapsnavn")
        nt   = st.selectbox("Type", ["listed","unlisted"], format_func=lambda x: "Notert" if x=="listed" else "Unotert")
        ntk  = st.text_input("Ticker (f.eks. ARCHER.OL)") if nt == "listed" else ""
        ns   = st.selectbox("Sektor", ["Industri","Shipping","Energi","Finans","IT","EdTech","Telekom","Cleantech","Fintech","Annet"])
        nsh  = st.number_input("Antall aksjer", min_value=1, value=100)
        nav  = st.number_input("Snittkurs (kr)", min_value=0.01, value=10.0, format="%.2f")
        nmp  = st.number_input("Nåkurs (kr)", min_value=0.01, value=10.0, format="%.2f") if nt == "unlisted" else None
        if st.button("Legg til", use_container_width=True) and nn:
            st.session_state.holdings.append({
                "id": f"custom_{datetime.now().timestamp()}", "name": nn,
                "ticker": ntk or None, "type": nt, "sector": ns,
                "shares": nsh, "avg_cost": nav, "manual_price": nmp,
                "last_updated": str(date.today()) if nt == "unlisted" else None,
                "own_target": None, "dcf_params": {}, "catalysts": [],
            })
            save_data(st.session_state.holdings)
            st.success(f"✅ {nn} lagt til!")
            st.rerun()

    st.markdown("---")
    st.caption(f"Sist oppdatert: {datetime.now().strftime('%H:%M:%S')}")

# ─── PRICES + PORTFOLIO ───────────────────────────────────────────────────────
with st.spinner("Henter kurser fra Oslo Børs..."):
    prices = fetch_prices(listed_tickers)

df = compute_portfolio(st.session_state.holdings, prices)

total_val    = df["mkt_val"].sum()
total_cost   = df["cost"].sum()
total_gain   = total_val - total_cost
total_gain_p = (total_gain / total_cost * 100) if total_cost else 0
listed_val   = df[df["type"] == "listed"]["mkt_val"].sum()
unlisted_val = df[df["type"] == "unlisted"]["mkt_val"].sum()

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("# 📊 Porteføljedashboard")
st.markdown(f"*CALMA HOLDING AS — {datetime.now().strftime('%d.%m.%Y %H:%M')}*")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("💼 Total verdi",        fmt_nok(total_val),    f"Kostpris {fmt_nok(total_cost)}")
c2.metric("📈 Urealisert gevinst", fmt_nok(total_gain),   fmt_pct(total_gain_p), delta_color="normal")
c3.metric("🔵 Noterte aksjer",     fmt_nok(listed_val),   f"{listed_val/total_val*100:.1f}% av total" if total_val else "–")
c4.metric("🔴 Unoterte aksjer",    fmt_nok(unlisted_val), f"{unlisted_val/total_val*100:.1f}% av total" if total_val else "–")
st.markdown("---")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_pos, tab_val, tab_ana, tab_trig, tab_risk, tab_sek, tab_hist, tab_edit = st.tabs([
    "📋 Posisjoner", "💎 Verdivurdering", "📡 Analytikere", "🎯 Triggere",
    "⚠️ Risiko", "🥧 Sektorfordeling", "📈 Historikk", "✏️ Rediger",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – POSISJONER
# ══════════════════════════════════════════════════════════════════════════════
with tab_pos:
    st.subheader("Alle posisjoner")
    d = df[["name","type","sector","shares","avg_cost","price","mkt_val","gain","gain_pct","day_chg"]].copy()
    d.columns = ["Selskap","Type","Sektor","Antall","Snittkurs","Nåkurs","Markedsverdi","Gevinst (kr)","Gevinst (%)","Dag %"]
    d["Type"] = d["Type"].map({"listed":"🔵 Notert","unlisted":"🔴 Unotert"})

    st.dataframe(
        d.style.format({
            "Snittkurs":    "{:.2f} kr",
            "Nåkurs":       "{:.2f} kr",
            "Markedsverdi": lambda x: fmt_nok(x),
            "Gevinst (kr)": lambda x: f"{'+'if x>=0 else''}{fmt_nok(x)}",
            "Gevinst (%)":  lambda x: fmt_pct(x),
            "Dag %":        lambda x: fmt_pct(x) if pd.notna(x) else "–",
        }).map(color_val, subset=["Gevinst (kr)","Gevinst (%)","Dag %"]),
        use_container_width=True, height=400,
    )
    st.markdown(f"**Total:** {fmt_nok(total_val)} &nbsp;|&nbsp; Gevinst: **{'+'if total_gain>=0 else''}{fmt_nok(total_gain)}** ({fmt_pct(total_gain_p)})")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – VERDIVURDERING
# ══════════════════════════════════════════════════════════════════════════════
with tab_val:
    st.subheader("Intrinsic value og margin of safety")
    sel_name = st.selectbox("Velg selskap", [h["name"] for h in st.session_state.holdings], key="val_sel")
    sel_h    = next(h for h in st.session_state.holdings if h["name"] == sel_name)
    sel_idx  = next(i for i, h in enumerate(st.session_state.holdings) if h["name"] == sel_name)
    sel_price = get_price(sel_h, prices)

    with st.spinner("Henter fundamentaldata..."):
        fund = fetch_fundamentals(sel_h["ticker"]) if sel_h["type"] == "listed" and sel_h["ticker"] else {}

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Nøkkeltall")
        for label, val in [
            ("Nåkurs",        f"{sel_price:.2f} kr"),
            ("EPS (ttm)",     fmt_n(fund.get("eps"))),
            ("EPS (fwd)",     fmt_n(fund.get("forward_eps"))),
            ("P/E (ttm)",     fmt_n(fund.get("pe"))),
            ("P/E (fwd)",     fmt_n(fund.get("forward_pe"))),
            ("PEG",           fmt_n(fund.get("peg"))),
            ("Bokverdi/aksje",fmt_n(fund.get("book_value"))),
            ("P/B",           fmt_n(fund.get("pb"))),
            ("FCF",           fmt_nok(fund.get("fcf")) if fund.get("fcf") else "–"),
            ("Utbytte",       f"{fund['div_yield']*100:.1f}%" if fund.get("div_yield") else "–"),
        ]:
            st.markdown(f"**{label}:** {val}")

        gnum = graham_number(fund.get("eps"), fund.get("book_value"))
        st.markdown("---")
        st.markdown("#### Graham Number")
        if gnum:
            mos_g = mos(gnum, sel_price)
            col = "#22c55e" if (mos_g and mos_g > 0) else "#ef4444"
            st.markdown(f"**Verdi:** {gnum:.2f} kr")
            st.markdown(f"**MoS:** <span style='color:{col};font-weight:bold'>{fmt_pct(mos_g)}</span>", unsafe_allow_html=True)
            if mos_g and mos_g > 20:
                st.success("✅ God margin of safety (>20%)")
            elif mos_g and mos_g > 0:
                st.warning("⚠️ Lav margin of safety")
            else:
                st.error("❌ Kurs over Graham-tall")
        else:
            st.info("Krever EPS > 0 og positiv bokverdi")

        st.markdown("---")
        st.markdown("#### Eget kursmål")
        own_tgt = st.number_input("Kursmål (kr)", value=float(sel_h.get("own_target") or sel_price),
                                   min_value=0.01, format="%.2f", key="own_tgt_input")
        if st.button("Lagre kursmål"):
            st.session_state.holdings[sel_idx]["own_target"] = own_tgt
            save_data(st.session_state.holdings)
            st.success("Lagret!")
        if sel_h.get("own_target"):
            up = (sel_h["own_target"] - sel_price) / sel_price * 100
            col = "#22c55e" if up >= 0 else "#ef4444"
            st.markdown(f"Oppside: <span style='color:{col};font-weight:bold'>{fmt_pct(up)}</span>", unsafe_allow_html=True)

    with col_right:
        st.markdown("#### DCF-modell")
        dcf_p = sel_h.get("dcf_params", {})
        fcf_auto    = fund.get("fcf")
        shares_auto = fund.get("shares_out")

        if sel_h["type"] == "listed" and fcf_auto and shares_auto:
            use_auto = st.checkbox("Bruk FCF fra Yahoo Finance", value=True)
            if use_auto:
                base_fcf   = fcf_auto
                shares_out = shares_auto
                st.info(f"FCF: {fmt_nok(fcf_auto)} | Aksjer: {shares_out:,.0f}")
            else:
                ca, cb = st.columns(2)
                base_fcf   = ca.number_input("FCF totalt (kr)", value=float(dcf_p.get("base_fcf", fcf_auto)), step=1e6, format="%.0f")
                shares_out = cb.number_input("Aksjer utestående", value=float(dcf_p.get("shares_out", shares_auto)), step=1e5, format="%.0f")
        else:
            ca, cb = st.columns(2)
            base_fcf   = ca.number_input("FCF / inntjening (kr)", value=float(dcf_p.get("base_fcf", 0)), step=1e5, format="%.0f")
            shares_out = cb.number_input("Aksjer utestående", value=float(dcf_p.get("shares_out", float(sel_h["shares"]))), step=1000.0, format="%.0f")

        r1, r2, r3, r4 = st.columns(4)
        growth   = r1.number_input("Vekst (%)",    value=float(dcf_p.get("growth_rate",   10.0)), min_value=-50.0, max_value=100.0, step=0.5, format="%.1f") / 100
        discount = r2.number_input("Diskontering (%)", value=float(dcf_p.get("discount_rate", 10.0)), min_value=1.0, max_value=30.0, step=0.5, format="%.1f") / 100
        tg       = r3.number_input("Terminal vekst (%)", value=float(dcf_p.get("terminal_growth", 2.5)), min_value=0.0, max_value=10.0, step=0.5, format="%.1f") / 100
        years    = r4.number_input("År", value=int(dcf_p.get("years", 10)), min_value=3, max_value=20)

        if st.button("Beregn DCF", type="primary"):
            dval = dcf_value(base_fcf, growth, discount, tg, years, shares_out)
            if dval:
                mos_d = mos(dval, sel_price)
                m1, m2, m3 = st.columns(3)
                m1.metric("DCF-verdi", f"{dval:.2f} kr")
                m2.metric("Nåkurs",    f"{sel_price:.2f} kr")
                m3.metric("Margin of Safety", fmt_pct(mos_d),
                          fmt_pct((dval - sel_price) / sel_price * 100))

                # Sensitivity: vary growth rate
                growths = [g / 100 for g in range(-5, 31, 5)]
                sens_vals = [dcf_value(base_fcf, g, discount, tg, years, shares_out) for g in growths]
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(
                    x=[g * 100 for g in growths], y=sens_vals,
                    mode="lines+markers", line=dict(color="#3b82f6", width=2)))
                fig_s.add_hline(y=sel_price, line_dash="dash", line_color="#ef4444",
                                annotation_text=f"Nåkurs {sel_price:.2f} kr")
                fig_s.update_layout(
                    title="Sensitivitet – vekstrate vs. DCF-verdi",
                    xaxis_title="Vekstrate (%)", yaxis_title="DCF-verdi (kr)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                st.plotly_chart(fig_s, use_container_width=True)

                st.session_state.holdings[sel_idx]["dcf_params"] = {
                    "base_fcf": base_fcf, "shares_out": shares_out,
                    "growth_rate": growth * 100, "discount_rate": discount * 100,
                    "terminal_growth": tg * 100, "years": years,
                }
                save_data(st.session_state.holdings)
            else:
                st.warning("Kan ikke beregne DCF – sjekk at FCF og aksjetall er satt og at diskonteringsrente > terminal vekst.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ANALYTIKERE
# ══════════════════════════════════════════════════════════════════════════════
with tab_ana:
    st.subheader("Analytikervurderinger og kursmål")
    listed_h = [h for h in st.session_state.holdings if h["type"] == "listed" and h["ticker"]]

    if not listed_h:
        st.info("Ingen noterte aksjer i porteføljen.")
    else:
        rows = []
        for h in listed_h:
            f = fetch_fundamentals(h["ticker"])
            price = get_price(h, prices)
            tm = f.get("target_mean")
            ot = h.get("own_target")
            rows.append({
                "Selskap":              h["name"],
                "Nåkurs":               price,
                "Anbefaling":           rec_label(f.get("rec_key")),
                "Analytiker snitt":     tm,
                "Analytiker høy":       f.get("target_high"),
                "Analytiker lav":       f.get("target_low"),
                "Antall analytikere":   f.get("num_analysts"),
                "Mitt kursmål":         ot,
                "Oppside analytiker":   (tm - price) / price * 100 if tm and price else None,
                "Oppside eget mål":     (ot - price) / price * 100 if ot and price else None,
            })

        ana_df = pd.DataFrame(rows)

        st.dataframe(
            ana_df.style.format({
                "Nåkurs":             "{:.2f} kr",
                "Analytiker snitt":   lambda x: f"{x:.2f} kr" if pd.notna(x) else "–",
                "Analytiker høy":     lambda x: f"{x:.2f} kr" if pd.notna(x) else "–",
                "Analytiker lav":     lambda x: f"{x:.2f} kr" if pd.notna(x) else "–",
                "Mitt kursmål":       lambda x: f"{x:.2f} kr" if pd.notna(x) and x else "–",
                "Antall analytikere": lambda x: str(int(x)) if pd.notna(x) else "–",
                "Oppside analytiker": lambda x: fmt_pct(x) if pd.notna(x) else "–",
                "Oppside eget mål":   lambda x: fmt_pct(x) if pd.notna(x) else "–",
            }).map(color_val, subset=["Oppside analytiker", "Oppside eget mål"]),
            use_container_width=True,
        )

        # Chart: nåkurs vs. analytiker range + eget mål
        chart_rows = [r for r in rows if r["Analytiker snitt"] is not None]
        if chart_rows:
            st.markdown("---")
            fig_a = go.Figure()
            for r in chart_rows:
                lo = r["Analytiker lav"]
                hi = r["Analytiker høy"]
                if lo and hi:
                    fig_a.add_trace(go.Bar(x=[r["Selskap"]], y=[hi - lo], base=[lo],
                                           marker_color="rgba(59,130,246,0.35)",
                                           name="Analytiker range", showlegend=False))
                fig_a.add_trace(go.Scatter(x=[r["Selskap"]], y=[r["Nåkurs"]],
                                           mode="markers", marker=dict(color="#ef4444", size=11, symbol="diamond"),
                                           name="Nåkurs", showlegend=(r == chart_rows[0])))
                if r["Analytiker snitt"]:
                    fig_a.add_trace(go.Scatter(x=[r["Selskap"]], y=[r["Analytiker snitt"]],
                                               mode="markers", marker=dict(color="#22c55e", size=11),
                                               name="Analytiker snitt", showlegend=(r == chart_rows[0])))
                if r["Mitt kursmål"]:
                    fig_a.add_trace(go.Scatter(x=[r["Selskap"]], y=[r["Mitt kursmål"]],
                                               mode="markers", marker=dict(color="#f59e0b", size=11, symbol="star"),
                                               name="Mitt kursmål", showlegend=(r == chart_rows[0])))

            fig_a.update_layout(title="Kursmål vs. nåkurs", yaxis_title="NOK",
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#e2e8f0", barmode="overlay")
            st.plotly_chart(fig_a, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – TRIGGERE
# ══════════════════════════════════════════════════════════════════════════════
with tab_trig:
    st.subheader("Triggere og katalysatorer")

    st.markdown("#### 📅 Kvartalsrapporter (Yahoo Finance)")
    for h in st.session_state.holdings:
        if h["type"] == "listed" and h["ticker"]:
            cal = fetch_calendar(h["ticker"])
            if cal and isinstance(cal, dict) and "Earnings Date" in cal:
                try:
                    earn = cal["Earnings Date"]
                    next_e = earn[0] if isinstance(earn, list) else earn
                    if hasattr(next_e, "date"):
                        days = (next_e.date() - date.today()).days
                        icon = "🟢" if days <= 30 else ("🟡" if days <= 90 else "⚪")
                        st.markdown(f"{icon} **{h['name']}** — {next_e.strftime('%d.%m.%Y')} ({days} dager)")
                except Exception:
                    pass

    st.markdown("---")
    st.markdown("#### ✏️ Egne triggere per selskap")
    trig_name = st.selectbox("Selskap", [h["name"] for h in st.session_state.holdings], key="trig_sel")
    trig_h    = next(h for h in st.session_state.holdings if h["name"] == trig_name)
    trig_idx  = next(i for i, h in enumerate(st.session_state.holdings) if h["name"] == trig_name)

    cats = trig_h.get("catalysts", [])
    if cats:
        cat_df = pd.DataFrame(cats).sort_values("date")
        st.dataframe(cat_df, use_container_width=True)

    with st.expander("Legg til trigger"):
        tc1, tc2 = st.columns(2)
        cat_type   = tc1.selectbox("Type", ["Kvartalsrapport","Kapitalmarkedsdag","Utbytte","Produktlansering","M&A","Regulatorisk","Annet"])
        cat_status = tc2.selectbox("Status", ["upcoming","done","cancelled"])
        cat_date   = st.date_input("Dato")
        cat_note   = st.text_input("Notat")
        if st.button("Legg til"):
            st.session_state.holdings[trig_idx].setdefault("catalysts", []).append(
                {"date": str(cat_date), "type": cat_type, "note": cat_note, "status": cat_status})
            save_data(st.session_state.holdings)
            st.success("Lagt til!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🗓 Alle kommende triggere")
    all_cats = []
    for h in st.session_state.holdings:
        for c in h.get("catalysts", []):
            if c.get("status") == "upcoming":
                all_cats.append({"Selskap": h["name"], **c})
    if all_cats:
        st.dataframe(pd.DataFrame(all_cats).sort_values("date"), use_container_width=True)
    else:
        st.info("Ingen kommende triggere registrert ennå.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – RISIKO
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("Risikoanalyse")
    risk_rows = []
    for h in st.session_state.holdings:
        if h["type"] == "listed" and h["ticker"]:
            f = fetch_fundamentals(h["ticker"])
            price = get_price(h, prices)
            hi52  = f.get("hi52")
            lo52  = f.get("lo52")
            risk_rows.append({
                "Selskap":       h["name"],
                "Beta":          f.get("beta"),
                "Gjeld/EK":      f.get("debt_equity"),
                "Current Ratio": f.get("current_ratio"),
                "ROE":           f.get("roe"),
                "Bruttomargin":  f.get("gross_margin"),
                "Nettom.":       f.get("net_margin"),
                "52u høy":       hi52,
                "52u lav":       lo52,
                "Fra topp (%)":  ((price - hi52) / hi52 * 100) if hi52 else None,
                "Fra bunn (%)":  ((price - lo52) / lo52 * 100) if lo52 else None,
            })

    if risk_rows:
        risk_df = pd.DataFrame(risk_rows)

        def color_beta(v):
            if not isinstance(v, float) or math.isnan(v): return ""
            if v < 0.8: return "color: #22c55e"
            if v < 1.2: return "color: #eab308"
            return "color: #ef4444"

        st.dataframe(
            risk_df.style.format({
                "Beta":          lambda x: fmt_n(x) if pd.notna(x) else "–",
                "Gjeld/EK":      lambda x: fmt_n(x) if pd.notna(x) else "–",
                "Current Ratio": lambda x: fmt_n(x) if pd.notna(x) else "–",
                "ROE":           lambda x: f"{x*100:.1f}%" if pd.notna(x) else "–",
                "Bruttomargin":  lambda x: f"{x*100:.1f}%" if pd.notna(x) else "–",
                "Nettom.":       lambda x: f"{x*100:.1f}%" if pd.notna(x) else "–",
                "52u høy":       lambda x: f"{x:.2f} kr" if pd.notna(x) else "–",
                "52u lav":       lambda x: f"{x:.2f} kr" if pd.notna(x) else "–",
                "Fra topp (%)":  lambda x: fmt_pct(x) if pd.notna(x) else "–",
                "Fra bunn (%)":  lambda x: fmt_pct(x) if pd.notna(x) else "–",
            }).map(color_beta, subset=["Beta"])
             .map(color_val,  subset=["Fra topp (%)","Fra bunn (%)"]),
            use_container_width=True,
        )

        beta_df = risk_df[risk_df["Beta"].notna()].sort_values("Beta")
        if not beta_df.empty:
            fig_b = px.bar(beta_df, x="Selskap", y="Beta",
                           title="Beta per aksje (markedsrisiko)",
                           color="Beta", color_continuous_scale="RdYlGn_r", text="Beta")
            fig_b.add_hline(y=1.0, line_dash="dash", line_color="white",
                            annotation_text="Markedet (Beta = 1)")
            fig_b.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#e2e8f0", showlegend=False)
            st.plotly_chart(fig_b, use_container_width=True)

        margin_cols = ["Selskap","Bruttomargin","Nettom."]
        mdf = risk_df[margin_cols].copy().dropna(subset=["Bruttomargin","Nettom."], how="all")
        if not mdf.empty:
            mdf["Bruttomargin"] = mdf["Bruttomargin"] * 100
            mdf["Nettom."]      = mdf["Nettom."] * 100
            melt = mdf.melt(id_vars="Selskap", var_name="Type", value_name="Margin (%)")
            fig_m = px.bar(melt.dropna(), x="Selskap", y="Margin (%)", color="Type",
                           barmode="group", title="Marginer per selskap (%)")
            fig_m.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#e2e8f0")
            st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("Ingen noterte aksjer med risikodata tilgjengelig.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – SEKTORFORDELING
# ══════════════════════════════════════════════════════════════════════════════
with tab_sek:
    col_a, col_b = st.columns(2)
    sector_df = df.groupby("sector")["mkt_val"].sum().reset_index()
    sector_df.columns = ["Sektor","Verdi"]
    sector_df["Andel"] = sector_df["Verdi"] / total_val * 100

    with col_a:
        fig_pie = px.pie(sector_df, values="Verdi", names="Sektor",
                         title="Fordeling per sektor", hole=0.45,
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig_pie.update_traces(textposition="outside", textinfo="label+percent")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_bar = px.bar(sector_df.sort_values("Verdi"), x="Verdi", y="Sektor",
                         orientation="h", title="Verdi per sektor",
                         color="Sektor", color_discrete_sequence=px.colors.qualitative.Bold)
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", showlegend=False, xaxis_title="NOK", yaxis_title="")
        fig_bar.update_traces(texttemplate="%{x:,.0f}", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Notert vs. Unotert")
    type_df = pd.DataFrame({
        "Type": ["🔵 Noterte","🔴 Unoterte"],
        "Verdi": [listed_val, unlisted_val],
        "Andel": [listed_val/total_val*100 if total_val else 0, unlisted_val/total_val*100 if total_val else 0],
    })
    fig_t = px.bar(type_df, x="Type", y="Verdi", color="Type",
                   color_discrete_map={"🔵 Noterte":"#3b82f6","🔴 Unoterte":"#a855f7"}, text="Andel")
    fig_t.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0", showlegend=False)
    st.plotly_chart(fig_t, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – HISTORIKK
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.subheader("Kurshistorikk – noterte aksjer")
    listed_h = [h for h in st.session_state.holdings if h["type"] == "listed" and h["ticker"]]

    if listed_h:
        selected  = st.multiselect("Velg aksjer", [h["name"] for h in listed_h],
                                    default=[h["name"] for h in listed_h[:3]])
        period    = st.select_slider("Periode", ["1mo","3mo","6mo","1y","2y","5y"], value="6mo")
        normalize = st.checkbox("Normaliser (relativ avkastning fra start)")

        ticker_map = {h["name"]: h["ticker"] for h in listed_h}
        sel_tickers = tuple(ticker_map[n] for n in selected if n in ticker_map)

        if sel_tickers:
            hist_df = get_history(sel_tickers, period)
            if not hist_df.empty:
                name_map = {v: k for k, v in ticker_map.items()}
                hist_df["Selskap"] = hist_df["Ticker"].map(name_map)
                if normalize:
                    hist_df["base"] = hist_df.groupby("Ticker")["Close"].transform("first")
                    hist_df["Avkastning (%)"] = (hist_df["Close"] / hist_df["base"] - 1) * 100
                    y_col, y_lbl = "Avkastning (%)", "Avkastning (%)"
                else:
                    y_col, y_lbl = "Close", "NOK"

                fig_h = px.line(hist_df, x="Date", y=y_col, color="Selskap",
                                title=f"Kursutvikling siste {period}")
                if normalize:
                    fig_h.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color="#e2e8f0", xaxis_title="", yaxis_title=y_lbl)
                st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Ingen noterte aksjer i porteføljen.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 – REDIGER
# ══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.subheader("Rediger posisjoner")
    st.caption("Endringer lagres til portfolio_data.json")

    for i, h in enumerate(st.session_state.holdings):
        with st.expander(f"{'🔵' if h['type']=='listed' else '🔴'} {h['name']}"):
            e1, e2, e3 = st.columns(3)
            new_shares = e1.number_input("Antall", value=h["shares"], key=f"sh_{i}", min_value=0)
            new_avg    = e2.number_input("Snittkurs", value=float(h["avg_cost"]), key=f"av_{i}", format="%.2f")
            new_mp     = e3.number_input("Manuell kurs", value=float(h["manual_price"] or 0), key=f"mp_{i}", format="%.2f") if h["type"] == "unlisted" else None
            new_ticker = e1.text_input("Ticker", value=h.get("ticker") or "", key=f"tk_{i}") if h["type"] == "listed" else None

            cs, cd = st.columns([3, 1])
            if cs.button("💾 Lagre", key=f"sv_{i}"):
                st.session_state.holdings[i]["shares"]   = new_shares
                st.session_state.holdings[i]["avg_cost"] = new_avg
                if h["type"] == "unlisted":
                    st.session_state.holdings[i]["manual_price"] = new_mp
                    st.session_state.holdings[i]["last_updated"] = str(date.today())
                if h["type"] == "listed" and new_ticker:
                    st.session_state.holdings[i]["ticker"] = new_ticker
                save_data(st.session_state.holdings)
                st.success("Lagret!")
                st.rerun()
            if cd.button("🗑 Slett", key=f"dl_{i}"):
                st.session_state.holdings.pop(i)
                save_data(st.session_state.holdings)
                st.rerun()
