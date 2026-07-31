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
import re
import io

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
            h.setdefault("transactions", [])
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
def recalc_from_transactions(transactions):
    """Weighted average cost and total shares from transaction log."""
    shares = 0.0
    cost_basis = 0.0
    for t in sorted(transactions, key=lambda x: x["date"]):
        qty = float(t["shares"])
        price = float(t["price_per_share"])
        if t["type"] == "buy":
            cost_basis += qty * price
            shares += qty
        elif t["type"] == "sell":
            if shares > 0:
                cost_basis = (cost_basis / shares) * max(0, shares - qty)
            shares = max(0, shares - qty)
    avg = (cost_basis / shares) if shares > 0 else 0.0
    return round(shares, 4), round(avg, 4)

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
        mp = h.get("manual_price")
        return mp if mp is not None and mp > 0 else h["avg_cost"]
    p = prices.get(h.get("ticker"))
    if p and p.get("price"):
        return p["price"]
    mp = h.get("manual_price")
    return mp if mp is not None and mp > 0 else h["avg_cost"]

def _nor(v, d=0):
    """Norwegian number format: 1.234.567,50"""
    s = f"{abs(v):,.{d}f}"  # English: 1,234,567.50
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # → 1.234.567,50
    return ("-" + s) if v < 0 else s

def fmt_nok(v, d=0):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "–"
    return f"{_nor(v, d)} kr"

def fmt_pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "–"
    sign = "+" if v >= 0 else ""
    return f"{sign}{_nor(v, 1)}%"

def fmt_n(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "–"
    return _nor(v, d)

def rec_label(key):
    return {"strong_buy": "🟢 Sterkt kjøp", "buy": "🟢 Kjøp",
            "hold": "🟡 Hold", "underperform": "🔴 Underperform",
            "sell": "🔴 Selg"}.get(key, key or "–")

def color_val(v):
    if isinstance(v, (int, float)) and not math.isnan(v):
        return "color: #22c55e" if v >= 0 else "color: #ef4444"
    return ""

def parse_nor_number(s):
    """Parse Norwegian number format: '1.234,56' → 1234.56, handles negatives."""
    s = str(s).strip().replace("\xa0", "").replace(" ", "")
    negative = s.startswith("-")
    s = s.lstrip("-")
    if "," in s:
        integer_part = s.split(",")[0].replace(".", "")
        decimal_part = s.split(",")[1]
        try:
            val = float(integer_part + "." + decimal_part)
        except ValueError:
            val = 0.0
    else:
        try:
            val = float(s.replace(".", "")) if s else 0.0
        except ValueError:
            val = 0.0
    return -val if negative else val

def parse_saldobalanse_calma(df):
    """
    Parse CALMA-style saldobalanse:
      1300–1399 → unoterte aksjer
      1850–1899 (UB > 0) → børsnoterte aksjer
    Extracts share count from Kontonavn where present.
    avg_cost = Utgående saldo / antall aksjer.
    """
    results = []
    for _, row in df.iterrows():
        try:
            konto     = int(str(row.iloc[0]).strip())
            kontonavn = str(row.iloc[1]).strip()
            ub        = parse_nor_number(row.iloc[4])  # Utgående saldo

            if ub <= 0:
                continue
            if 1300 <= konto <= 1399:
                stock_type = "unlisted"
            elif 1850 <= konto <= 1899:
                stock_type = "listed"
            else:
                continue

            # Extract share count from name (e.g. "500 AKSJER", "(100 aksjer", "- 5090 AKSJER")
            shares = None
            m = re.search(r'(\d[\d.,]*)\s+aksjer', kontonavn, re.IGNORECASE)
            if m:
                shares = parse_nor_number(m.group(1))

            # avg cost = UB / shares; fall back to UB as total cost with shares=1
            if shares and shares > 0:
                avg_cost = round(ub / shares, 4)
            else:
                shares   = 1.0
                avg_cost = round(ub, 4)

            # Clean company name: strip share-count parentheticals and suffixes
            name = kontonavn
            name = re.sub(r'\s*\([^)]*\d[^)]*\)', '', name)              # (21 AKSJER à...)
            name = re.sub(r'\s*[-–]\s*\d[\d.,]*\s+aksjer.*', '', name, flags=re.IGNORECASE)  # - 500 AKSJER À 11,33 NOK
            name = re.sub(r'\s+\d[\d.,]*\s+aksjer.*', '', name, flags=re.IGNORECASE)         # 53 aksjer a 46,52
            name = name.strip().strip('-–').strip()

            results.append({
                "konto":    konto,
                "name":     name,
                "type":     stock_type,
                "shares":   shares,
                "avg_cost": avg_cost,
                "ub":       ub,
                "raw_name": kontonavn,
            })
        except Exception:
            continue
    return results

EMPTY_COLS = ["id","name","ticker","type","sector","shares","avg_cost","manual_price",
              "last_updated","own_target","dcf_params","catalysts",
              "price","cost","mkt_val","gain","gain_pct","day_chg"]

def compute_portfolio(holdings, prices):
    if not holdings:
        return pd.DataFrame(columns=EMPTY_COLS)
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

    st.markdown("### 📂 Importer saldobalanse")
    with st.expander("Last opp CSV (CALMA-format)"):
        uploaded = st.file_uploader(
            "Velg saldobalanse (CSV med semikolon, norsk tallformat)",
            type=["csv","xlsx","xls"], key="sal_upload",
        )
        if uploaded:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    raw_bytes = uploaded.read()
                    for enc in ("utf-8-sig", "latin-1", "windows-1252", "utf-8"):
                        try:
                            raw_df = pd.read_csv(io.BytesIO(raw_bytes), sep=";", dtype=str, encoding=enc)
                            break
                        except (UnicodeDecodeError, Exception):
                            continue
                    else:
                        st.error("Kunne ikke lese CSV-filen – ukjent enkoding.")
                        st.stop()
                else:
                    raw_df = pd.read_excel(uploaded, dtype=str)

                parsed = parse_saldobalanse_calma(raw_df)

                if not parsed:
                    st.warning("Fant ingen aksjekontoer (1300–1399 eller 1850–1899 med UB > 0).")
                else:
                    prev_df = pd.DataFrame(parsed)[["konto","name","type","shares","avg_cost","ub","raw_name"]]
                    prev_df.columns = ["Konto","Navn (renset)","Type","Antall","Snittkurs (kr)","UB (kr)","Originalt kontonavn"]
                    prev_df["Type"] = prev_df["Type"].map({"unlisted":"🔴 Unotert","listed":"🔵 Notert"})
                    st.dataframe(prev_df, use_container_width=True, hide_index=True)
                    st.caption(f"{len(parsed)} posisjoner funnet. Allerede eksisterende (samme navn) hoppes over.")

                    existing_names = {h["name"].lower() for h in st.session_state.holdings}
                    new_count = sum(1 for p in parsed if p["name"].lower() not in existing_names)
                    st.info(f"{new_count} nye posisjoner vil bli importert ({len(parsed) - new_count} finnes allerede).")

                    if st.button("✅ Importer til portefølje", type="primary", use_container_width=True):
                        added = 0
                        for p in parsed:
                            if p["name"].lower() in existing_names:
                                continue
                            st.session_state.holdings.append({
                                "id":           f"sal_{p['konto']}_{datetime.now().timestamp()}",
                                "name":         p["name"],
                                "ticker":       None,
                                "type":         p["type"],
                                "sector":       "Annet",
                                "shares":       p["shares"],
                                "avg_cost":     p["avg_cost"],
                                "manual_price": p["avg_cost"],
                                "last_updated": str(date.today()),
                                "own_target":   None,
                                "dcf_params":   {},
                                "catalysts":    [],
                                "transactions": [{
                                    "date":            str(date.today()),
                                    "type":            "buy",
                                    "shares":          p["shares"],
                                    "price_per_share": p["avg_cost"],
                                    "note":            f"Importert fra saldobalanse (konto {p['konto']})",
                                }],
                            })
                            added += 1
                        save_data(st.session_state.holdings)
                        st.success(f"✅ {added} posisjoner importert!")
                        st.rerun()
            except Exception as e:
                st.error(f"Kunne ikke lese fil: {e}")

    st.markdown("---")
    st.caption(f"Sist oppdatert: {datetime.now().strftime('%H:%M:%S')}")

# ─── PRICES + PORTFOLIO ───────────────────────────────────────────────────────
with st.spinner("Henter kurser fra Oslo Børs..."):
    prices = fetch_prices(listed_tickers)

df = compute_portfolio(st.session_state.holdings, prices)

total_val    = df["mkt_val"].sum() if not df.empty else 0.0
total_cost   = df["cost"].sum()   if not df.empty else 0.0
total_gain   = total_val - total_cost
total_gain_p = (total_gain / total_cost * 100) if total_cost else 0
listed_val   = df[df["type"] == "listed"]["mkt_val"].sum()   if not df.empty else 0.0
unlisted_val = df[df["type"] == "unlisted"]["mkt_val"].sum() if not df.empty else 0.0

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("# 📊 Porteføljedashboard")
st.markdown(f"*CALMA HOLDING AS — {datetime.now().strftime('%d.%m.%Y %H:%M')}*")
st.markdown("---")

if df.empty:
    st.info("Ingen posisjoner ennå. Legg til aksjer via sidepanelet til venstre.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("💼 Total verdi",        fmt_nok(total_val),    f"Kostpris {fmt_nok(total_cost)}")
c2.metric("📈 Urealisert gevinst", fmt_nok(total_gain),   fmt_pct(total_gain_p), delta_color="normal")
c3.metric("🔵 Noterte aksjer",     fmt_nok(listed_val),   f"{_nor(listed_val/total_val*100, 1)}% av total" if total_val else "–")
c4.metric("🔴 Unoterte aksjer",    fmt_nok(unlisted_val), f"{_nor(unlisted_val/total_val*100, 1)}% av total" if total_val else "–")
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
    if df.empty:
        st.info("Ingen posisjoner ennå. Legg til aksjer via sidepanelet.")
    else:
        d = df[["name","type","sector","shares","avg_cost","price","mkt_val","gain","gain_pct","day_chg"]].copy()
        d.columns = ["Selskap","Type","Sektor","Antall","Snittkurs","Nåkurs","Markedsverdi","Gevinst (kr)","Gevinst (%)","Dag %"]
        d["Type"] = d["Type"].map({"listed":"🔵 Notert","unlisted":"🔴 Unotert"})
        st.dataframe(
            d.style.format({
                "Snittkurs":    lambda x: fmt_nok(x, 2),
                "Nåkurs":       lambda x: fmt_nok(x, 2),
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
            ("Nåkurs",        fmt_nok(sel_price, 2)),
            ("EPS (ttm)",     fmt_n(fund.get("eps"))),
            ("EPS (fwd)",     fmt_n(fund.get("forward_eps"))),
            ("P/E (ttm)",     fmt_n(fund.get("pe"))),
            ("P/E (fwd)",     fmt_n(fund.get("forward_pe"))),
            ("PEG",           fmt_n(fund.get("peg"))),
            ("Bokverdi/aksje",fmt_n(fund.get("book_value"))),
            ("P/B",           fmt_n(fund.get("pb"))),
            ("FCF",           fmt_nok(fund.get("fcf")) if fund.get("fcf") else "–"),
            ("Utbytte",       f"{_nor(fund['div_yield']*100, 1)}%" if fund.get("div_yield") else "–"),
        ]:
            st.markdown(f"**{label}:** {val}")

        gnum = graham_number(fund.get("eps"), fund.get("book_value"))
        st.markdown("---")
        st.markdown("#### Graham Number")
        if gnum:
            mos_g = mos(gnum, sel_price)
            col = "#22c55e" if (mos_g and mos_g > 0) else "#ef4444"
            st.markdown(f"**Verdi:** {fmt_nok(gnum, 2)}")
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
                st.info(f"FCF: {fmt_nok(fcf_auto)} | Aksjer: {_nor(shares_out, 0)}")
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
                m1.metric("DCF-verdi", fmt_nok(dval, 2))
                m2.metric("Nåkurs",    fmt_nok(sel_price, 2))
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
                                annotation_text=f"Nåkurs {_nor(sel_price, 2)} kr")
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
                "Nåkurs":             lambda x: fmt_nok(x, 2),
                "Analytiker snitt":   lambda x: fmt_nok(x, 2) if pd.notna(x) else "–",
                "Analytiker høy":     lambda x: fmt_nok(x, 2) if pd.notna(x) else "–",
                "Analytiker lav":     lambda x: fmt_nok(x, 2) if pd.notna(x) else "–",
                "Mitt kursmål":       lambda x: fmt_nok(x, 2) if pd.notna(x) and x else "–",
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
                "ROE":           lambda x: f"{_nor(x*100, 1)}%" if pd.notna(x) else "–",
                "Bruttomargin":  lambda x: f"{_nor(x*100, 1)}%" if pd.notna(x) else "–",
                "Nettom.":       lambda x: f"{_nor(x*100, 1)}%" if pd.notna(x) else "–",
                "52u høy":       lambda x: fmt_nok(x, 2) if pd.notna(x) else "–",
                "52u lav":       lambda x: fmt_nok(x, 2) if pd.notna(x) else "–",
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
    st.caption("Snittkurs beregnes automatisk fra transaksjonslogg. Endringer lagres til portfolio_data.json.")

    for i, h in enumerate(st.session_state.holdings):
        txns = h.get("transactions", [])
        label = f"{'🔵' if h['type']=='listed' else '🔴'} {h['name']}  —  {_nor(h['shares'], 0)} aksjer  |  snittkurs {_nor(h['avg_cost'], 2)} kr"
        with st.expander(label):

            # ── Grunnleggende felt ────────────────────────────────────────────
            SECTORS = ["Industri","Shipping","Energi","Finans","IT","EdTech","Telekom","Cleantech","Fintech","Annet"]
            e1, e2, e3 = st.columns(3)
            new_ticker = e1.text_input("Ticker", value=h.get("ticker") or "", key=f"tk_{i}") if h["type"] == "listed" else None
            new_name   = e2.text_input("Selskapsnavn", value=h["name"], key=f"nm_{i}")
            new_sector = e3.selectbox("Sektor", SECTORS,
                                      index=SECTORS.index(h.get("sector","Annet")) if h.get("sector","Annet") in SECTORS else 0,
                                      key=f"sec_{i}")

            cs, cd = st.columns([3, 1])
            if cs.button("💾 Lagre navn / sektor", key=f"sv_{i}"):
                st.session_state.holdings[i]["name"]   = new_name
                st.session_state.holdings[i]["sector"] = new_sector
                if h["type"] == "listed" and new_ticker is not None:
                    st.session_state.holdings[i]["ticker"] = new_ticker or None
                save_data(st.session_state.holdings)
                st.success("Lagret!")
                st.rerun()
            if cd.button("🗑 Slett", key=f"dl_{i}"):
                st.session_state.holdings.pop(i)
                save_data(st.session_state.holdings)
                st.rerun()

            st.markdown("---")

            # ── Korriger antall aksjer ────────────────────────────────────────
            st.markdown("#### ✏️ Korriger antall aksjer")
            total_ub = round(h["avg_cost"] * h["shares"], 2)
            ka1, ka2, ka3 = st.columns(3)
            korr_shares = ka1.number_input(
                "Antall aksjer", value=float(h["shares"]), min_value=0.0,
                step=1.0, format="%.4f", key=f"ksh_{i}")
            ka2.metric("Total bokverdi (UB)", fmt_nok(total_ub))
            preview_avg = round(total_ub / korr_shares, 4) if korr_shares > 0 else 0.0
            ka3.metric("Ny snittkurs", fmt_nok(preview_avg, 2),
                       delta=f"{'+' if preview_avg >= h['avg_cost'] else ''}{_nor(preview_avg - h['avg_cost'], 2)} kr")
            st.caption("Bokverdi (UB) holdes fast — snittkurs = UB ÷ antall aksjer. Nåkurs synkroniseres automatisk.")
            if st.button("Oppdater antall + snittkurs", key=f"kshbtn_{i}"):
                if korr_shares > 0:
                    st.session_state.holdings[i]["shares"]       = korr_shares
                    st.session_state.holdings[i]["avg_cost"]     = preview_avg
                    st.session_state.holdings[i]["manual_price"] = preview_avg
                else:
                    st.session_state.holdings[i]["shares"] = 0.0
                save_data(st.session_state.holdings)
                st.success(f"Oppdatert: {_nor(korr_shares, 2)} aksjer · snittkurs {fmt_nok(preview_avg, 2)}")
                st.rerun()

            st.markdown("---")

            # ── Oppdater kurs / verdivurdering ───────────────────────────────
            st.markdown("#### 💰 Oppdater kurs")
            if h["type"] == "unlisted":
                shares   = h["shares"]
                cur_mp   = h.get("manual_price") or h["avg_cost"]
                kostpris = round(h["avg_cost"] * shares, 2)

                kk1, kk2 = st.columns(2)
                new_mp   = kk1.number_input(
                    "Nåkurs per aksje (kr)",
                    value=float(cur_mp), min_value=0.0, format="%.2f", key=f"mp_{i}")
                mp_note  = kk2.text_input("Kilde / notat",
                                          placeholder="f.eks. Emisjon Q2 2026", key=f"mpn_{i}")

                new_mkt = new_mp * shares
                st.markdown(
                    f"**Markedsverdi** = {fmt_nok(new_mp, 2)} × {_nor(shares, 0)} aksjer "
                    f"= **{fmt_nok(new_mkt)}**"
                )
                mk1, mk2 = st.columns(2)
                mk1.metric("Kostpris (UB)", fmt_nok(kostpris))
                urealisert = new_mkt - kostpris
                mk2.metric("Urealisert gevinst/tap", fmt_nok(urealisert),
                           delta=fmt_pct(urealisert / kostpris * 100) if kostpris else "–")

                if st.button("Oppdater kurs", key=f"mpbtn_{i}"):
                    st.session_state.holdings[i]["manual_price"] = new_mp
                    st.session_state.holdings[i]["last_updated"] = str(date.today())
                    if mp_note:
                        st.session_state.holdings[i].setdefault("catalysts", []).append({
                            "date": str(date.today()), "type": "Kursoppdatering",
                            "note": mp_note, "status": "done",
                        })
                    save_data(st.session_state.holdings)
                    st.success(f"Kurs oppdatert: {fmt_nok(new_mp, 2)} × {_nor(shares, 0)} aksjer = {fmt_nok(new_mkt)}")
                    st.rerun()
            else:
                # Børsnotert: vis live-kurs, tillat manuell overstyr når ticker mangler
                live = prices.get(h.get("ticker"))
                live_price = live["price"] if live and live.get("price") else None
                cur_manual = h.get("manual_price")
                if live_price:
                    st.success(f"Live-kurs fra Yahoo Finance: **{fmt_nok(live_price, 2)}**")
                    if cur_manual:
                        st.caption(f"Manuell overstyr aktiv ({fmt_nok(cur_manual, 2)}) — fjern for å bruke børskurs.")
                else:
                    st.warning("Ingen kurs fra Yahoo Finance. Sett manuell kurs nedenfor.")
                kk1, kk2 = st.columns(2)
                new_lmp  = kk1.number_input("Manuell kurs per aksje (kr)",
                                            value=float(cur_manual or live_price or h["avg_cost"]),
                                            min_value=0.0, format="%.4f", key=f"lmp_{i}")
                lmp_note = kk2.text_input("Notat", placeholder="f.eks. Hacksaw AB – kurs 2026-07-31", key=f"lmpn_{i}")
                lc1, lc2 = st.columns(2)
                if lc1.button("💾 Lagre manuell kurs", key=f"lmpbtn_{i}"):
                    st.session_state.holdings[i]["manual_price"] = new_lmp
                    if lmp_note:
                        st.session_state.holdings[i].setdefault("catalysts", []).append({
                            "date": str(date.today()), "type": "Kursoppdatering",
                            "note": lmp_note, "status": "done",
                        })
                    save_data(st.session_state.holdings)
                    st.success(f"Manuell kurs satt til {fmt_nok(new_lmp, 2)}")
                    st.rerun()
                if lc2.button("🔄 Bruk børskurs (fjern manuell)", key=f"lmpclr_{i}"):
                    st.session_state.holdings[i]["manual_price"] = None
                    save_data(st.session_state.holdings)
                    st.success("Manuell kurs fjernet – bruker børskurs.")
                    st.rerun()

            st.markdown("---")

            # ── Transaksjonslogg ──────────────────────────────────────────────
            st.markdown("#### 🧾 Transaksjoner")
            if txns:
                txn_df = pd.DataFrame(txns).sort_values("date", ascending=False)
                txn_df["Pris/aksje"] = txn_df["price_per_share"].apply(lambda x: fmt_nok(x, 2))
                txn_df["Total"]      = (txn_df["shares"] * txn_df["price_per_share"]).apply(lambda x: fmt_nok(x))
                disp = txn_df[["date","type","shares","Pris/aksje","Total","note"]].copy()
                disp.columns = ["Dato","Type","Antall","Pris/aksje","Total","Notat"]
                disp["Type"] = disp["Type"].map({"buy":"🟢 Kjøp","sell":"🔴 Salg"})
                st.dataframe(disp, use_container_width=True, hide_index=True)
                # Sletteknapp per rad
                del_idx = st.selectbox("Slett transaksjon (radnummer)", ["–"] + list(range(len(txns))), key=f"delt_{i}")
                if del_idx != "–" and st.button("Slett valgt transaksjon", key=f"deltbtn_{i}"):
                    st.session_state.holdings[i]["transactions"].pop(int(del_idx))
                    new_s, new_a = recalc_from_transactions(st.session_state.holdings[i]["transactions"])
                    st.session_state.holdings[i]["shares"]   = new_s
                    st.session_state.holdings[i]["avg_cost"] = new_a
                    save_data(st.session_state.holdings)
                    st.rerun()
            else:
                st.info("Ingen transaksjoner registrert ennå.")

            # ── Legg til ny transaksjon ───────────────────────────────────────
            st.markdown("#### ➕ Legg til transaksjon")
            t1, t2, t3 = st.columns(3)
            t_type   = t1.selectbox("Type", ["buy","sell"], format_func=lambda x: "🟢 Kjøp" if x=="buy" else "🔴 Salg", key=f"ttype_{i}")
            t_date   = t2.date_input("Dato", value=date.today(), key=f"tdate_{i}")
            t_shares = t3.number_input("Antall aksjer", min_value=0.0, value=0.0, step=1.0, format="%.4f", key=f"tshares_{i}")

            input_mode = st.radio("Angi pris som", ["Pris per aksje (kr)", "Total beløp (kr)"],
                                  horizontal=True, key=f"tmode_{i}")
            ta, tb = st.columns(2)
            if input_mode == "Pris per aksje (kr)":
                t_price = ta.number_input("Pris per aksje (kr)", min_value=0.0, value=0.0, format="%.4f", key=f"tprice_{i}")
                t_total = t_shares * t_price
                tb.metric("Total beløp", fmt_nok(t_total))
            else:
                t_total_input = ta.number_input("Total beløp (kr)", min_value=0.0, value=0.0, format="%.2f", key=f"ttotal_{i}")
                t_price = t_total_input / t_shares if t_shares > 0 else 0.0
                t_total = t_total_input
                tb.metric("Pris per aksje", fmt_nok(t_price, 2))

            t_note = st.text_input("Notat (valgfritt)", key=f"tnote_{i}")

            if st.button("Legg til transaksjon", type="primary", key=f"tadd_{i}"):
                if t_shares > 0 and t_price > 0:
                    st.session_state.holdings[i].setdefault("transactions", []).append({
                        "date":            str(t_date),
                        "type":            t_type,
                        "shares":          t_shares,
                        "price_per_share": t_price,
                        "note":            t_note,
                    })
                    new_s, new_a = recalc_from_transactions(st.session_state.holdings[i]["transactions"])
                    st.session_state.holdings[i]["shares"]   = new_s
                    st.session_state.holdings[i]["avg_cost"] = new_a
                    save_data(st.session_state.holdings)
                    st.success(f"✅ Lagt til: {_nor(t_shares, 2)} aksjer à {fmt_nok(t_price, 2)} = {fmt_nok(t_total)}")
                    st.rerun()
                else:
                    st.warning("Fyll inn antall aksjer og pris.")
