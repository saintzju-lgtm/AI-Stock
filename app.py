import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import threading
import requests
from io import StringIO
from curl_cffi import requests as cffi_requests

# ==========================================
# 0. 页面全局配置与时区定义
# ==========================================
st.set_page_config(layout="wide", page_title="专业量化决策终端 Pro")

BEIJING_TZ = ZoneInfo("Asia/Shanghai")
US_EASTERN_TZ = ZoneInfo("America/New_York")  # 自动处理夏令时/冬令时
GLOBAL_RATE_INTERVAL = 1.0
RISK_FREE_RATE = 0.045  # 无风险利率(用于 Black-Scholes Gamma)

# ==========================================
# 🧠 1. 全局解耦内存存储中心
# ==========================================
@st.cache_resource
def get_global_data_store():
    return {
        "stock_cache": {},
        "options_cache": {},
        "exp_list_cache": {},
        "darkpool_cache": {},
        "macro_cache": {"data": None, "ts": 0.0},
        "fetch_timestamps": {},
        "active_queue": set(["AAPL", "TSLA", "NVDA", "BTDR"]),
        "lock": threading.Lock(),
        "last_yahoo_call_ts": 0.0,
    }

GLOBAL_STORE = get_global_data_store()

def get_dynamic_ttl():
    """盘后/周末 TTL 自动延长至 30 分钟，美股交易时段保持 3 分钟"""
    now_et = datetime.now(US_EASTERN_TZ)
    if now_et.weekday() >= 5:
        return 1800
    t_min = now_et.hour * 60 + now_et.minute
    if t_min < 9 * 60 + 30 or t_min >= 16 * 60:
        return 1800
    return 180

def _throttled_yahoo_call(func, max_retries=1):
    last_err = None
    for attempt in range(max_retries + 1):
        with GLOBAL_STORE["lock"]:
            wait = GLOBAL_RATE_INTERVAL - (time.time() - GLOBAL_STORE["last_yahoo_call_ts"])
            if wait > 0:
                time.sleep(wait)
            GLOBAL_STORE["last_yahoo_call_ts"] = time.time()
        try:
            return func()
        except Exception as e:
            last_err = e
            msg = str(e)
            if ("429" in msg or "Too Many Requests" in msg) and attempt < max_retries:
                time.sleep(4 + attempt * 4)
                continue
            raise
    raise last_err

# ==========================================
# 🔍 2. 全美股 + ETF 动态股票库 (Nasdaq Trader + SEC)
# ==========================================
@st.cache_data(ttl=86400)
def load_us_stock_library():
    """合并 Nasdaq Trader (含全美股+ETF) 与 SEC EDGAR，支持全市场 10,000+ 标的"""
    stock_dict = {
        "AAPL": "Apple Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla, Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com, Inc.",
        "GOOG": "Alphabet Inc.",
        "META": "Meta Platforms, Inc.",
        "BTDR": "Bitdeer Technologies Group",
        "SPY": "SPDR S&P 500 ETF Trust",
        "QQQ": "Invesco QQQ Trust",
        "IWM": "iShares Russell 2000 ETF",
    }

    # 通道 1: Nasdaq Trader 官方全量列表 (含 NASDAQ, NYSE, AMEX 及所有 ETF)
    nasdaq_urls = [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol", "Security Name"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol", "Security Name"),
    ]
    for url, sym_col, name_col in nasdaq_urls:
        try:
            res = requests.get(url, timeout=8)
            if res.status_code == 200:
                lines = [l for l in res.text.splitlines() if not l.startswith("File Creation Time")]
                df = pd.read_csv(StringIO("\n".join(lines)), sep="|")
                for _, row in df.iterrows():
                    sym = str(row.get(sym_col, "")).strip()
                    name = str(row.get(name_col, "")).strip()
                    if sym and sym.isalpha() and len(sym) <= 5 and sym not in stock_dict:
                        stock_dict[sym] = name.split(" - ")[0][:45]
        except Exception:
            pass

    # 通道 2: SEC EDGAR 补充 (必须带合规 User-Agent)
    try:
        sec_headers = {"User-Agent": "QuantTerminalPro/2.0 (research@quantterminal.local)"}
        res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=sec_headers, timeout=6)
        if res.status_code == 200:
            data = res.json()
            for item in data.values():
                sym = str(item.get("ticker", "")).strip().upper()
                title = str(item.get("title", "")).strip().title()
                if sym and sym not in stock_dict:
                    stock_dict[sym] = title[:45]
    except Exception:
        pass

    return [f"{k} - {v}" for k, v in stock_dict.items()]

STOCK_LIBRARY = load_us_stock_library()

# ==========================================
# 📐 3. 精准股本与换手率计算 (解耦 Session State)
# ==========================================
def fetch_share_stats(ticker_symbol):
    """同时获取流通股本 (Float) 与总股本 (Outstanding)"""
    info = {}
    try:
        info = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).info)
    except Exception:
        info = {}

    float_shares = info.get("floatShares") if info else None
    shares_out = info.get("sharesOutstanding") if info else None

    float_shares = float(float_shares) if float_shares and float_shares > 0 else None
    shares_out = float(shares_out) if shares_out and shares_out > 0 else None
    return float_shares, shares_out

def calculate_turnover_metrics(volume, capital, timestamp_unix):
    if not capital or capital <= 0 or not volume or volume <= 0:
        return 0.0, 0.0, "数据不足"

    realtime_turnover = (volume / capital) * 100.0
    dt_et = datetime.fromtimestamp(timestamp_unix, tz=US_EASTERN_TZ)

    weekday = dt_et.weekday()
    time_min = dt_et.hour * 60 + dt_et.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
    total_trade_min = 390

    if weekday < 5 and open_min <= time_min < close_min:
        elapsed = time_min - open_min
        if elapsed < 30:
            return realtime_turnover, realtime_turnover, f"盘中(开盘{elapsed}分, 暂不外推)"
        projected_vol = volume * (total_trade_min / elapsed)
        projected_turnover = (projected_vol / capital) * 100.0
        return realtime_turnover, projected_turnover, f"盘中外推(已交易{elapsed}分)"
    elif weekday < 5 and time_min < open_min:
        return realtime_turnover, realtime_turnover, "盘前/未开盘"
    else:
        return realtime_turnover, realtime_turnover, "已收盘"

# ==========================================
# 🌑 4. 真实暗池 / 场外数据引擎 (FINRA Reg SHO)
# ==========================================
def fetch_finra_darkpool_data(ticker_symbol, lookback_days=7):
    """
    从 FINRA 官方抓取近 N 个交易日的场外/暗池 (TRF/ADF) 成交与做空数据。
    说明：FINRA 数据在美东时间每日盘后更新，涵盖全美 NMS 股票的场外撮合量。
    """
    cache_key = f"dp_{ticker_symbol}"
    cached = GLOBAL_STORE["darkpool_cache"].get(cache_key)
    if cached and (time.time() - cached["ts"] < 3600):
        return cached["df"]

    records = []
    today_et = datetime.now(US_EASTERN_TZ).date()
    checked = 0
    delta = 0

    while len(records) < lookback_days and delta < 15:
        d = today_et - timedelta(days=delta)
        delta += 1
        if d.weekday() >= 5:
            continue
        checked += 1
        date_str = d.strftime("%Y%m%d")
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date_str}.txt"
        try:
            res = requests.get(url, timeout=4)
            if res.status_code == 200:
                # 快速按行定位目标 ticker，避免解析整个大文件
                target_prefix = f"{date_str}|{ticker_symbol.upper()}|"
                for line in res.text.splitlines():
                    if line.startswith(target_prefix):
                        parts = line.split("|")
                        # 格式: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
                        if len(parts) >= 5:
                            short_vol = float(parts[2])
                            total_off_ex = float(parts[4])
                            short_ratio = (short_vol / total_off_ex * 100) if total_off_ex > 0 else 0.0
                            records.append({
                                "日期": d,
                                "暗池/场外总量": int(total_off_ex),
                                "暗池做空量": int(short_vol),
                                "暗池做空占比%": round(short_ratio, 2),
                            })
                        break
        except Exception:
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("日期").sort_index(ascending=False)

    with GLOBAL_STORE["lock"]:
        GLOBAL_STORE["darkpool_cache"][cache_key] = {"df": df, "ts": time.time()}
    return df

# ==========================================
# 🚀 5. K 线与宏观行情抓取引擎
# ==========================================
def fetch_chart_yahoo_direct(symbol):
    try:
        session = cffi_requests.Session(impersonate="chrome110")
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range=120d&interval=1d&includePrePost=false"
        res = session.get(url, timeout=5)
        if res.status_code == 200:
            result = res.json()["chart"]["result"][0]
            meta = result.get("meta", {})
            timestamps = result.get("timestamp", [])
            quote = result["indicators"]["quote"][0]

            dt_index = [datetime.fromtimestamp(ts, tz=US_EASTERN_TZ).date() for ts in timestamps]
            df = pd.DataFrame({
                "Open": quote.get("open", []),
                "High": quote.get("high", []),
                "Low": quote.get("low", []),
                "Close": quote.get("close", []),
                "Volume": quote.get("volume", []),
            }, index=dt_index)

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(how="all", subset=["Open", "High", "Low", "Close"]).copy()

            p_close = meta.get("regularMarketPrice")
            p_high = meta.get("regularMarketDayHigh", p_close)
            p_low = meta.get("regularMarketDayLow", p_close)
            p_vol = meta.get("regularMarketVolume", 0)
            m_time = meta.get("regularMarketTime")

            if m_time and p_close and p_close > 0:
                m_date = datetime.fromtimestamp(m_time, tz=US_EASTERN_TZ).date()
                p_open = meta.get("regularMarketOpen", p_close)
                if df.empty or df.index[-1] < m_date:
                    patch_row = pd.DataFrame(
                        {"Open": [p_open], "High": [p_high], "Low": [p_low], "Close": [p_close], "Volume": [p_vol]},
                        index=[m_date],
                    )
                    df = pd.concat([df, patch_row])
                elif df.index[-1] == m_date:
                    df.loc[m_date, "Close"] = p_close
                    if pd.isna(df.loc[m_date, "Open"]) or df.loc[m_date, "Open"] == 0:
                        df.loc[m_date, "Open"] = p_open
                    df.loc[m_date, "High"] = max(float(df.loc[m_date, "High"] or 0), float(p_high or p_close), float(p_close))
                    low_cur = float(df.loc[m_date, "Low"] or p_close)
                    df.loc[m_date, "Low"] = min(low_cur, float(p_low or p_close), float(p_close))
                    if p_vol and p_vol > float(df.loc[m_date, "Volume"] or 0):
                        df.loc[m_date, "Volume"] = p_vol

            df = df[~df.index.duplicated(keep="last")].sort_index().ffill().bfill()
            if len(df) >= 2:
                return df, "Yahoo TLS 引擎"
    except Exception:
        pass
    return pd.DataFrame(), ""

def fetch_chart_stooq_csv(symbol):
    try:
        clean_sym = symbol.lower().split(".")[0].replace("^", "")
        stooq_sym = f"{clean_sym}.us" if not symbol.startswith("^") and "-USD" not in symbol else clean_sym
        url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
        if res.status_code == 200 and "Date" in res.text:
            df = pd.read_csv(StringIO(res.text))
            df.columns = [c.capitalize() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Close"]).copy()
            df.index = pd.to_datetime(df.index).date
            if len(df) >= 2:
                return df, "Stooq CSV 通道"
    except Exception:
        pass
    return pd.DataFrame(), ""

def fetch_macro_bundle():
    """全局缓存宏观指标，避免切换个股时重复请求"""
    mc = GLOBAL_STORE["macro_cache"]
    if mc["data"] and (time.time() - mc["ts"] < 180):
        return mc["data"]

    def _one(symbol):
        try:
            session = cffi_requests.Session(impersonate="chrome110")
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
            res = session.get(url, timeout=3)
            if res.status_code == 200:
                meta = res.json()["chart"]["result"][0]["meta"]
                price = meta.get("regularMarketPrice", 0.0)
                prev = meta.get("chartPreviousClose", meta.get("previousClose", price))
                if price > 0:
                    return price, ((price / prev - 1) if prev else 0.0)
        except Exception:
            pass
        return 0.0, 0.0

    btc, _ = _one("BTC-USD")
    nasdaq, nasdaq_pct = _one("^IXIC")
    vix, vix_pct = _one("^VIX")
    bundle = {"btc": btc, "nasdaq": nasdaq, "nasdaq_pct": nasdaq_pct, "vix": vix, "vix_pct": vix_pct}
    with GLOBAL_STORE["lock"]:
        GLOBAL_STORE["macro_cache"] = {"data": bundle, "ts": time.time()}
    return bundle

def get_live_quote(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        fi = _throttled_yahoo_call(lambda: tk.fast_info)
        price = fi.get("last_price") or fi.get("lastPrice")
        volume = fi.get("last_volume") or fi.get("lastVolume") or fi.get("regular_market_volume")
        return price, volume
    except Exception:
        return None, None

# ==========================================
# 📈 6. 场景回归预测模型
# ==========================================
def build_scenario_model(hist_df):
    fit_df = hist_df.dropna(subset=["Open", "High", "Low", "Close", "昨收"]).copy()
    if len(fit_df) < 15:
        return {"mode": "insufficient"}

    fit_df["gap"] = (fit_df["Open"] - fit_df["昨收"]) / fit_df["昨收"]
    fit_df["recent_vol"] = fit_df["Close"].pct_change().rolling(10).std()
    fit_df = fit_df.dropna(subset=["gap", "recent_vol"])
    if len(fit_df) < 15:
        return {"mode": "insufficient"}

    y_h = (fit_df["High"] / fit_df["昨收"] - 1).values
    y_l = (fit_df["Low"] / fit_df["昨收"] - 1).values

    X_simple = fit_df[["gap"]].values
    fallback = {}
    for tag, y in [("h", y_h), ("l", y_l)]:
        m = LinearRegression().fit(X_simple, y)
        fallback[f"s_{tag}"], fallback[f"i_{tag}"] = m.coef_[0], m.intercept_

    if len(fit_df) >= 25:
        try:
            from sklearn.linear_model import QuantileRegressor
            X = fit_df[["gap", "recent_vol"]].values
            models = {"h": {}, "l": {}}
            for scene, q in {"乐观": 0.9, "中性": 0.5, "悲观": 0.1}.items():
                models["h"][scene] = QuantileRegressor(quantile=q, alpha=0.3, solver="highs").fit(X, y_h)
                models["l"][scene] = QuantileRegressor(quantile=q, alpha=0.3, solver="highs").fit(X, y_l)
            return {"mode": "quantile", "models": models, "params": fallback}
        except Exception:
            pass

    return {"mode": "linear_fallback", "params": fallback}

def predict_scenarios(scenario_model, hist_df):
    if not scenario_model or scenario_model.get("mode") == "insufficient":
        return None, "insufficient"

    last = hist_df.iloc[-1]
    prev_close = last["昨收"]
    if pd.isna(prev_close) or prev_close <= 0:
        return None, "insufficient"

    open_p = last["Open"] if pd.notnull(last["Open"]) else prev_close
    gap = (open_p - prev_close) / prev_close
    if pd.isna(gap) or np.isinf(gap):
        gap = 0.0

    if scenario_model["mode"] == "quantile":
        try:
            ret = hist_df["Close"].pct_change()
            recent_vol = ret.rolling(10).std().iloc[-1]
            if pd.isna(recent_vol) or np.isinf(recent_vol):
                recent_vol = ret.std() or 0.01
            X_pred = np.array([[float(gap), float(recent_vol)]])
            models = scenario_model["models"]
            h_sorted = sorted([prev_close * (1 + models["h"][s].predict(X_pred)[0]) for s in ["乐观", "中性", "悲观"]], reverse=True)
            l_sorted = sorted([prev_close * (1 + models["l"][s].predict(X_pred)[0]) for s in ["乐观", "中性", "悲观"]], reverse=True)
            return [(s, h_sorted[i], l_sorted[i]) for i, s in enumerate(["乐观", "中性", "悲观"])], "quantile"
        except Exception:
            pass

    if scenario_model.get("params"):
        p = scenario_model["params"]
        p_h = prev_close * (1 + (p["i_h"] + p["s_h"] * gap))
        p_l = prev_close * (1 + (p["i_l"] + p["s_l"] * gap))
        return [("乐观", p_h * 1.06, p_l * 1.06), ("中性", p_h, p_l), ("悲观", p_h * 0.94, p_l * 0.94)], "linear_fallback"

    return None, "insufficient"

# ==========================================
# 🎯 7. 核心数据聚合 (线程安全，不触碰 session_state)
# ==========================================
def do_fetch_stock_data(ticker_symbol):
    try:
        now_ts = time.time()
        fetch_time_bj = datetime.fromtimestamp(now_ts, tz=BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")

        hist, source = fetch_chart_yahoo_direct(ticker_symbol)
        if hist.empty or len(hist) < 2:
            try:
                hist = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).history(period="120d", interval="1d"))
                if not hist.empty:
                    hist.index = pd.to_datetime(hist.index).date
                    source = "yfinance 官方库"
            except Exception:
                hist = pd.DataFrame()
        if hist.empty or len(hist) < 2:
            hist, source = fetch_chart_stooq_csv(ticker_symbol)
        if hist.empty or len(hist) < 2:
            return None

        macro = fetch_macro_bundle()
        float_shares, shares_out = fetch_share_stats(ticker_symbol)

        hist["昨收"] = hist["Close"].shift(1)
        hist["MA5"] = hist["Close"].rolling(5).mean()
        hist["MA20"] = hist["Close"].rolling(20).mean()
        std20 = hist["Close"].rolling(20).std().fillna(0)
        hist["Upper"], hist["Lower"] = hist["MA20"] + std20 * 2, hist["MA20"] - std20 * 2

        tp = (hist["High"] + hist["Low"] + hist["Close"]) / 3
        rmf = tp * hist["Volume"]
        pos_flow = pd.Series(np.where(tp > tp.shift(1), rmf, 0), index=hist.index)
        neg_flow = pd.Series(np.where(tp < tp.shift(1), rmf, 0), index=hist.index)
        mfr = pos_flow.rolling(14, min_periods=1).sum() / neg_flow.rolling(14, min_periods=1).sum().replace(0, 1)
        hist["MFI"] = (100 - (100 / (1 + mfr))).fillna(50)

        scenario_model = build_scenario_model(hist)
        live_price, live_volume = get_live_quote(ticker_symbol)
        vol_latest = max(float(live_volume or 0), float(hist["Volume"].iloc[-1] or 0))

        # 抓取 FINRA 真实场外/暗池数据并关联总成交量
        dp_df = fetch_finra_darkpool_data(ticker_symbol, lookback_days=7)
        if not dp_df.empty:
            total_vol_map = hist["Volume"].to_dict()
            dp_df["全市场总成交量"] = dp_df.index.map(lambda d: total_vol_map.get(d, np.nan))
            dp_df["暗池成交占比%"] = np.where(
                dp_df["全市场总成交量"] > 0,
                (dp_df["暗池/场外总量"] / dp_df["全市场总成交量"] * 100).round(2),
                np.nan,
            )

        return hist, scenario_model, dp_df, {
            **macro,
            "float_shares": float_shares,
            "shares_out": shares_out,
            "volume": vol_latest,
            "source": source,
            "fetch_time": fetch_time_bj,
            "timestamp": now_ts,
        }
    except Exception:
        return None

# ==========================================
# 🕯️ 8. 专业期权链与精准 GEX 计算引擎
# ==========================================
def fetch_yahoo_options_raw(ticker_symbol, selected_exp=None):
    session = cffi_requests.Session(impersonate="chrome110")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    url_params = ""
    if selected_exp:
        try:
            exp_ts = int(datetime.strptime(selected_exp, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
            url_params = f"?date={exp_ts}"
        except Exception:
            pass

    for domain in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        try:
            res = session.get(f"https://{domain}/v7/finance/options/{ticker_symbol}{url_params}", headers=headers, timeout=5)
            if res.status_code == 200:
                result = res.json().get("optionChain", {}).get("result", [])
                if result:
                    return result[0]
        except Exception:
            pass
        try:
            session.get("https://fc.yahoo.com", headers=headers, timeout=3)
            crumb = session.get(f"https://{domain}/v1/test/getcrumb", headers=headers, timeout=3).text.strip()
            if crumb and len(crumb) < 30 and "{" not in crumb:
                delim = "&" if url_params else "?"
                res_c = session.get(f"https://{domain}/v7/finance/options/{ticker_symbol}{url_params}{delim}crumb={crumb}", headers=headers, timeout=5)
                if res_c.status_code == 200:
                    result = res_c.json().get("optionChain", {}).get("result", [])
                    if result:
                        return result[0]
        except Exception:
            pass
    return None

def get_expiration_list(ticker_symbol):
    """带缓存的到期日列表获取，防止高频请求触发 429"""
    cached = GLOBAL_STORE["exp_list_cache"].get(ticker_symbol)
    if cached and (time.time() - cached["ts"] < 1800):
        return cached["list"]

    exp_list = []
    res_data = fetch_yahoo_options_raw(ticker_symbol)
    if res_data and "expirationDates" in res_data:
        exp_list = [datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") for ts in res_data["expirationDates"]]
    else:
        try:
            exp_list = _throttled_yahoo_call(lambda: list(yf.Ticker(ticker_symbol).options))
        except Exception:
            exp_list = []

    with GLOBAL_STORE["lock"]:
        GLOBAL_STORE["exp_list_cache"][ticker_symbol] = {"list": exp_list, "ts": time.time()}
    return exp_list

def do_fetch_option_details(ticker_symbol, expiration_date, fallback_current_price):
    calls_df, puts_df = pd.DataFrame(), pd.DataFrame()
    call_wall, put_wall, gamma_flip, pcr_oi, pcr_vol = np.nan, np.nan, np.nan, np.nan, np.nan
    atm_iv, move_1d, move_exp = np.nan, np.nan, np.nan
    calc_mode = "持仓量 (OI)"
    status_msg = None

    try:
        res_data = fetch_yahoo_options_raw(ticker_symbol, expiration_date)
        calls, puts = pd.DataFrame(), pd.DataFrame()
        if res_data and res_data.get("options"):
            opt = res_data["options"][0]
            calls = pd.DataFrame(opt.get("calls", []))
            puts = pd.DataFrame(opt.get("puts", []))

        if calls.empty and puts.empty:
            try:
                opt = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).option_chain(expiration_date))
                calls, puts = opt.calls.copy(), opt.puts.copy()
            except Exception:
                pass

        if calls.empty and puts.empty:
            return calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_oi, pcr_vol, "无数据", "该到期日暂无期权报价", atm_iv, move_1d, move_exp

        live_price, _ = get_live_quote(ticker_symbol)
        current_price = float(live_price) if (live_price and live_price > 0) else float(fallback_current_price)

        for df in (calls, puts):
            if not df.empty:
                for col in ["strike", "lastPrice", "bid", "ask", "openInterest", "volume", "impliedVolatility"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0
                df["iv_valid"] = (df["impliedVolatility"] > 0.01) & (df["impliedVolatility"] < 5.0)

        # 精确计算距美东收盘 (16:00 ET) 的剩余年化时间 T
        exp_close_et = datetime.strptime(expiration_date, "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=US_EASTERN_TZ)
        hours_to_exp = max((exp_close_et - datetime.now(US_EASTERN_TZ)).total_seconds() / 3600.0, 1.0)
        T = hours_to_exp / (365.0 * 24.0)

        # ATM IV 与预期波幅 (单日按 252 交易日，到期日按实际剩余时间 T)
        all_valid = pd.concat([calls[calls["iv_valid"]], puts[puts["iv_valid"]]], ignore_index=True)
        if not all_valid.empty:
            # 取最接近现价的 Call & Put 平均 IV
            atm_strikes = all_valid.iloc[(all_valid["strike"] - current_price).abs().argsort()[:2]]
            atm_iv = float(atm_strikes["impliedVolatility"].mean())
            move_1d = current_price * atm_iv * np.sqrt(1.0 / 252.0)
            move_exp = current_price * atm_iv * np.sqrt(T)

        # 过滤极端行权价
        calls = calls[(calls["strike"] >= current_price * 0.3) & (calls["strike"] <= current_price * 2.5)].copy()
        puts = puts[(puts["strike"] >= current_price * 0.3) & (puts["strike"] <= current_price * 2.5)].copy()

        tot_c_oi, tot_p_oi = calls["openInterest"].sum(), puts["openInterest"].sum()
        tot_c_vol, tot_p_vol = calls["volume"].sum(), puts["volume"].sum()

        pcr_oi = (tot_p_oi / tot_c_oi) if tot_c_oi > 0 else np.nan
        pcr_vol = (tot_p_vol / tot_c_vol) if tot_c_vol > 0 else np.nan

        if (tot_c_oi + tot_p_oi) > 0:
            w_col, calc_mode = "openInterest", "持仓量 (OI)"
        elif (tot_c_vol + tot_p_vol) > 0:
            w_col, calc_mode = "volume", "成交量 (Volume)"
        else:
            w_col, calc_mode = None, "无有效持仓/成交"

        if w_col:
            c_above = calls[(calls["strike"] >= current_price) & (calls[w_col] > 0)]
            if not c_above.empty:
                call_wall = float(c_above.loc[c_above[w_col].idxmax(), "strike"])
            p_below = puts[(puts["strike"] <= current_price) & (puts[w_col] > 0)]
            if not p_below.empty:
                put_wall = float(p_below.loc[p_below[w_col].idxmax(), "strike"])

            # 精准 Black-Scholes 净伽马敞口 (Net GEX) 过零点计算 (跳过 0 持仓合约)
            s_range = np.linspace(current_price * 0.75, current_price * 1.25, 100)
            valid_calls = calls[calls["iv_valid"] & (calls[w_col] > 0)]
            valid_puts = puts[puts["iv_valid"] & (puts[w_col] > 0)]

            net_gammas = []
            sqrt_T = np.sqrt(T)
            for s_test in s_range:
                tot_g = 0.0
                for df_leg, sign in ((valid_calls, 1.0), (valid_puts, -1.0)):
                    if df_leg.empty:
                        continue
                    k = df_leg["strike"].values
                    w = df_leg[w_col].values
                    iv = df_leg["impliedVolatility"].values
                    d1 = (np.log(s_test / k) + (RISK_FREE_RATE + 0.5 * iv ** 2) * T) / (iv * sqrt_T)
                    gamma = np.exp(-0.5 * d1 ** 2) / (s_test * iv * np.sqrt(2 * np.pi * T))
                    # 乘以 100 股/张及现价平方缩放为 Dollar GEX
                    tot_g += np.sum(sign * w * 100.0 * gamma * (s_test ** 2) * 0.01)
                net_gammas.append(tot_g)

            net_gammas = np.array(net_gammas)
            crossings = np.where(np.diff(np.sign(net_gammas)))[0]
            if len(crossings) > 0:
                # 选取最接近现价的翻转点
                best_idx = min(crossings, key=lambda idx: abs(s_range[idx] - current_price))
                y1, y2 = net_gammas[best_idx], net_gammas[best_idx + 1]
                x1, x2 = s_range[best_idx], s_range[best_idx + 1]
                gamma_flip = float(x1 - y1 * (x2 - x1) / (y2 - y1)) if (y2 - y1) != 0 else float(x1)

        # 截取平值附近 9 档合约展示
        for df_type, target_df in [("calls", calls), ("puts", puts)]:
            if not target_df.empty:
                target_df = target_df.sort_values("strike").reset_index(drop=True)
                idx = (target_df["strike"] - current_price).abs().idxmin()
                slice_df = target_df.iloc[max(0, idx - 4): min(len(target_df), idx + 5)]
                if df_type == "calls":
                    calls_df = slice_df
                else:
                    puts_df = slice_df

    except Exception as e:
        status_msg = f"期权数据获取失败: {str(e)}"
        calc_mode = "接口异常"

    return calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_oi, pcr_vol, calc_mode, status_msg, atm_iv, move_1d, move_exp

# ==========================================
# 🔄 9. 线程安全的后台守护进程
# ==========================================
def background_updater_loop():
    while True:
        try:
            with GLOBAL_STORE["lock"]:
                tickers_to_process = list(GLOBAL_STORE["active_queue"])
            for ticker in tickers_to_process:
                try:
                    data = do_fetch_stock_data(ticker)
                    if data:
                        with GLOBAL_STORE["lock"]:
                            GLOBAL_STORE["stock_cache"][f"v10_{ticker}"] = data
                            GLOBAL_STORE["fetch_timestamps"][f"v10_{ticker}"] = data[3]["timestamp"]
                except Exception:
                    pass
                time.sleep(2.5)
        except Exception:
            pass
        time.sleep(get_dynamic_ttl())

@st.cache_resource
def start_background_engine():
    t = threading.Thread(target=background_updater_loop, daemon=True)
    t.start()
    return True

start_background_engine()

# ==========================================
# 🖥️ 10. 前端 UI 渲染层
# ==========================================
st.markdown("""<style> .main { background-color: #FFFFFF !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; } </style>""", unsafe_allow_html=True)

if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = "NVDA"

with st.sidebar:
    st.header("🔍 全美股检索")
    # 1. 支持直接输入任意美股/ETF代码
    direct_input = st.text_input(
        "直接输入任意代码 (回车生效)",
        placeholder="如: PLTR, MSTR, SOXL, BRK-B",
    ).strip().upper()

    if direct_input and direct_input != st.session_state.current_ticker:
        st.session_state.current_ticker = direct_input
        with GLOBAL_STORE["lock"]:
            GLOBAL_STORE["active_queue"].add(direct_input)
        st.rerun()

    # 2. 全量下拉搜索框 (10,000+ 美股及 ETF)
    default_idx = 0
    for i, item in enumerate(STOCK_LIBRARY):
        if item.startswith(st.session_state.current_ticker + " -"):
            default_idx = i
            break

    selected_item = st.selectbox(
        f"或从全美股库搜索 (共 {len(STOCK_LIBRARY):,} 只)",
        options=STOCK_LIBRARY,
        index=default_idx,
    )
    new_tk = selected_item.split(" - ")[0].strip()
    if new_tk and new_tk != st.session_state.current_ticker and not direct_input:
        st.session_state.current_ticker = new_tk
        with GLOBAL_STORE["lock"]:
            GLOBAL_STORE["active_queue"].add(new_tk)
        st.rerun()

    st.divider()
    st.subheader("📐 换手率口径设置")
    float_mode = st.radio("默认分母口径", ["流通股本 (Float)", "总股本 (Outstanding)"], index=0)
    manual_float = st.number_input(
        f"{st.session_state.current_ticker} 手动修正股本 (留0自动)",
        min_value=0, value=0, step=1000000,
    )

    if st.button("🔄 强制刷新当前标的"):
        with GLOBAL_STORE["lock"]:
            GLOBAL_STORE["stock_cache"].pop(f"v10_{st.session_state.current_ticker}", None)
            GLOBAL_STORE["darkpool_cache"].pop(f"dp_{st.session_state.current_ticker}", None)
        st.rerun()

ticker = st.session_state.current_ticker
st.title(f"🎯 {ticker} 专业量化决策终端 Pro")

stock_data = GLOBAL_STORE["stock_cache"].get(f"v10_{ticker}")
last_ts = GLOBAL_STORE["fetch_timestamps"].get(f"v10_{ticker}", 0)
is_expired = (time.time() - last_ts) > get_dynamic_ttl()

if not stock_data or is_expired:
    with st.spinner(f"🔄 正在同步 **{ticker}** 行情与 FINRA 暗池数据..."):
        fresh_data = do_fetch_stock_data(ticker)
        if fresh_data:
            stock_data = fresh_data
            with GLOBAL_STORE["lock"]:
                GLOBAL_STORE["stock_cache"][f"v10_{ticker}"] = fresh_data
                GLOBAL_STORE["fetch_timestamps"][f"v10_{ticker}"] = fresh_data[3]["timestamp"]
                GLOBAL_STORE["active_queue"].add(ticker)

if not stock_data:
    st.error(f"⚠️ 无法拉取 **{ticker}** 的数据，请检查代码是否正确（美股含点号的代码如 BRK.B 请写为 BRK-B）。")
else:
    hist_df, reg, dp_df, mkt = stock_data
    last = hist_df.iloc[-1]

    # 动态确定换手率分母 (修改侧边栏立即生效)
    if manual_float > 0:
        eff_float, float_label = float(manual_float), "手动修正"
    elif float_mode == "流通股本 (Float)" and mkt["float_shares"]:
        eff_float, float_label = mkt["float_shares"], "流通股(Float)"
    elif mkt["shares_out"]:
        eff_float, float_label = mkt["shares_out"], "总股本(Outstanding)"
    elif mkt["float_shares"]:
        eff_float, float_label = mkt["float_shares"], "流通股(Float)"
    else:
        eff_float, float_label = None, "无股本数据"

    rt_turnover, proj_turnover, trade_status = calculate_turnover_metrics(mkt["volume"], eff_float, mkt["timestamp"])
    hist_df = hist_df.copy()
    hist_df["换手率_raw"] = (hist_df["Volume"] / eff_float) if (eff_float and eff_float > 0) else np.nan

    st.caption(f"🟢 **{ticker}** 数据抓取于 (北京时间): **{mkt['fetch_time']}** | 行情源: **{mkt['source']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bitcoin", f"${mkt['btc']:,.0f}" if mkt["btc"] > 0 else "N/A")
    m2.metric("Nasdaq", f"{mkt['nasdaq']:,.2f}" if mkt["nasdaq"] > 0 else "N/A", f"{mkt['nasdaq_pct']:.2%}" if mkt["nasdaq"] > 0 else None)
    m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}" if mkt["vix"] > 0 else "N/A", f"{mkt['vix_pct']:.2%}" if mkt["vix"] > 0 else None, delta_color="inverse")
    m4.metric(f"{ticker} 现价", f"${last['Close']:.2f}", f"{(last['Close']/last['昨收']-1):.2%}" if pd.notnull(last["昨收"]) else None)

    st.divider()

    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时指标与换手率")
        if eff_float and eff_float > 0:
            st.write(f"实时换手 ({float_label}): **{rt_turnover:.2f}%**")
            st.write(f"预估全天换手: **{proj_turnover:.2f}%** (`{trade_status}`)")
            st.caption(f"分母股本: {eff_float:,.0f} 股 | 成交量: {mkt['volume']:,.0f} 股")
        else:
            st.write("实时换手: **N/A (ETF或无股本数据)**")
        st.write(f"BOLL 上/下轨: **{last['Upper']:.2f} / {last['Lower']:.2f}**")
        st.write(f"资金流量 MFI(14): **{last['MFI']:.2f}**")

    with c2:
        st.subheader("📍 场景回归预测")
        rows, mode = predict_scenarios(reg, hist_df)
        if rows:
            st.caption("模型: 分位数回归 (10/50/90分位)" if mode == "quantile" else "模型: 线性回归兜底")
            st.table(pd.DataFrame({
                "场景": [r[0] for r in rows],
                "压力参考": [r[1] for r in rows],
                "支撑参考": [r[2] for r in rows],
            }).style.format(precision=2))
        else:
            st.info("历史样本不足，暂无法生成场景预测。")

    st.divider()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(45).copy()
    p_df["label"] = pd.to_datetime(p_df.index).strftime("%m-%d")

    fig.add_trace(go.Scatter(x=p_df["label"], y=p_df["Upper"], line=dict(color="rgba(0,102,204,0.3)"), name="BOLL上轨"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df["label"], y=p_df["Lower"], line=dict(color="rgba(0,102,204,0.3)"), fill="tonexty", name="BOLL下轨"), row=1, col=1)
    fig.add_trace(go.Candlestick(x=p_df["label"], open=p_df["Open"], high=p_df["High"], low=p_df["Low"], close=p_df["Close"], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df["label"], y=p_df["MA5"], line=dict(color="#FF9800"), name="MA5"), row=1, col=1)

    colors = ["#E53935" if p_df["Close"].iloc[i] >= p_df["Open"].iloc[i] else "#43A047" for i in range(len(p_df))]
    if p_df["换手率_raw"].notnull().any():
        fig.add_trace(go.Bar(x=p_df["label"], y=p_df["换手率_raw"] * 100, marker_color=colors, name="换手率%"), row=2, col=1)
    else:
        fig.add_trace(go.Bar(x=p_df["label"], y=p_df["Volume"] / 10000, marker_color=colors, name="成交量(万股)"), row=2, col=1)

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type="category", tickmode="linear", dtick=1, tickangle=-90)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    o_col, d_col = st.columns([1.5, 1])

    with o_col:
        st.subheader("🕯️ 全景期权与 GEX 分析")
        exp_list = get_expiration_list(ticker)
        if exp_list:
            today_str = datetime.now(US_EASTERN_TZ).strftime("%Y-%m-%d")
            default_exp_idx = next((i for i, d in enumerate(exp_list) if d >= today_str), 0)
            selected_exp = st.selectbox("📅 选择期权到期日", options=exp_list, index=default_exp_idx)

            opt_key = f"v10_{ticker}_{selected_exp}"
            opt_data = GLOBAL_STORE["options_cache"].get(opt_key)
            if not opt_data or is_expired:
                opt_data = do_fetch_option_details(ticker, selected_exp, last["Close"])
                with GLOBAL_STORE["lock"]:
                    GLOBAL_STORE["options_cache"][opt_key] = opt_data

            calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_oi, pcr_vol, calc_mode, status_msg, atm_iv, move_1d, move_exp = opt_data
            if status_msg:
                st.info(f"ℹ️ {status_msg}")

            iv_txt = f"{atm_iv:.2%}" if pd.notnull(atm_iv) else "N/A"
            st.caption(f"权重维度: **{calc_mode}** | ATM 隐含波动率: **{iv_txt}** | PCR(成交量): **{pcr_vol:.2f}**" if pd.notnull(pcr_vol) else f"权重维度: **{calc_mode}** | ATM IV: **{iv_txt}**")

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("🧱 Call Wall", f"${call_wall:.2f}" if pd.notnull(call_wall) else "N/A")
            q2.metric("🧱 Put Wall", f"${put_wall:.2f}" if pd.notnull(put_wall) else "N/A")
            q3.metric("🌀 Gamma Flip", f"${gamma_flip:.2f}" if pd.notnull(gamma_flip) else "N/A")
            q4.metric("📊 PCR (持仓)", f"{pcr_oi:.2f}" if pd.notnull(pcr_oi) else "N/A")

            if pd.notnull(move_1d) and pd.notnull(move_exp):
                em1, em2 = st.columns(2)
                em1.info(f"🎯 单交易日预期波幅: ±${move_1d:.2f} (${last['Close']-move_1d:.2f} ~ ${last['Close']+move_1d:.2f})")
                em2.success(f"📅 至 {selected_exp} 预期波幅: ±${move_exp:.2f} (${last['Close']-move_exp:.2f} ~ ${last['Close']+move_exp:.2f})")

            t1, t2 = st.tabs(["📈 看涨 (Calls)", "📉 看跌 (Puts)"])
            with t1:
                if not calls_df.empty:
                    st.dataframe(calls_df[["strike", "lastPrice", "volume", "openInterest", "impliedVolatility"]].style.format(
                        {"impliedVolatility": "{:.2%}", "lastPrice": "{:.2f}", "strike": "{:.2f}", "openInterest": "{:,.0f}", "volume": "{:,.0f}"}
                    ), use_container_width=True)
            with t2:
                if not puts_df.empty:
                    st.dataframe(puts_df[["strike", "lastPrice", "volume", "openInterest", "impliedVolatility"]].style.format(
                        {"impliedVolatility": "{:.2%}", "lastPrice": "{:.2f}", "strike": "{:.2f}", "openInterest": "{:,.0f}", "volume": "{:,.0f}"}
                    ), use_container_width=True)
        else:
            st.info("💡 当前标的暂无期权链，或接口暂时受限。")

    with d_col:
        st.subheader("🌑 FINRA 暗池 / 场外数据")
        st.caption("数据来源: FINRA Reg SHO 官方每日场外 (TRF/ADF) 统计")
        if not dp_df.empty:
            show_cols = ["暗池/场外总量", "暗池成交占比%", "暗池做空占比%"]
            st.dataframe(
                dp_df[show_cols].style.format({
                    "暗池/场外总量": "{:,.0f}",
                    "暗池成交占比%": "{:.2f}%",
                    "暗池做空占比%": "{:.2f}%",
                }, na_rep="N/A"),
                use_container_width=True,
            )
            latest_sr = dp_df["暗池做空占比%"].iloc[0]
            if latest_sr > 55:
                st.warning(f"⚠️ 最新交易日暗池做空占比偏高 (**{latest_sr:.1f}%**)，需警惕机构派发或对冲压制。")
            elif latest_sr < 40:
                st.success(f"🟢 最新交易日暗池做空占比偏低 (**{latest_sr:.1f}%**)，场外买盘承接较强。")
        else:
            st.info("暂无该标的近期的 FINRA 场外记录（非美股 NMS 标的或数据尚未公布）。")

    st.subheader("📋 近期历史明细")
    hist_show = hist_df.tail(15).sort_index(ascending=False).copy()
    hist_show["换手%"] = (hist_show["换手率_raw"] * 100).map("{:.2f}%".format) if hist_show["换手率_raw"].notnull().any() else "N/A"
    st.dataframe(hist_show[["Open", "High", "Low", "Close", "Volume", "换手%", "MFI", "MA5", "MA20"]].style.format(
        {"Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}", "Volume": "{:,.0f}", "MFI": "{:.2f}", "MA5": "{:.2f}", "MA20": "{:.2f}"}
    ), use_container_width=True)
