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
from concurrent.futures import ThreadPoolExecutor
import requests
from io import StringIO
from curl_cffi import requests as cffi_requests

# ==========================================
# 0. 页面全局配置与时区定义
# ==========================================
st.set_page_config(layout="wide", page_title="专业量化决策终端 Pro")

BEIJING_TZ = ZoneInfo("Asia/Shanghai")
US_EASTERN_TZ = ZoneInfo("America/New_York")
GLOBAL_RATE_INTERVAL = 0.8
RISK_FREE_RATE = 0.045

# ==========================================
# 🧠 1. 全局解耦内存存储中心
# ==========================================
@st.cache_resource
def get_global_data_store():
    return {
        "stock_cache": {},
        "intraday_cache": {},
        "options_cache": {},
        "exp_list_cache": {},
        "darkpool_cache": {},
        "macro_cache": {"data": None, "ts": 0.0},
        "fetch_timestamps": {},
        "active_queue": set(["NVDA", "AAPL", "TSLA", "BTDR"]),
        "lock": threading.Lock(),
        "last_yahoo_call_ts": 0.0,
    }

GLOBAL_STORE = get_global_data_store()

def get_dynamic_ttl():
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
# 🔍 2. 全美股 + ETF 动态股票库
# ==========================================
@st.cache_data(ttl=86400)
def load_us_stock_library():
    stock_dict = {
        "NVDA": "NVIDIA Corporation",
        "AAPL": "Apple Inc.",
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

    try:
        sec_headers = {"User-Agent": "QuantTerminalPro/2.0 (research@quantterminal.local)"}
        res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=sec_headers, timeout=6)
        if res.status_code == 200:
            for item in res.json().values():
                sym = str(item.get("ticker", "")).strip().upper()
                title = str(item.get("title", "")).strip().title()
                if sym and sym not in stock_dict:
                    stock_dict[sym] = title[:45]
    except Exception:
        pass

    return [f"{k} - {v}" for k, v in stock_dict.items()]

STOCK_LIBRARY = load_us_stock_library()

# ==========================================
# 📐 3. 基础信息 (股本 + 财报日) 与换手率
# ==========================================
def fetch_fundamental_stats(ticker_symbol):
    """一次请求同时获取流通股、总股本与下次财报日期"""
    info = {}
    try:
        info = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).info)
    except Exception:
        info = {}

    float_shares = info.get("floatShares") if info else None
    shares_out = info.get("sharesOutstanding") if info else None
    float_shares = float(float_shares) if float_shares and float_shares > 0 else None
    shares_out = float(shares_out) if shares_out and shares_out > 0 else None

    earnings_date = None
    if info:
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if ts and isinstance(ts, (int, float)) and ts > 0:
            ed = datetime.fromtimestamp(ts, tz=US_EASTERN_TZ).date()
            if ed >= datetime.now(US_EASTERN_TZ).date():
                earnings_date = ed

    return float_shares, shares_out, earnings_date

def calculate_turnover_metrics(volume, capital, timestamp_unix):
    if not capital or capital <= 0 or not volume or volume <= 0:
        return 0.0, 0.0, "数据不足"

    realtime_turnover = (volume / capital) * 100.0
    dt_et = datetime.fromtimestamp(timestamp_unix, tz=US_EASTERN_TZ)

    weekday = dt_et.weekday()
    time_min = dt_et.hour * 60 + dt_et.minute
    open_min, close_min, total_trade_min = 9 * 60 + 30, 16 * 60, 390

    if weekday < 5 and open_min <= time_min < close_min:
        elapsed = time_min - open_min
        if elapsed < 30:
            return realtime_turnover, realtime_turnover, f"盘中(开盘{elapsed}分, 暂不外推)"
        projected_vol = volume * (total_trade_min / elapsed)
        return realtime_turnover, (projected_vol / capital) * 100.0, f"盘中外推(已交易{elapsed}分)"
    elif weekday < 5 and time_min < open_min:
        return realtime_turnover, realtime_turnover, "盘前/未开盘"
    return realtime_turnover, realtime_turnover, "已收盘"

# ==========================================
# 🌑 4. 多线程并发 FINRA 暗池数据引擎
# ==========================================
def fetch_finra_darkpool_data(ticker_symbol, lookback_days=7):
    cache_key = f"dp_{ticker_symbol}"
    cached = GLOBAL_STORE["darkpool_cache"].get(cache_key)
    if cached and (time.time() - cached["ts"] < 3600):
        return cached["df"]

    today_et = datetime.now(US_EASTERN_TZ).date()
    candidate_dates = []
    delta = 0
    while len(candidate_dates) < 12:
        d = today_et - timedelta(days=delta)
        delta += 1
        if d.weekday() < 5:
            candidate_dates.append(d)

    sym_upper = ticker_symbol.upper()

    def _fetch_one_day(d):
        date_str = d.strftime("%Y%m%d")
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date_str}.txt"
        try:
            res = requests.get(url, timeout=4)
            if res.status_code == 200:
                target_prefix = f"{date_str}|{sym_upper}|"
                for line in res.text.splitlines():
                    if line.startswith(target_prefix):
                        parts = line.split("|")
                        if len(parts) >= 5:
                            short_vol = float(parts[2])
                            total_off_ex = float(parts[4])
                            short_ratio = (short_vol / total_off_ex * 100) if total_off_ex > 0 else 0.0
                            return {
                                "日期": d,
                                "暗池/场外总量": int(total_off_ex),
                                "暗池做空量": int(short_vol),
                                "暗池做空占比%": round(short_ratio, 2),
                            }
                        break
        except Exception:
            pass
        return None

    records = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for res in executor.map(_fetch_one_day, candidate_dates):
            if res:
                records.append(res)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("日期").sort_index(ascending=False).head(lookback_days)

    with GLOBAL_STORE["lock"]:
        GLOBAL_STORE["darkpool_cache"][cache_key] = {"df": df, "ts": time.time()}
    return df

# ==========================================
# 🚀 5. K 线 (日线 + 分时 VWAP) 与宏观引擎
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

def fetch_intraday_chart(symbol, interval="15m"):
    """抓取 15m 或 60m 分时 K 线并计算按日重置的 VWAP"""
    cache_key = f"{symbol}_{interval}"
    cached = GLOBAL_STORE["intraday_cache"].get(cache_key)
    if cached and (time.time() - cached["ts"] < 180):
        return cached["df"]

    rng = "5d" if interval == "15m" else "1mo"
    try:
        session = cffi_requests.Session(impersonate="chrome110")
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range={rng}&interval={interval}&includePrePost=false"
        res = session.get(url, timeout=5)
        if res.status_code == 200:
            result = res.json()["chart"]["result"][0]
            timestamps = result.get("timestamp", [])
            quote = result["indicators"]["quote"][0]
            dt_index = [datetime.fromtimestamp(ts, tz=US_EASTERN_TZ) for ts in timestamps]
            df = pd.DataFrame({
                "Open": quote.get("open", []),
                "High": quote.get("high", []),
                "Low": quote.get("low", []),
                "Close": quote.get("close", []),
                "Volume": quote.get("volume", []),
            }, index=dt_index)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Close"]).copy()
            df["Volume"] = df["Volume"].fillna(0.0)

            # 按美东交易日重置计算 VWAP
            df["trade_date"] = [ts.date() for ts in df.index]
            tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
            df["tp_vol"] = tp * df["Volume"]
            cum_tp_vol = df.groupby("trade_date")["tp_vol"].cumsum()
            cum_vol = df.groupby("trade_date")["Volume"].cumsum().replace(0, np.nan)
            df["VWAP"] = (cum_tp_vol / cum_vol).ffill()
            df["MA20"] = df["Close"].rolling(20).mean()

            with GLOBAL_STORE["lock"]:
                GLOBAL_STORE["intraday_cache"][cache_key] = {"df": df, "ts": time.time()}
            return df
    except Exception:
        pass
    return pd.DataFrame()

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
    mc = GLOBAL_STORE["macro_cache"]
    if mc["data"] and (time.time() - mc["ts"] < 180):
        return mc["data"]

    def _one(symbol):
        try:
            session = cffi_requests.Session(impersonate="chrome110")
            res = session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d", timeout=3)
            if res.status_code == 200:
                meta = res.json()["chart"]["result"][0]["meta"]
                price = meta.get("regularMarketPrice", 0.0)
                prev = meta.get("chartPreviousClose", meta.get("previousClose", price))
                if price > 0:
                    return price, ((price / prev - 1) if prev else 0.0)
        except Exception:
            pass
        return 0.0, 0.0

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_btc = ex.submit(_one, "BTC-USD")
        f_ixic = ex.submit(_one, "^IXIC")
        f_vix = ex.submit(_one, "^VIX")
        btc, _ = f_btc.result()
        nasdaq, nasdaq_pct = f_ixic.result()
        vix, vix_pct = f_vix.result()

    bundle = {"btc": btc, "nasdaq": nasdaq, "nasdaq_pct": nasdaq_pct, "vix": vix, "vix_pct": vix_pct}
    with GLOBAL_STORE["lock"]:
        GLOBAL_STORE["macro_cache"] = {"data": bundle, "ts": time.time()}
    return bundle

def get_live_quote(ticker_symbol):
    try:
        fi = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).fast_info)
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
    fallback = {}
    for tag, y in [("h", y_h), ("l", y_l)]:
        m = LinearRegression().fit(fit_df[["gap"]].values, y)
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
# 🎯 7. 核心数据聚合
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
        float_shares, shares_out, earnings_date = fetch_fundamental_stats(ticker_symbol)

        hist["昨收"] = hist["Close"].shift(1)
        hist["MA5"] = hist["Close"].rolling(5).mean()
        hist["MA20"] = hist["Close"].rolling(20).mean()
        std20 = hist["Close"].rolling(20).std().fillna(0)
        hist["Upper"], hist["Lower"] = hist["MA20"] + std20 * 2, hist["MA20"] - std20 * 2

        # 20日年化历史波动率 (HV20)
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1))
        hv20 = float(log_ret.tail(20).std() * np.sqrt(252)) if len(log_ret.dropna()) >= 10 else np.nan

        tp = (hist["High"] + hist["Low"] + hist["Close"]) / 3
        rmf = tp * hist["Volume"]
        pos_flow = pd.Series(np.where(tp > tp.shift(1), rmf, 0), index=hist.index)
        neg_flow = pd.Series(np.where(tp < tp.shift(1), rmf, 0), index=hist.index)
        mfr = pos_flow.rolling(14, min_periods=1).sum() / neg_flow.rolling(14, min_periods=1).sum().replace(0, 1)
        hist["MFI"] = (100 - (100 / (1 + mfr))).fillna(50)

        scenario_model = build_scenario_model(hist)
        live_price, live_volume = get_live_quote(ticker_symbol)
        vol_latest = max(float(live_volume or 0), float(hist["Volume"].iloc[-1] or 0))

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
            "earnings_date": earnings_date,
            "hv20": hv20,
            "volume": vol_latest,
            "source": source,
            "fetch_time": fetch_time_bj,
            "timestamp": now_ts,
        }
    except Exception:
        return None

# ==========================================
# 🕯️ 8. 期权链、多到期日汇总 GEX 与 Max Pain 引擎
# ==========================================
def fetch_yahoo_options_raw(ticker_symbol, selected_exp=None):
    session = cffi_requests.Session(impersonate="chrome110")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
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

def _fetch_single_exp_chain(ticker_symbol, exp_str):
    res_data = fetch_yahoo_options_raw(ticker_symbol, exp_str)
    calls, puts = pd.DataFrame(), pd.DataFrame()
    if res_data and res_data.get("options"):
        opt = res_data["options"][0]
        calls = pd.DataFrame(opt.get("calls", []))
        puts = pd.DataFrame(opt.get("puts", []))
    if calls.empty and puts.empty:
        try:
            opt = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).option_chain(exp_str))
            calls, puts = opt.calls.copy(), opt.puts.copy()
        except Exception:
            pass
    return calls, puts

def calculate_max_pain(calls, puts, w_col="openInterest"):
    """计算最大痛点 (Max Pain)：使买方到期总收益最小的行权价"""
    if calls.empty and puts.empty:
        return np.nan
    strikes = np.sort(np.unique(np.concatenate([
        calls["strike"].values if not calls.empty else np.array([]),
        puts["strike"].values if not puts.empty else np.array([]),
    ])))
    if len(strikes) == 0:
        return np.nan

    c_k = calls["strike"].values if not calls.empty else np.array([])
    c_w = calls[w_col].values if not calls.empty else np.array([])
    p_k = puts["strike"].values if not puts.empty else np.array([])
    p_w = puts[w_col].values if not puts.empty else np.array([])

    min_pain = float("inf")
    best_strike = np.nan
    for s in strikes:
        c_pain = np.sum(np.maximum(0.0, s - c_k) * c_w) if len(c_k) > 0 else 0.0
        p_pain = np.sum(np.maximum(0.0, p_k - s) * p_w) if len(p_k) > 0 else 0.0
        total_pain = c_pain + p_pain
        if total_pain < min_pain:
            min_pain = total_pain
            best_strike = float(s)
    return best_strike

def do_fetch_option_details(ticker_symbol, exp_list_target, fallback_current_price):
    """支持单到期日或近 N 期汇总 GEX、Walls、Gamma Flip 与 Max Pain"""
    calls_disp, puts_disp, gex_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    call_wall, put_wall, gamma_flip, max_pain = np.nan, np.nan, np.nan, np.nan
    pcr_oi, pcr_vol, atm_iv, move_1d, move_exp = np.nan, np.nan, np.nan, np.nan, np.nan
    calc_mode, status_msg = "持仓量 (OI)", None

    try:
        live_price, _ = get_live_quote(ticker_symbol)
        current_price = float(live_price) if (live_price and live_price > 0) else float(fallback_current_price)

        all_calls_list, all_puts_list = [], []
        now_et = datetime.now(US_EASTERN_TZ)

        for idx, exp_str in enumerate(exp_list_target):
            c_raw, p_raw = _fetch_single_exp_chain(ticker_symbol, exp_str)
            exp_close_et = datetime.strptime(exp_str, "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=US_EASTERN_TZ)
            hours_to_exp = max((exp_close_et - now_et).total_seconds() / 3600.0, 1.0)
            T = hours_to_exp / (365.0 * 24.0)

            for df, bucket in ((c_raw, all_calls_list), (p_raw, all_puts_list)):
                if not df.empty:
                    for col in ["strike", "lastPrice", "openInterest", "volume", "impliedVolatility"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0
                    df["iv_valid"] = (df["impliedVolatility"] > 0.01) & (df["impliedVolatility"] < 5.0)
                    df["T"] = T
                    df["exp"] = exp_str
                    df = df[(df["strike"] >= current_price * 0.3) & (df["strike"] <= current_price * 2.5)].copy()
                    bucket.append(df)

            # 第一个到期日用于 ATM IV、预期波幅与明细展示
            if idx == 0 and (not c_raw.empty or not p_raw.empty):
                c0 = all_calls_list[0] if all_calls_list else pd.DataFrame()
                p0 = all_puts_list[0] if all_puts_list else pd.DataFrame()
                valid0 = pd.concat([c0[c0["iv_valid"]] if not c0.empty else pd.DataFrame(),
                                    p0[p0["iv_valid"]] if not p0.empty else pd.DataFrame()], ignore_index=True)
                if not valid0.empty:
                    atm_rows = valid0.iloc[(valid0["strike"] - current_price).abs().argsort()[:2]]
                    atm_iv = float(atm_rows["impliedVolatility"].mean())
                    move_1d = current_price * atm_iv * np.sqrt(1.0 / 252.0)
                    move_exp = current_price * atm_iv * np.sqrt(T)

                for df_type, target_df in [("calls", c0), ("puts", p0)]:
                    if not target_df.empty:
                        t_sorted = target_df.sort_values("strike").reset_index(drop=True)
                        i_atm = (t_sorted["strike"] - current_price).abs().idxmin()
                        sl = t_sorted.iloc[max(0, i_atm - 4): min(len(t_sorted), i_atm + 5)]
                        if df_type == "calls":
                            calls_disp = sl
                        else:
                            puts_disp = sl

        calls = pd.concat(all_calls_list, ignore_index=True) if all_calls_list else pd.DataFrame()
        puts = pd.concat(all_puts_list, ignore_index=True) if all_puts_list else pd.DataFrame()

        if calls.empty and puts.empty:
            return calls_disp, puts_disp, call_wall, put_wall, gamma_flip, max_pain, pcr_oi, pcr_vol, "无数据", "未获取到有效期权报价", atm_iv, move_1d, move_exp, gex_df

        tot_c_oi = calls["openInterest"].sum() if not calls.empty else 0.0
        tot_p_oi = puts["openInterest"].sum() if not puts.empty else 0.0
        tot_c_vol = calls["volume"].sum() if not calls.empty else 0.0
        tot_p_vol = puts["volume"].sum() if not puts.empty else 0.0

        pcr_oi = (tot_p_oi / tot_c_oi) if tot_c_oi > 0 else np.nan
        pcr_vol = (tot_p_vol / tot_c_vol) if tot_c_vol > 0 else np.nan

        if (tot_c_oi + tot_p_oi) > 0:
            w_col, calc_mode = "openInterest", "持仓量 (OI)"
        elif (tot_c_vol + tot_p_vol) > 0:
            w_col, calc_mode = "volume", "成交量 (Volume)"
        else:
            w_col, calc_mode = None, "无有效持仓/成交"

        if w_col:
            # 按行权价汇总求 Walls 与 Max Pain
            c_grp = calls.groupby("strike", as_index=False)[w_col].sum() if not calls.empty else pd.DataFrame(columns=["strike", w_col])
            p_grp = puts.groupby("strike", as_index=False)[w_col].sum() if not puts.empty else pd.DataFrame(columns=["strike", w_col])

            c_above = c_grp[(c_grp["strike"] >= current_price) & (c_grp[w_col] > 0)]
            if not c_above.empty:
                call_wall = float(c_above.loc[c_above[w_col].idxmax(), "strike"])
            p_below = p_grp[(p_grp["strike"] <= current_price) & (p_grp[w_col] > 0)]
            if not p_below.empty:
                put_wall = float(p_below.loc[p_below[w_col].idxmax(), "strike"])

            max_pain = calculate_max_pain(c_grp, p_grp, w_col)

            # 计算各行权价汇总 Dollar GEX ($M)
            valid_calls = calls[calls["iv_valid"] & (calls[w_col] > 0)].copy() if not calls.empty else pd.DataFrame()
            valid_puts = puts[puts["iv_valid"] & (puts[w_col] > 0)].copy() if not puts.empty else pd.DataFrame()

            def _calc_strike_gex(df_leg, sign):
                if df_leg.empty:
                    return pd.DataFrame(columns=["strike", "gex"])
                k = df_leg["strike"].values
                w = df_leg[w_col].values
                iv = df_leg["impliedVolatility"].values
                t_arr = df_leg["T"].values
                sqrt_t = np.sqrt(t_arr)
                d1 = (np.log(current_price / k) + (RISK_FREE_RATE + 0.5 * iv ** 2) * t_arr) / (iv * sqrt_t)
                gamma = np.exp(-0.5 * d1 ** 2) / (current_price * iv * np.sqrt(2 * np.pi * t_arr))
                gex_m = sign * w * 100.0 * gamma * (current_price ** 2) * 0.01 / 1e6
                return pd.DataFrame({"strike": k, "gex": gex_m}).groupby("strike", as_index=False)["gex"].sum()

            c_gex = _calc_strike_gex(valid_calls, 1.0).rename(columns={"gex": "call_gex"})
            p_gex = _calc_strike_gex(valid_puts, -1.0).rename(columns={"gex": "put_gex"})
            gex_merged = pd.merge(c_gex, p_gex, on="strike", how="outer").fillna(0.0)
            gex_merged["net_gex"] = gex_merged["call_gex"] + gex_merged["put_gex"]
            gex_df = gex_merged[
                (gex_merged["strike"] >= current_price * 0.80) & (gex_merged["strike"] <= current_price * 1.20)
            ].sort_values("strike").reset_index(drop=True)
            if gex_df.empty:
                gex_df = gex_merged.sort_values("strike").reset_index(drop=True)

            # 计算 Gamma Flip
            s_range = np.linspace(current_price * 0.75, current_price * 1.25, 90)
            net_gammas = []
            for s_test in s_range:
                tot_g = 0.0
                for df_leg, sign in ((valid_calls, 1.0), (valid_puts, -1.0)):
                    if df_leg.empty:
                        continue
                    k = df_leg["strike"].values
                    w = df_leg[w_col].values
                    iv = df_leg["impliedVolatility"].values
                    t_arr = df_leg["T"].values
                    d1 = (np.log(s_test / k) + (RISK_FREE_RATE + 0.5 * iv ** 2) * t_arr) / (iv * np.sqrt(t_arr))
                    gamma = np.exp(-0.5 * d1 ** 2) / (s_test * iv * np.sqrt(2 * np.pi * t_arr))
                    tot_g += np.sum(sign * w * 100.0 * gamma * (s_test ** 2) * 0.01)
                net_gammas.append(tot_g)

            net_gammas = np.array(net_gammas)
            crossings = np.where(np.diff(np.sign(net_gammas)))[0]
            if len(crossings) > 0:
                best_idx = min(crossings, key=lambda i: abs(s_range[i] - current_price))
                y1, y2 = net_gammas[best_idx], net_gammas[best_idx + 1]
                x1, x2 = s_range[best_idx], s_range[best_idx + 1]
                gamma_flip = float(x1 - y1 * (x2 - x1) / (y2 - y1)) if (y2 - y1) != 0 else float(x1)

    except Exception as e:
        status_msg = f"期权数据获取失败: {str(e)}"
        calc_mode = "接口异常"

    return calls_disp, puts_disp, call_wall, put_wall, gamma_flip, max_pain, pcr_oi, pcr_vol, calc_mode, status_msg, atm_iv, move_1d, move_exp, gex_df

def plot_gex_chart(gex_df, current_price, gamma_flip, max_pain, ticker, title_suffix):
    colors = ["#26A69A" if v >= 0 else "#EF5350" for v in gex_df["net_gex"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gex_df["strike"], y=gex_df["net_gex"], marker_color=colors, name="Net GEX ($M)",
        customdata=np.stack((gex_df["call_gex"], gex_df["put_gex"]), axis=-1),
        hovertemplate="<b>行权价: $%{x:.2f}</b><br>Net GEX: %{y:+.3f} M<br>Call GEX: %{customdata[0]:+.3f} M<br>Put GEX: %{customdata[1]:+.3f} M<extra></extra>",
    ))
    fig.add_vline(x=current_price, line_width=2, line_dash="dash", line_color="#1A237E", annotation_text=f"现价 ${current_price:.2f}", annotation_position="top right")
    if pd.notnull(gamma_flip) and gex_df["strike"].min() <= gamma_flip <= gex_df["strike"].max():
        fig.add_vline(x=gamma_flip, line_width=1.5, line_dash="dot", line_color="#FF9800", annotation_text=f"Flip ${gamma_flip:.2f}", annotation_position="bottom left")
    if pd.notnull(max_pain) and gex_df["strike"].min() <= max_pain <= gex_df["strike"].max():
        fig.add_vline(x=max_pain, line_width=1.5, line_dash="dashdot", line_color="#7B1FA2", annotation_text=f"MaxPain ${max_pain:.2f}", annotation_position="top left")

    fig.update_layout(
        title=f"{ticker} ({title_suffix}) 净伽马敞口分布 (Net GEX)",
        xaxis_title="行权价 ($)", yaxis_title="Net GEX ($M / 1% 波动)",
        height=340, margin=dict(l=10, r=10, t=40, b=20), template="plotly_white", showlegend=False,
    )
    fig.add_hline(y=0, line_width=1, line_color="#9E9E9E")
    return fig

# ==========================================
# 🧠 9. 综合量化诊断与操作建议引擎
# ==========================================
def generate_quant_diagnosis(ticker, hist_df, dp_df, mkt, opt_data):
    """融合趋势、暗池、GEX、波动率与财报风险，输出直观判断与策略建议"""
    last = hist_df.iloc[-1]
    price = float(last["Close"])
    ma5, ma20, mfi = float(last["MA5"]), float(last["MA20"]), float(last["MFI"])
    hv20 = mkt.get("hv20", np.nan)
    earnings_date = mkt.get("earnings_date")

    score = 0
    bullets = []
    warnings = []

    # 1. 趋势与均线结构
    if pd.notnull(ma20) and pd.notnull(ma5):
        if price > ma5 > ma20:
            score += 2
            bullets.append(f"📈 **趋势偏多**：现价 (${price:.2f}) 站稳 MA5 (${ma5:.2f}) 与 MA20 (${ma20:.2f}) 上方，短中期处于多头排列。")
        elif price < ma5 < ma20:
            score -= 2
            bullets.append(f"📉 **趋势偏空**：现价跌破 MA5 与 MA20，短线处于弱势下跌通道，切忌盲目左侧重仓抄底。")
        else:
            bullets.append(f"⚖️ **震荡整理**：均线交织，现价在 MA20 (${ma20:.2f}) 附近拉锯，适合高抛低吸而非追涨杀跌。")

    # 2. 期权 GEX 与关键墙位
    if opt_data:
        _, _, call_wall, put_wall, gamma_flip, max_pain, pcr_oi, _, _, _, atm_iv, _, _, _ = opt_data
        if pd.notnull(gamma_flip):
            if price >= gamma_flip:
                score += 1
                bullets.append(f"🟢 **正伽马环境 (现价 ≥ Flip ${gamma_flip:.2f})**：做市商倾向高抛低吸抑制波动，行情偏向稳步震荡或缓涨，适合依托支撑位低吸。")
            else:
                score -= 1
                warnings.append(f"🔴 **负伽马环境 (现价 < Flip ${gamma_flip:.2f})**：做市商对冲会放大日内波动，容易出现急跌或剧烈洗盘，务必严控仓位与止损。")

        if pd.notnull(call_wall) and pd.notnull(put_wall):
            dist_c = (call_wall / price - 1) * 100
            dist_p = (1 - put_wall / price) * 100
            bullets.append(f"🧱 **核心博弈区间**：上方强阻力 **Call Wall ${call_wall:.2f}** (+{dist_c:.1f}%)，下方强支撑 **Put Wall ${put_wall:.2f}** (-{dist_p:.1f}%)。")

        if pd.notnull(atm_iv) and pd.notnull(hv20) and hv20 > 0:
            iv_hv = atm_iv / hv20
            if iv_hv > 1.25:
                bullets.append(f"💰 **波动率溢价偏高 (IV {atm_iv:.0%} vs HV20 {hv20:.0%})**：期权定价偏贵，**不利于单边买期权 (Long)**，更适合卖方策略 (如 Covered Call / 卖 Put 收租)。")
            elif iv_hv < 0.85:
                bullets.append(f"🎯 **波动率偏低 (IV {atm_iv:.0%} vs HV20 {hv20:.0%})**：期权定价便宜，若判断有方向性突破，**买方期权盈亏比较高**。")

    # 3. FINRA 暗池与 MFI 资金流
    if dp_df is not None and not dp_df.empty:
        sr = float(dp_df["暗池做空占比%"].iloc[0])
        if sr >= 55:
            score -= 1
            bullets.append(f"🌑 **暗池抛压偏重**：最新 FINRA 暗池做空占比达 **{sr:.1f}%**，场外机构存在明显对冲或派发迹象，拉升时注意减仓。")
        elif sr <= 42:
            score += 1
            bullets.append(f"🌑 **暗池买盘承接强**：最新 FINRA 暗池做空占比仅 **{sr:.1f}%**，场外机构偏多承接，回调空间相对受限。")

    if mfi >= 80:
        warnings.append(f"⚠️ **资金流超买 (MFI={mfi:.1f})**：短线获利盘较重，警惕冲高回落。")
    elif mfi <= 20:
        bullets.append(f"💡 **资金流超卖 (MFI={mfi:.1f})**：短线情绪冰点，存在技术性反弹动能。")

    # 4. 财报事件风险
    if earnings_date:
        days_left = (earnings_date - datetime.now(US_EASTERN_TZ).date()).days
        if 0 <= days_left <= 7:
            warnings.append(f"🚨 **财报倒计时预警**：距离下次财报 ({earnings_date}) 仅剩 **{days_left} 天**！面临跳空及 IV Crush 风险，建议降低杠杆或避开裸买末日期权。")

    # 综合定性
    if score >= 2:
        verdict, color = "🟢 偏多格局 · 逢低布局 / 持股待涨", "success"
    elif score <= -2:
        verdict, color = "🔴 偏空格局 · 防守为主 / 逢高减仓", "error"
    else:
        verdict, color = "🟡 震荡博弈 · 严守支撑阻力区间", "warning"

    return verdict, color, bullets, warnings

# ==========================================
# 🔄 10. 后台守护线程
# ==========================================
def background_updater_loop():
    while True:
        try:
            with GLOBAL_STORE["lock"]:
                tickers_to_process = list(GLOBAL_STORE["active_queue"])
            for tk in tickers_to_process:
                try:
                    data = do_fetch_stock_data(tk)
                    if data:
                        with GLOBAL_STORE["lock"]:
                            GLOBAL_STORE["stock_cache"][f"v12_{tk}"] = data
                            GLOBAL_STORE["fetch_timestamps"][f"v12_{tk}"] = data[3]["timestamp"]
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
# 🖥️ 11. 前端 UI 渲染层
# ==========================================
st.markdown("""<style> .main { background-color: #FFFFFF !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; } </style>""", unsafe_allow_html=True)

if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = "NVDA"
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["NVDA", "AAPL", "TSLA", "MSFT", "BTDR"]

with st.sidebar:
    st.header("🔍 全美股检索")
    direct_input = st.text_input("直接输入任意代码 (回车生效)", placeholder="如: PLTR, MSTR, SOXL, BRK-B").strip().upper()
    if direct_input and direct_input != st.session_state.current_ticker:
        st.session_state.current_ticker = direct_input
        with GLOBAL_STORE["lock"]:
            GLOBAL_STORE["active_queue"].add(direct_input)
        st.rerun()

    default_idx = next((i for i, item in enumerate(STOCK_LIBRARY) if item.startswith(st.session_state.current_ticker + " -")), 0)
    selected_item = st.selectbox(f"或从全美股库搜索 (共 {len(STOCK_LIBRARY):,} 只)", options=STOCK_LIBRARY, index=default_idx)
    new_tk = selected_item.split(" - ")[0].strip()
    if new_tk and new_tk != st.session_state.current_ticker and not direct_input:
        st.session_state.current_ticker = new_tk
        with GLOBAL_STORE["lock"]:
            GLOBAL_STORE["active_queue"].add(new_tk)
        st.rerun()

    st.divider()
    st.subheader("📐 换手率口径设置")
    float_mode = st.radio("默认分母口径", ["流通股本 (Float)", "总股本 (Outstanding)"], index=0)
    manual_float = st.number_input(f"{st.session_state.current_ticker} 手动修正股本 (留0自动)", min_value=0, value=0, step=1000000)

    if st.button("🔄 强制刷新当前标的"):
        with GLOBAL_STORE["lock"]:
            GLOBAL_STORE["stock_cache"].pop(f"v12_{st.session_state.current_ticker}", None)
            GLOBAL_STORE["darkpool_cache"].pop(f"dp_{st.session_state.current_ticker}", None)
            GLOBAL_STORE["intraday_cache"].pop(f"{st.session_state.current_ticker}_15m", None)
            GLOBAL_STORE["intraday_cache"].pop(f"{st.session_state.current_ticker}_60m", None)
        st.rerun()

tab_main, tab_watch = st.tabs(["📊 个股深度决策终端", "🔭 自选股监控大屏"])

# ==========================================
# TAB 1: 个股深度决策终端
# ==========================================
with tab_main:
    ticker = st.session_state.current_ticker
    st.title(f"🎯 {ticker} 专业量化决策终端 Pro")

    stock_data = GLOBAL_STORE["stock_cache"].get(f"v12_{ticker}")
    last_ts = GLOBAL_STORE["fetch_timestamps"].get(f"v12_{ticker}", 0)
    is_expired = (time.time() - last_ts) > get_dynamic_ttl()

    if not stock_data or is_expired:
        with st.spinner(f"🔄 正在并发同步 **{ticker}** 行情、暗池与基础数据..."):
            fresh_data = do_fetch_stock_data(ticker)
            if fresh_data:
                stock_data = fresh_data
                with GLOBAL_STORE["lock"]:
                    GLOBAL_STORE["stock_cache"][f"v12_{ticker}"] = fresh_data
                    GLOBAL_STORE["fetch_timestamps"][f"v12_{ticker}"] = fresh_data[3]["timestamp"]
                    GLOBAL_STORE["active_queue"].add(ticker)

    if not stock_data:
        st.error(f"⚠️ 无法拉取 **{ticker}** 的数据，请检查代码是否正确（如 BRK.B 请写为 BRK-B）。")
    else:
        hist_df, reg, dp_df, mkt = stock_data
        last = hist_df.iloc[-1]

        # 换手率分母计算
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

        # 顶部宏观与现价栏
        st.caption(f"🟢 **{ticker}** 数据抓取于 (北京时间): **{mkt['fetch_time']}** | 行情源: **{mkt['source']}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bitcoin", f"${mkt['btc']:,.0f}" if mkt["btc"] > 0 else "N/A")
        m2.metric("Nasdaq", f"{mkt['nasdaq']:,.2f}" if mkt["nasdaq"] > 0 else "N/A", f"{mkt['nasdaq_pct']:.2%}" if mkt["nasdaq"] > 0 else None)
        m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}" if mkt["vix"] > 0 else "N/A", f"{mkt['vix_pct']:.2%}" if mkt["vix"] > 0 else None, delta_color="inverse")
        m4.metric(f"{ticker} 现价", f"${last['Close']:.2f}", f"{(last['Close']/last['昨收']-1):.2%}" if pd.notnull(last["昨收"]) else None)

        # 提前获取期权数据，供顶层诊断引擎 & K 线图叠加使用
        exp_list = get_expiration_list(ticker)
        opt_data = None
        selected_exp_label = ""
        ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1.2])
        with ctrl1:
            tf_choice = st.radio("📈 K线周期", ["1d (日线)", "60m (小时+VWAP)", "15m (15分+VWAP)"], horizontal=True)
        with ctrl2:
            gex_mode = st.radio("🕯️ GEX 统计维度", ["单到期日", "近3期汇总 (Aggregated)"], horizontal=True)
        with ctrl3:
            if exp_list:
                today_str = datetime.now(US_EASTERN_TZ).strftime("%Y-%m-%d")
                default_exp_idx = next((i for i, d in enumerate(exp_list) if d >= today_str), 0)
                selected_exp = st.selectbox("📅 基准到期日", options=exp_list, index=default_exp_idx)
                if gex_mode == "近3期汇总 (Aggregated)":
                    start_i = exp_list.index(selected_exp)
                    target_exps = exp_list[start_i: start_i + 3]
                    selected_exp_label = f"近{len(target_exps)}期汇总: {target_exps[0]}~{target_exps[-1]}"
                else:
                    target_exps = [selected_exp]
                    selected_exp_label = selected_exp

                opt_key = f"v12_{ticker}_{'_'.join(target_exps)}"
                opt_data = GLOBAL_STORE["options_cache"].get(opt_key)
                if not opt_data or is_expired:
                    with st.spinner("🕯️ 正在计算期权链与 GEX..."):
                        opt_data = do_fetch_option_details(ticker, target_exps, last["Close"])
                        with GLOBAL_STORE["lock"]:
                            GLOBAL_STORE["options_cache"][opt_key] = opt_data
            else:
                st.caption("该标的暂无期权链")

        # 🧠 综合量化诊断与操作建议面板
        verdict, v_color, bullets, warnings = generate_quant_diagnosis(ticker, hist_df, dp_df, mkt, opt_data)
        st.subheader("🧠 综合量化诊断与操作建议")
        getattr(st, v_color)(f"**综合研判：{verdict}**")
        for w in warnings:
            st.error(w)
        for b in bullets:
            st.markdown(f"- {b}")

        st.divider()

        # 📈 主 K 线图 (支持多周期 + VWAP + 期权墙位叠加)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
        if tf_choice.startswith("1d"):
            p_df = hist_df.tail(50).copy()
            p_df["label"] = pd.to_datetime(p_df.index).strftime("%m-%d")
            fig.add_trace(go.Scatter(x=p_df["label"], y=p_df["Upper"], line=dict(color="rgba(0,102,204,0.25)"), name="BOLL上轨"), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df["label"], y=p_df["Lower"], line=dict(color="rgba(0,102,204,0.25)"), fill="tonexty", name="BOLL下轨"), row=1, col=1)
            fig.add_trace(go.Candlestick(x=p_df["label"], open=p_df["Open"], high=p_df["High"], low=p_df["Low"], close=p_df["Close"], name="日K"), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df["label"], y=p_df["MA5"], line=dict(color="#FF9800", width=1.5), name="MA5"), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df["label"], y=p_df["MA20"], line=dict(color="#2196F3", width=1.5), name="MA20"), row=1, col=1)
            colors = ["#E53935" if p_df["Close"].iloc[i] >= p_df["Open"].iloc[i] else "#43A047" for i in range(len(p_df))]
            if p_df["换手率_raw"].notnull().any():
                fig.add_trace(go.Bar(x=p_df["label"], y=p_df["换手率_raw"] * 100, marker_color=colors, name="换手率%"), row=2, col=1)
            else:
                fig.add_trace(go.Bar(x=p_df["label"], y=p_df["Volume"] / 10000, marker_color=colors, name="成交量(万股)"), row=2, col=1)
        else:
            iv_code = "60m" if tf_choice.startswith("60m") else "15m"
            intra_df = fetch_intraday_chart(ticker, iv_code)
            if not intra_df.empty:
                p_df = intra_df.tail(65).copy()
                p_df["label"] = [ts.strftime("%m-%d %H:%M") for ts in p_df.index]
                fig.add_trace(go.Candlestick(x=p_df["label"], open=p_df["Open"], high=p_df["High"], low=p_df["Low"], close=p_df["Close"], name=f"{iv_code} K线"), row=1, col=1)
                fig.add_trace(go.Scatter(x=p_df["label"], y=p_df["VWAP"], line=dict(color="#9C27B0", width=2), name="机构 VWAP"), row=1, col=1)
                fig.add_trace(go.Scatter(x=p_df["label"], y=p_df["MA20"], line=dict(color="#2196F3", width=1.2), name="MA20"), row=1, col=1)
                colors = ["#E53935" if p_df["Close"].iloc[i] >= p_df["Open"].iloc[i] else "#43A047" for i in range(len(p_df))]
                fig.add_trace(go.Bar(x=p_df["label"], y=p_df["Volume"] / 10000, marker_color=colors, name="成交量(万股)"), row=2, col=1)
            else:
                st.warning("分时数据拉取失败，请切回日线查看。")

        # 在主 K 线图上叠加 Call Wall / Put Wall / Gamma Flip / Max Pain
        if opt_data:
            _, _, cw, pw, gf, mp, _, _, _, _, _, _, _, _ = opt_data
            if pd.notnull(cw):
                fig.add_hline(y=cw, line_dash="dash", line_color="#2E7D32", line_width=1.5, annotation_text=f"Call Wall ${cw:.2f}", annotation_position="top right", row=1, col=1)
            if pd.notnull(pw):
                fig.add_hline(y=pw, line_dash="dash", line_color="#C62828", line_width=1.5, annotation_text=f"Put Wall ${pw:.2f}", annotation_position="bottom right", row=1, col=1)
            if pd.notnull(gf):
                fig.add_hline(y=gf, line_dash="dot", line_color="#FF9800", line_width=1.5, annotation_text=f"Gamma Flip ${gf:.2f}", annotation_position="top left", row=1, col=1)
            if pd.notnull(mp):
                fig.add_hline(y=mp, line_dash="dashdot", line_color="#7B1FA2", line_width=1.2, annotation_text=f"Max Pain ${mp:.2f}", annotation_position="bottom left", row=1, col=1)

        fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False, margin=dict(l=10, r=10, t=20, b=10))
        fig.update_xaxes(type="category", tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # 实时指标 & 场景回归
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.subheader("📊 实时指标与波动率")
            if eff_float and eff_float > 0:
                st.write(f"实时换手 ({float_label}): **{rt_turnover:.2f}%**")
                st.write(f"预估全天换手: **{proj_turnover:.2f}%** (`{trade_status}`)")
                st.caption(f"分母股本: {eff_float:,.0f} 股 | 成交量: {mkt['volume']:,.0f} 股")
            else:
                st.write("实时换手: **N/A (ETF或无股本数据)**")
            st.write(f"BOLL 上/下轨: **{last['Upper']:.2f} / {last['Lower']:.2f}**")
            st.write(f"资金流量 MFI(14): **{last['MFI']:.2f}** | 20日历史波动率 HV20: **{mkt['hv20']:.1%}**" if pd.notnull(mkt["hv20"]) else f"资金流量 MFI(14): **{last['MFI']:.2f}**")
            if mkt["earnings_date"]:
                st.write(f"📅 下次财报日期: **{mkt['earnings_date']}**")

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

        # 期权 GEX & FINRA 暗池
        o_col, d_col = st.columns([1.5, 1])
        with o_col:
            st.subheader("🕯️ 全景期权与 GEX 分析")
            if opt_data:
                calls_df, puts_df, call_wall, put_wall, gamma_flip, max_pain, pcr_oi, pcr_vol, calc_mode, status_msg, atm_iv, move_1d, move_exp, gex_df = opt_data
                if status_msg:
                    st.info(f"ℹ️ {status_msg}")

                iv_txt = f"{atm_iv:.1%}" if pd.notnull(atm_iv) else "N/A"
                hv_txt = f"{mkt['hv20']:.1%}" if pd.notnull(mkt["hv20"]) else "N/A"
                st.caption(f"维度: **{selected_exp_label}** | 权重: **{calc_mode}** | ATM IV: **{iv_txt}** vs HV20: **{hv_txt}**")

                q1, q2, q3, q4, q5 = st.columns(5)
                q1.metric("🧱 Call Wall", f"${call_wall:.2f}" if pd.notnull(call_wall) else "N/A")
                q2.metric("🧱 Put Wall", f"${put_wall:.2f}" if pd.notnull(put_wall) else "N/A")
                q3.metric("🌀 Gamma Flip", f"${gamma_flip:.2f}" if pd.notnull(gamma_flip) else "N/A")
                q4.metric("🎯 Max Pain", f"${max_pain:.2f}" if pd.notnull(max_pain) else "N/A")
                q5.metric("📊 PCR(OI)", f"{pcr_oi:.2f}" if pd.notnull(pcr_oi) else "N/A")

                if pd.notnull(move_1d) and pd.notnull(move_exp):
                    em1, em2 = st.columns(2)
                    em1.info(f"🎯 单交易日预期波幅: ±${move_1d:.2f} (${last['Close']-move_1d:.2f} ~ ${last['Close']+move_1d:.2f})")
                    em2.success(f"📅 首期到期预期波幅: ±${move_exp:.2f} (${last['Close']-move_exp:.2f} ~ ${last['Close']+move_exp:.2f})")

                if not gex_df.empty and gex_df["net_gex"].abs().sum() > 0:
                    st.plotly_chart(plot_gex_chart(gex_df, last["Close"], gamma_flip, max_pain, ticker, selected_exp_label), use_container_width=True)

                t1, t2 = st.tabs(["📈 首期看涨 (Calls)", "📉 首期看跌 (Puts)"])
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
            st.caption("数据来源: FINRA Reg SHO 官方每日场外 (TRF/ADF) 并发抓取")
            if not dp_df.empty:
                st.dataframe(
                    dp_df[["暗池/场外总量", "暗池成交占比%", "暗池做空占比%"]].style.format({
                        "暗池/场外总量": "{:,.0f}",
                        "暗池成交占比%": "{:.2f}%",
                        "暗池做空占比%": "{:.2f}%",
                    }, na_rep="N/A"),
                    use_container_width=True,
                )
            else:
                st.info("暂无该标的近期的 FINRA 场外记录。")

        st.subheader("📋 近期历史明细")
        hist_show = hist_df.tail(15).sort_index(ascending=False).copy()
        hist_show["换手%"] = (hist_show["换手率_raw"] * 100).map("{:.2f}%".format) if hist_show["换手率_raw"].notnull().any() else "N/A"
        st.dataframe(hist_show[["Open", "High", "Low", "Close", "Volume", "换手%", "MFI", "MA5", "MA20"]].style.format(
            {"Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}", "Volume": "{:,.0f}", "MFI": "{:.2f}", "MA5": "{:.2f}", "MA20": "{:.2f}"}
        ), use_container_width=True)

# ==========================================
# TAB 2: 自选股监控大屏
# ==========================================
with tab_watch:
    st.title("🔭 自选股多标的监控大屏")
    wl_input = st.text_input("编辑自选股列表 (用英文逗号分隔)", value=", ".join(st.session_state.watchlist))
    wl_tickers = [t.strip().upper() for t in wl_input.split(",") if t.strip()]
    st.session_state.watchlist = wl_tickers

    if st.button("🚀 一键扫描自选股", type="primary"):
        rows_wl = []
        progress = st.progress(0)
        for idx, tk in enumerate(wl_tickers):
            data = GLOBAL_STORE["stock_cache"].get(f"v12_{tk}")
            if not data:
                data = do_fetch_stock_data(tk)
                if data:
                    with GLOBAL_STORE["lock"]:
                        GLOBAL_STORE["stock_cache"][f"v12_{tk}"] = data
                        GLOBAL_STORE["fetch_timestamps"][f"v12_{tk}"] = data[3]["timestamp"]
            if data:
                h_df, _, d_df, m = data
                l_row = h_df.iloc[-1]
                chg = (l_row["Close"] / l_row["昨收"] - 1) * 100 if pd.notnull(l_row["昨收"]) else 0.0
                cap = m["float_shares"] or m["shares_out"]
                _, proj_to, _ = calculate_turnover_metrics(m["volume"], cap, m["timestamp"])
                dp_sr = float(d_df["暗池做空占比%"].iloc[0]) if (d_df is not None and not d_df.empty) else np.nan
                trend = "🟢 多头" if (l_row["Close"] > l_row["MA5"] > l_row["MA20"]) else ("🔴 空头" if (l_row["Close"] < l_row["MA5"] < l_row["MA20"]) else "🟡 震荡")
                rows_wl.append({
                    "代码": tk,
                    "现价($)": round(float(l_row["Close"]), 2),
                    "涨跌幅%": round(chg, 2),
                    "趋势结构": trend,
                    "预估换手%": round(proj_to, 2) if cap else np.nan,
                    "MFI(14)": round(float(l_row["MFI"]), 1),
                    "HV20%": round(m["hv20"] * 100, 1) if pd.notnull(m["hv20"]) else np.nan,
                    "暗池做空%": dp_sr,
                    "下次财报": str(m["earnings_date"]) if m["earnings_date"] else "-",
                })
            progress.progress((idx + 1) / max(len(wl_tickers), 1))

        if rows_wl:
            st.dataframe(
                pd.DataFrame(rows_wl).style.format({
                    "现价($)": "{:.2f}",
                    "涨跌幅%": "{:+.2f}%",
                    "预估换手%": "{:.2f}%",
                    "MFI(14)": "{:.1f}",
                    "HV20%": "{:.1f}%",
                    "暗池做空%": "{:.1f}%",
                }, na_rep="N/A"),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("点击上方「一键扫描自选股」即可并发拉取并汇总所有自选标的的核心量化指标。")
