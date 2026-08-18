import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timezone, timedelta
import threading
import requests
from io import StringIO
import pandas_datareader.data as web
from curl_cffi import requests as cffi_requests

# ==========================================
# 0. 页面全局配置与时区定义
# ==========================================
st.set_page_config(layout="wide", page_title="专业量化决策终端")

BEIJING_TZ = timezone(timedelta(hours=8))
GLOBAL_RATE_INTERVAL = 1.0

PRESET_FLOATS = {
    "BTDR": 123715025,
    "AAPL": 15200000000,
    "TSLA": 3180000000,
    "NVDA": 24500000000,
    "MSFT": 7430000000,
    "GOOG": 12300000000,
    "QQQ": 600000000
}

# ==========================================
# 🧠 1. 全局解耦内存存储中心
# ==========================================
@st.cache_resource
def get_global_data_store():
    return {
        "stock_cache": {},
        "options_cache": {},
        "fetch_timestamps": {},
        "active_queue": set(["BTDR", "AAPL", "TSLA", "NVDA"]),
        "lock": threading.Lock(),
        "last_yahoo_call_ts": 0.0,
    }

GLOBAL_STORE = get_global_data_store()

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
                time.sleep(5 + attempt * 5)
                continue
            raise
    raise last_err

# ==========================================
# 🔍 2. 动态加载全量美股库
# ==========================================
@st.cache_data(ttl=86400)
def load_us_stock_library():
    custom_library = [
        "BTDR - 比特小鹿 (自选)", "AAPL - 苹果 (Apple Inc.)", "TSLA - 特斯拉 (Tesla)",
        "NVDA - 英伟达 (NVIDIA)", "MSFT - 微软 (Microsoft)", "GOOG - 谷歌 (Alphabet)",
        "BTC-USD - 比特币 (Bitcoin)", "QQQ - 纳指ETF", "^VIX - 恐慌指数",
        "000409.SZ - 云鼎科技", "600325.SS - 华发股份"
    ]
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=5)
        data = res.json()
        us_stocks = [f"{item['ticker']} - {item['title'].title()}" for item in data.values()]
        return custom_library + [s for s in us_stocks if s.split(" - ")[0] not in [c.split(" - ")[0] for c in custom_library]]
    except Exception:
        return custom_library + ["AMZN - Amazon", "META - Meta Platforms", "BABA - Alibaba", "TCEHY - Tencent", "BILI - Bilibili"]

STOCK_LIBRARY = load_us_stock_library()

# ==========================================
# 📐 3. 股本数据与换手率计算
# ==========================================
@st.cache_data(ttl=86400)
def get_share_stats(ticker_symbol):
    info = {}
    try:
        info = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).info)
    except Exception:
        info = {}

    float_shares = info.get('floatShares') if info else None
    shares_out = info.get('sharesOutstanding') if info else None

    float_shares = float(float_shares) if float_shares and float_shares > 0 else None
    shares_out = float(shares_out) if shares_out and shares_out > 0 else None

    if not float_shares and not shares_out and ticker_symbol in PRESET_FLOATS:
        return float(PRESET_FLOATS[ticker_symbol]), "预设股本(兜底)"

    if float_shares and shares_out and float_shares > shares_out:
        float_shares = shares_out

    if float_shares:
        return float_shares, "流通股(Float)"
    if shares_out:
        return shares_out, "总股本(Outstanding·近似)"
        
    return None, "无股本数据"

def get_effective_float(ticker_symbol):
    override_key = f"float_override_{ticker_symbol}"
    override_val = st.session_state.get(override_key)
    if override_val and override_val > 0:
        return float(override_val), "手动修正"
    return get_share_stats(ticker_symbol)

# ==========================================
# 🚀 4. 抗崩溃行情数据抓取引擎
# ==========================================
def fetch_chart_stooq_csv(symbol):
    try:
        clean_sym = symbol.lower().split('.')[0].replace('^', '')
        stooq_sym = f"{clean_sym}.us" if not symbol.startswith('^') and '-USD' not in symbol else clean_sym
        url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200 and "Date" in res.text:
            df = pd.read_csv(StringIO(res.text))
            df.columns = [c.capitalize() for c in df.columns]
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['Close', 'Open', 'High', 'Low']).copy()
            if len(df) >= 2:
                return df
    except Exception:
        pass
    return pd.DataFrame()

def fetch_macro_api(symbol, stooq_symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
        res = _throttled_yahoo_call(lambda: requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3))
        if res.status_code == 200:
            meta = res.json()['chart']['result'][0]['meta']
            price = meta.get('regularMarketPrice', 0.0)
            prev = meta.get('chartPreviousClose', meta.get('previousClose', price))
            if price > 0:
                return price, ((price / prev - 1) if prev else 0.0)
    except Exception:
        pass
    try:
        s_df = fetch_chart_stooq_csv(stooq_symbol)
        if not s_df.empty and len(s_df) >= 2:
            p1, p2 = s_df['Close'].iloc[-1], s_df['Close'].iloc[-2]
            return p1, (p1 / p2 - 1)
    except Exception:
        pass
    return 0.0, 0.0

def get_live_quote(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        fi = _throttled_yahoo_call(lambda: tk.fast_info)
        price = fi.get('last_price') or fi.get('lastPrice')
        volume = fi.get('last_volume') or fi.get('lastVolume') or fi.get('regular_market_volume')
        return price, volume
    except Exception:
        return None, None

def build_scenario_model(hist_df):
    fit_df = hist_df.dropna(subset=['Open', 'High', 'Low', 'Close', '昨收']).copy()
    if len(fit_df) < 15:
        return {'mode': 'insufficient'}

    fit_df['gap'] = (fit_df['Open'] - fit_df['昨收']) / fit_df['昨收']
    ret = fit_df['Close'].pct_change()
    fit_df['recent_vol'] = ret.rolling(10).std()
    fit_df = fit_df.dropna(subset=['gap', 'recent_vol'])

    if len(fit_df) < 15:
        return {'mode': 'insufficient'}

    y_h = (fit_df['High'] / fit_df['昨收'] - 1).values
    y_l = (fit_df['Low'] / fit_df['昨收'] - 1).values

    X_simple = fit_df[['gap']].values
    fallback = {}
    for tag, y in [('h', y_h), ('l', y_l)]:
        m = LinearRegression().fit(X_simple, y)
        fallback[f's_{tag}'], fallback[f'i_{tag}'] = m.coef_[0], m.intercept_

    if len(fit_df) >= 25:
        try:
            from sklearn.linear_model import QuantileRegressor
            X = fit_df[['gap', 'recent_vol']].values
            quantile_map = {'乐观': 0.9, '中性': 0.5, '悲观': 0.1}
            models = {'h': {}, 'l': {}}
            for scene, q in quantile_map.items():
                models['h'][scene] = QuantileRegressor(quantile=q, alpha=0.3, solver='highs').fit(X, y_h)
                models['l'][scene] = QuantileRegressor(quantile=q, alpha=0.3, solver='highs').fit(X, y_l)
            return {'mode': 'quantile', 'models': models, 'params': fallback}
        except Exception:
            pass

    return {'mode': 'linear_fallback', 'params': fallback}

def predict_scenarios(scenario_model, hist_df):
    if not scenario_model or scenario_model.get('mode') == 'insufficient':
        return None, 'insufficient'

    last = hist_df.iloc[-1]
    prev_close = last['昨收']
    if pd.isna(prev_close) or prev_close <= 0:
        return None, 'insufficient'

    open_p = last['Open'] if pd.notnull(last['Open']) else prev_close
    gap = (open_p - prev_close) / prev_close
    if pd.isna(gap) or np.isinf(gap):
        gap = 0.0

    if scenario_model['mode'] == 'quantile':
        try:
            ret = hist_df['Close'].pct_change()
            recent_vol = ret.rolling(10).std().iloc[-1]
            if pd.isna(recent_vol) or np.isinf(recent_vol):
                recent_vol = ret.std()
            if pd.isna(recent_vol) or np.isinf(recent_vol):
                recent_vol = 0.01

            X_pred = np.array([[float(gap), float(recent_vol)]])
            models = scenario_model['models']

            h_vals = {scene: prev_close * (1 + models['h'][scene].predict(X_pred)[0]) for scene in ['乐观', '中性', '悲观']}
            l_vals = {scene: prev_close * (1 + models['l'][scene].predict(X_pred)[0]) for scene in ['乐观', '中性', '悲观']}

            h_sorted = sorted(h_vals.values(), reverse=True)
            l_sorted = sorted(l_vals.values(), reverse=True)
            rows = [(scene, h_sorted[i], l_sorted[i]) for i, scene in enumerate(['乐观', '中性', '悲观'])]
            return rows, 'quantile'
        except Exception:
            pass

    if scenario_model.get('params'):
        p = scenario_model['params']
        p_h = prev_close * (1 + (p['i_h'] + p['s_h'] * gap))
        p_l = prev_close * (1 + (p['i_l'] + p['s_l'] * gap))
        rows = [
            ('乐观', p_h * 1.06, p_l * 1.06),
            ('中性', p_h, p_l),
            ('悲观', p_h * 0.94, p_l * 0.94),
        ]
        return rows, 'linear_fallback'

    return None, 'insufficient'

def estimate_iv_expected_move(ticker_symbol, current_price):
    try:
        exp_list = get_expiration_list(ticker_symbol)
        if not exp_list:
            return None
        nearest_exp = exp_list[0]

        opt_key = f"{ticker_symbol}_{nearest_exp}"
        opt_data = GLOBAL_STORE["options_cache"].get(opt_key)
        if not opt_data:
            opt_data = do_fetch_option_details(ticker_symbol, nearest_exp, current_price)

        calls_df, puts_df = opt_data[0], opt_data[1]
        atm_ivs = []
        for df in (calls_df, puts_df):
            if df is not None and not df.empty:
                idx = (df['strike'] - current_price).abs().idxmin()
                iv = df.loc[idx, 'impliedVolatility']
                if iv and iv > 0.01:
                    atm_ivs.append(float(iv))

        if not atm_ivs:
            return None

        atm_iv = float(np.mean(atm_ivs))
        days_to_exp = max((datetime.strptime(nearest_exp, '%Y-%m-%d') - datetime.now()).days, 1)
        move_1day = current_price * atm_iv * np.sqrt(1 / 365)
        return {'atm_iv': atm_iv, 'nearest_exp': nearest_exp, 'days_to_exp': days_to_exp, 'move_1day': move_1day}
    except Exception:
        return None

def calculate_turnover_metrics(volume, capital, timestamp_unix):
    if not capital or capital <= 0 or not volume or volume <= 0:
        return 0.0, 0.0, "数据不足"

    realtime_turnover = (volume / capital) * 100.0

    dt_utc = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc)
    dt_est = dt_utc.astimezone(timezone(timedelta(hours=-4)))

    weekday = dt_est.weekday()
    time_min = dt_est.hour * 60 + dt_est.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
    total_trade_min = 390
    min_reliable_elapsed = 30

    if weekday < 5 and open_min <= time_min <= close_min:
        elapsed = time_min - open_min
        if elapsed < min_reliable_elapsed:
            return realtime_turnover, realtime_turnover, f"盘中(开盘{elapsed}分钟,量太少暂不外推全天)"
        projected_vol = volume * (total_trade_min / elapsed)
        projected_turnover = (projected_vol / capital) * 100.0
        return realtime_turnover, projected_turnover, f"盘中外推(已交易{elapsed}分钟)"
    elif weekday < 5 and time_min < open_min:
        return realtime_turnover, realtime_turnover, "未开盘(盘前)"
    else:
        return realtime_turnover, realtime_turnover, "已收盘/盘后"

def do_fetch_stock_data(ticker_symbol):
    try:
        now_ts = time.time()
        fetch_time_bj = datetime.fromtimestamp(now_ts, tz=timezone.utc).astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')

        hist = pd.DataFrame()
        source = "Yahoo Finance"

        try:
            hist = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).history(period="100d", interval="1d"))
        except Exception:
            hist = pd.DataFrame()

        if hist.empty or len(hist) < 2:
            try:
                hist = fetch_chart_stooq_csv(ticker_symbol)
                if not hist.empty:
                    source = "Stooq (网络兜底)"
            except Exception:
                pass

        if hist.empty or len(hist) < 2:
            return None

        # 🎯 强转数值类型，清洗字符串 'None' 和空数据
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors='coerce')

        hist = hist.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        hist = hist[(hist['Close'] > 0) & (hist['Open'] > 0)].copy()

        if hist.empty or len(hist) < 2:
            return None

        btc, _ = fetch_macro_api("BTC-USD", "btc.us")
        nasdaq, nasdaq_pct = fetch_macro_api("^IXIC", "^ndq")
        vix, vix_pct = fetch_macro_api("^VIX", "^vix")

        current_float, float_label = get_effective_float(ticker_symbol)

        hist.index = pd.to_datetime(hist.index).date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        std20 = hist['Close'].rolling(20).std().fillna(0)
        hist['Upper'], hist['Lower'] = hist['MA20'] + std20 * 2, hist['MA20'] - std20 * 2
        hist['换手率_raw'] = (hist['Volume'] / current_float) if (current_float and current_float > 0) else np.nan

        tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
        rmf = tp * hist['Volume']
        
        # 🎯 资金 MFI 索引强制对齐修复，防止产生 nan
        pos_flow = pd.Series(np.where(tp > tp.shift(1), rmf, 0), index=hist.index)
        neg_flow = pd.Series(np.where(tp < tp.shift(1), rmf, 0), index=hist.index)
        
        mfr = pos_flow.rolling(14, min_periods=1).sum() / neg_flow.rolling(14, min_periods=1).sum().replace(0, 1)
        hist['MFI'] = (100 - (100 / (1 + mfr))).fillna(50)

        avg_vol = hist['Volume'].mean()
        dark = hist[hist['Volume'] > avg_vol * 1.2].tail(8).copy()
        if not dark.empty:
            dark['Signal'] = dark.apply(lambda x: "机构吸筹" if x['Close'] > x['Open'] else "大宗派发", axis=1)

        scenario_model = build_scenario_model(hist)

        live_price, live_volume = get_live_quote(ticker_symbol)
        volume_for_turnover = live_volume if (live_volume and live_volume > 0) else hist['Volume'].iloc[-1]
        rt_turnover, proj_turnover, trade_status = calculate_turnover_metrics(volume_for_turnover, current_float, now_ts)

        return hist, scenario_model, dark, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct,
            'vix': vix, 'vix_pct': vix_pct,
            'float': current_float, 'float_label': float_label,
            'rt_turnover': rt_turnover, 'proj_turnover': proj_turnover, 'trade_status': trade_status,
            'volume': volume_for_turnover, 'source': source,
            'fetch_time': fetch_time_bj,
            'timestamp': now_ts
        }
    except Exception:
        return None

# ==========================================
# 🎯 5. 期权链模块 (伪造 TLS 穿透抗 429)
# ==========================================
def get_expiration_list(ticker_symbol):
    try:
        session = cffi_requests.Session(impersonate="chrome110")
        session.get("https://fc.yahoo.com", timeout=4)
        crumb = session.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=4).text.strip()
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker_symbol}?crumb={crumb}"
        res = session.get(url, timeout=5)
        if res.status_code == 200:
            timestamps = res.json()['optionChain']['result'][0].get('expirationDates', [])
            return [datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d') for ts in timestamps]
    except Exception:
        pass
    
    try:
        tk = yf.Ticker(ticker_symbol)
        return _throttled_yahoo_call(lambda: list(tk.options))
    except Exception:
        return []

def get_raw_options_with_crumb(ticker_symbol, selected_exp):
    try:
        session = cffi_requests.Session(impersonate="chrome110")
        session.get("https://fc.yahoo.com", timeout=4)
        crumb = session.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=4).text.strip()
        exp_ts = int(datetime.strptime(selected_exp, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker_symbol}?date={exp_ts}&crumb={crumb}"
        res = session.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            options = data['optionChain']['result'][0].get('options', [{}])[0]
            calls = pd.DataFrame(options.get('calls', []))
            puts = pd.DataFrame(options.get('puts', []))
            return calls, puts
    except Exception:
        pass
    return pd.DataFrame(), pd.DataFrame()

def do_fetch_option_details(ticker_symbol, expiration_date, fallback_current_price):
    calls_df, puts_df = pd.DataFrame(), pd.DataFrame()
    call_wall, put_wall, gamma_flip, pcr_value = np.nan, np.nan, np.nan, np.nan
    atm_iv, move_1d, move_exp = np.nan, np.nan, np.nan
    calc_mode = "持仓量 (OI)"
    status_msg = None

    try:
        calls, puts = get_raw_options_with_crumb(ticker_symbol, expiration_date)

        if calls.empty and puts.empty:
            try:
                tk = yf.Ticker(ticker_symbol)
                opt = _throttled_yahoo_call(lambda: tk.option_chain(expiration_date), max_retries=1)
                calls, puts = opt.calls.copy(), opt.puts.copy()
            except Exception:
                pass

        if calls.empty and puts.empty:
            status_msg = "该到期日 Yahoo 未返回任何期权报价(可能流动性太低或被限流)"

        live_price, _ = get_live_quote(ticker_symbol)
        current_price = live_price if (isinstance(live_price, (int, float)) and live_price and live_price > 0) else fallback_current_price
        if pd.isna(current_price) or current_price <= 0:
            current_price = 10.0

        for df in (calls, puts):
            if not df.empty:
                for col in ['strike', 'lastPrice', 'openInterest', 'volume', 'impliedVolatility']:
                    if col not in df.columns:
                        df[col] = 0.0
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                df['iv_valid'] = df['impliedVolatility'] > 0.01
                df['density_weight'] = 1.0 / (1.0 + (df['strike'] - current_price).abs())

        all_valid = pd.concat([
            calls[calls['iv_valid']] if not calls.empty else pd.DataFrame(),
            puts[puts['iv_valid']] if not puts.empty else pd.DataFrame(),
        ], ignore_index=True) if (not calls.empty or not puts.empty) else pd.DataFrame()

        if not all_valid.empty:
            atm_idx = (all_valid['strike'] - current_price).abs().idxmin()
            atm_iv = float(all_valid.loc[atm_idx, 'impliedVolatility'])
            days_to_exp = max((datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days, 1)
            move_1d = current_price * atm_iv * np.sqrt(1 / 365)
            move_exp = current_price * atm_iv * np.sqrt(days_to_exp / 365)

        if not calls.empty:
            calls = calls[(calls['strike'] >= current_price * 0.2) & (calls['strike'] <= current_price * 3.0)].copy()
        if not puts.empty:
            puts = puts[(puts['strike'] >= current_price * 0.2) & (puts['strike'] <= current_price * 3.0)].copy()

        tot_c_oi = calls['openInterest'].sum() if not calls.empty else 0
        tot_p_oi = puts['openInterest'].sum() if not puts.empty else 0
        tot_c_vol = calls['volume'].sum() if not calls.empty else 0
        tot_p_vol = puts['volume'].sum() if not puts.empty else 0

        if (tot_c_oi + tot_p_oi) > 0:
            w_col = 'openInterest'
            calc_mode = "持仓量 (OI)"
            pcr_value = tot_p_oi / tot_c_oi if tot_c_oi > 0 else 1.0
        elif (tot_c_vol + tot_p_vol) > 0:
            w_col = 'volume'
            calc_mode = "成交量 (Volume)"
            pcr_value = tot_p_vol / tot_c_vol if tot_c_vol > 0 else 1.0
        else:
            w_col = 'density_weight'
            calc_mode = "行权价分布"
            pcr_value = 1.0

        if not calls.empty:
            c_above = calls[calls['strike'] >= current_price]
            c_target = c_above if not c_above.empty else calls
            call_wall = float(c_target.loc[c_target[w_col].idxmax(), 'strike'])
        if not puts.empty:
            p_below = puts[puts['strike'] <= current_price]
            p_target = p_below if not p_below.empty else puts
            put_wall = float(p_target.loc[p_target[w_col].idxmax(), 'strike'])

        if pd.isna(call_wall) and not calls.empty:
            call_wall = float(calls['strike'].max())
        if pd.isna(put_wall) and not puts.empty:
            put_wall = float(puts['strike'].min())

        try:
            exp_date_dt = datetime.strptime(expiration_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            T = max((exp_date_dt - datetime.now(timezone.utc)).days, 1) / 365.0
            s_range = np.linspace(current_price * 0.7, current_price * 1.3, 80)
            net_gammas = []

            valid_calls = calls[calls['iv_valid']] if not calls.empty else calls
            valid_puts = puts[puts['iv_valid']] if not puts.empty else puts

            for s_test in s_range:
                tot_g = 0.0
                for df_leg, sign in ((valid_calls, 1.0), (valid_puts, -1.0)):
                    if df_leg.empty:
                        continue
                    for _, row in df_leg.iterrows():
                        k, w, iv = row['strike'], row[w_col], row['impliedVolatility']
                        w_val = max(float(w), 0.1)
                        if k > 0 and iv > 0:
                            d1 = (np.log(s_test / k) + 0.5 * (iv ** 2) * T) / (iv * np.sqrt(T))
                            tot_g += sign * w_val * (np.exp(-0.5 * d1 ** 2) / (s_test * iv * np.sqrt(2 * np.pi * T)))
                net_gammas.append(tot_g)

            net_gammas = np.array(net_gammas)
            zero_crossings = np.where(np.diff(np.sign(net_gammas)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                y1, y2 = net_gammas[idx], net_gammas[idx + 1]
                x1, x2 = s_range[idx], s_range[idx + 1]
                gamma_flip = float(x1 - y1 * (x2 - x1) / (y2 - y1)) if (y2 - y1) != 0 else float(x1)
            else:
                gamma_flip = float((call_wall + put_wall) / 2.0) if (pd.notnull(call_wall) and pd.notnull(put_wall)) else float(current_price)
        except Exception:
            gamma_flip = float(current_price)

        for df_type, target_df in [('calls', calls), ('puts', puts)]:
            if not target_df.empty:
                idx = (target_df['strike'] - current_price).abs().idxmin()
                slice_df = target_df.iloc[max(0, idx - 4): min(len(target_df), idx + 5)]
                if df_type == 'calls':
                    calls_df = slice_df
                else:
                    puts_df = slice_df

    except Exception as e:
        status_msg = f"期权数据获取失败: {str(e)}"
        calc_mode = "接口异常"

    return calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_value, calc_mode, status_msg, atm_iv, move_1d, move_exp

# ==========================================
# 🔄 6. 永不崩溃的后台守护线程
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
                            GLOBAL_STORE["stock_cache"][ticker] = data
                            GLOBAL_STORE["fetch_timestamps"][ticker] = data[3]['timestamp']

                        hist_df, _, _, _ = data
                        last_price = hist_df['Close'].iloc[-1]

                        try:
                            exp_list = get_expiration_list(ticker)
                        except Exception:
                            exp_list = []

                        for exp_date in exp_list[:2]:
                            try:
                                opt_key = f"{ticker}_{exp_date}"
                                opt_data = do_fetch_option_details(ticker, exp_date, last_price)
                                with GLOBAL_STORE["lock"]:
                                    GLOBAL_STORE["options_cache"][opt_key] = opt_data
                            except Exception:
                                pass

                except Exception:
                    pass
                time.sleep(2.0)

        except Exception:
            pass
        time.sleep(180)

@st.cache_resource
def start_background_engine():
    t = threading.Thread(target=background_updater_loop, daemon=True)
    t.start()
    return True

start_background_engine()

# ==========================================
# 🖥️ 7. 纯前端 UI 渲染层
# ==========================================
st.markdown("""<style> .main { background-color: #FFFFFF !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; } </style>""", unsafe_allow_html=True)

if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = "BTDR"

with st.sidebar:
    default_idx = 0
    for i, item in enumerate(STOCK_LIBRARY):
        if item.startswith(st.session_state.current_ticker + " -"):
            default_idx = i
            break

    selected_item = st.selectbox(
        "搜索股票",
        options=STOCK_LIBRARY,
        index=default_idx,
        placeholder="例如: BTDR 或 AAPL"
    )

    new_tk = selected_item.split(" - ")[0].strip()
    if new_tk and new_tk != st.session_state.current_ticker:
        st.session_state.current_ticker = new_tk
        with GLOBAL_STORE["lock"]:
            GLOBAL_STORE["active_queue"].add(new_tk)
        st.rerun()

    st.divider()
    st.caption("💡 换手率分母的手动修正(留空/填0则自动获取)")
    manual_float = st.number_input(
        f"{st.session_state.current_ticker} 流通股手动修正",
        min_value=0, value=0, step=1000000,
        help="如果你对这个标的的流通股数量有更准确的数据源,可以在这里手动填入,留 0 则走自动获取。"
    )
    override_key = f"float_override_{st.session_state.current_ticker}"
    st.session_state[override_key] = manual_float if manual_float > 0 else None

    st.caption(f"📡 期权数据源: Yahoo Finance(经 TLS 穿透) | 节流间隔: {GLOBAL_RATE_INTERVAL}s")

ticker = st.session_state.current_ticker
st.title(f"🎯 {ticker} 专业量化决策终端")

stock_data = GLOBAL_STORE["stock_cache"].get(ticker)
last_ts = GLOBAL_STORE["fetch_timestamps"].get(ticker, 0)
now_ts = time.time()
is_expired = (now_ts - last_ts) > 180

if not stock_data or is_expired:
    with st.spinner(f"🔄 正在同步 **{ticker}** 最新市场数据..."):
        fresh_data = do_fetch_stock_data(ticker)
        if fresh_data:
            stock_data = fresh_data
            with GLOBAL_STORE["lock"]:
                GLOBAL_STORE["stock_cache"][ticker] = fresh_data
                GLOBAL_STORE["fetch_timestamps"][ticker] = fresh_data[3]['timestamp']
                GLOBAL_STORE["active_queue"].add(ticker)

if not stock_data:
    st.error("⚠️ 当前股票数据拉取失败,可能是股票代码不匹配、数据源暂时中断,或被限流(稍后再试)。")
else:
    hist_df, reg, dark_df, mkt = stock_data
    last = hist_df.iloc[-1]

    st.caption(f"🟢 数据解耦防护中 | **{ticker}** 数据抓取于 (北京时间): **{mkt['fetch_time']}** | 行情源: **{mkt['source']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bitcoin", f"${mkt['btc']:,.0f}" if mkt['btc'] > 0 else "N/A")
    m2.metric("Nasdaq", f"{mkt['nasdaq']:,.2f}" if mkt['nasdaq'] > 0 else "N/A", f"{mkt['nasdaq_pct']:.2%}" if mkt['nasdaq'] > 0 else "N/A")
    m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}" if mkt['vix'] > 0 else "N/A", f"{mkt['vix_pct']:.2%}" if mkt['vix'] > 0 else "N/A", delta_color="inverse")
    m4.metric(f"{ticker} 现价", f"${last['Close']:.2f}", f"{(last['Close']/last['昨收']-1):.2%}" if pd.notnull(last['昨收']) else "N/A")

    st.divider()

    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时指标")
        if mkt['float'] and mkt['float'] > 0:
            st.write(f"实时已换手 ({mkt['float_label']}): **{mkt['rt_turnover']:.2f}%**")
            st.write(f"预估全天换手: **{mkt['proj_turnover']:.2f}%** (`{mkt['trade_status']}`)")
            st.caption(f"分母股本: {mkt['float']:,.0f} 股 | 分子成交量: {mkt['volume']:,.0f} 股")
        else:
            st.write("实时换手: **N/A (无股本数据)**")
        st.write(f"BOLL 高/低: **{last['Upper']:.2f} / {last['Lower']:.2f}**")
        st.write(f"资金 MFI: **{last['MFI']:.2f}**")
    with c2:
        st.subheader("📍 场景回归预测")
        rows, mode = predict_scenarios(reg, hist_df)

        if rows:
            mode_label = {
                'quantile': "分位数回归(10/50/90分位,已按近10日实际波动率自适应)",
                'linear_fallback': "样本不足或分位数回归失败,已退回简单线性回归+固定区间",
            }.get(mode, mode)
            st.caption(f"模型: {mode_label}")

            st.table(pd.DataFrame({
                "场景": [r[0] for r in rows],
                "压力参考": [r[1] for r in rows],
                "支撑参考": [r[2] for r in rows],
            }).style.format(precision=2))

            iv_ref = estimate_iv_expected_move(ticker, last['Close'])
            if iv_ref:
                st.caption(
                    f"🔮 期权隐含参考(交叉校验): 最近到期日 {iv_ref['nearest_exp']}"
                    f"(剩{iv_ref['days_to_exp']}天),ATM隐含波动率 {iv_ref['atm_iv']:.1%},"
                    f"换算1个交易日预期波动约 ±${iv_ref['move_1day']:.2f} "
                    f"(区间 ${last['Close']-iv_ref['move_1day']:.2f} ~ ${last['Close']+iv_ref['move_1day']:.2f})"
                )
            else:
                st.caption("🔮 期权隐含参考: 暂无法获取(可能被限流,或该标的期权流动性不足)")
        else:
            st.info("历史样本不足,暂无法生成场景预测(通常是新股/次新股历史K线太短)")

    st.divider()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(40).copy()
    p_df['label'] = pd.to_datetime(p_df.index).strftime('%m-%d')

    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Upper'], line=dict(color='rgba(0,102,204,0.3)'), name=f"High:{last['Upper']:.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Lower'], line=dict(color='rgba(0,102,204,0.3)'), fill='tonexty', name=f"Low:{last['Lower']:.2f}"), row=1, col=1)
    fig.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], line=dict(color='#FF9800'), name=f"MA5:{last['MA5']:.2f}"), row=1, col=1)

    colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]

    if p_df['换手率_raw'].notnull().any():
        fig.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_raw'] * 100, marker_color=colors, name="换手%"), row=2, col=1)
    else:
        fig.add_trace(go.Bar(x=p_df['label'], y=p_df['Volume'] / 10000, marker_color=colors, name="成交量(万股)"), row=2, col=1)

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type='category', tickmode='linear', dtick=1, tickangle=-90)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    o_col, d_col = st.columns([1.6, 1])

    with o_col:
        st.subheader("🕯️ 全景期权决策分析")

        try:
            exp_list = get_expiration_list(ticker)
        except Exception as e:
            exp_list = []

        if exp_list:
            today_str = datetime.now().strftime('%Y-%m-%d')
            default_exp_idx = 0
            for idx, d in enumerate(exp_list):
                if d > today_str:
                    default_exp_idx = idx
                    break

            selected_exp = st.selectbox("📅 选择期权到期日", options=exp_list, index=default_exp_idx)

            opt_key = f"{ticker}_{selected_exp}"
            opt_data = GLOBAL_STORE["options_cache"].get(opt_key)
            if not opt_data or is_expired:
                opt_data = do_fetch_option_details(ticker, selected_exp, last['Close'])
                if opt_data:
                    with GLOBAL_STORE["lock"]:
                        GLOBAL_STORE["options_cache"][opt_key] = opt_data

            if opt_data:
                calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_val, calc_mode, status_msg, atm_iv, move_1d, move_exp = opt_data

                if status_msg:
                    st.info(f"ℹ️ {status_msg}")

                iv_caption = f" | ATM隐含波动率: **{atm_iv:.2%}**" if pd.notnull(atm_iv) else " | ATM隐含波动率: 暂无有效IV"
                st.caption(f"ℹ️ 权重维度: **{calc_mode}** | 到期日: **{selected_exp}**{iv_caption}")

                q1, q2, q3, q4 = st.columns(4)
                q1.metric("🧱 看涨墙 (Call Wall)", f"${call_wall:.2f}" if pd.notnull(call_wall) else "N/A")
                q2.metric("🧱 看跌墙 (Put Wall)", f"${put_wall:.2f}" if pd.notnull(put_wall) else "N/A")
                q3.metric("🌀 伽马翻转点", f"${gamma_flip:.2f}" if pd.notnull(gamma_flip) else "N/A")
                q4.metric("📊 Put/Call Ratio", f"{pcr_val:.2f}" if pd.notnull(pcr_val) else "N/A")

                if pd.notnull(move_1d) and pd.notnull(move_exp):
                    em1, em2 = st.columns(2)
                    em1.info(f"🎯 期权隐含单日预期波幅: ±${move_1d:.2f} (区间 ${last['Close']-move_1d:.2f} ~ ${last['Close']+move_1d:.2f})")
                    em2.success(f"📅 至到期日({selected_exp})预期波幅: ±${move_exp:.2f} (区间 ${last['Close']-move_exp:.2f} ~ ${last['Close']+move_exp:.2f})")
                else:
                    st.caption("🎯 期权隐含预期波幅: 该到期日暂无有效IV数据,不做估算(不会拿假数字凑数)")

                t1, t2 = st.tabs(["📈 看涨 (Calls)", "📉 看跌 (Puts)"])
                with t1:
                    if not calls_df.empty:
                        st.dataframe(calls_df[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']].style.format(
                            {'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}
                        ), use_container_width=True)
                    else:
                        st.info("该到期日暂无看涨期权数据")
                with t2:
                    if not puts_df.empty:
                        st.dataframe(puts_df[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']].style.format(
                            {'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}
                        ), use_container_width=True)
                    else:
                        st.info("该到期日暂无看跌期权数据")
        else:
            st.info("💡 当前标的暂无期权链交易,或接口暂时不可用/被限流。")

    with d_col:
        st.subheader("🌑 大宗异动打印 (Dark Pool)")
        if not dark_df.empty:
            st.table(dark_df[['Volume', 'Signal']])
        else:
            st.info("近期无显著异动")

    st.subheader("📋 历史明细")
    hist_show = hist_df.tail(15).copy()
    if hist_show['换手率_raw'].notnull().any():
        hist_show['换手'] = (hist_show['换手率_raw'] * 100).map('{:.2f}%'.format)
    else:
        hist_show['换手'] = "N/A"
    st.dataframe(hist_show[['Open', 'High', 'Low', 'Close', '换手', 'MFI', 'MA20', 'MA5']].style.format(precision=2), use_container_width=True)
