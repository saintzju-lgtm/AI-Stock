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
import pandas_datareader.data as web
from curl_cffi import requests as cffi_requests

# ==========================================
# 0. 页面全局配置与时区定义
# ==========================================
st.set_page_config(layout="wide", page_title="专业量化决策终端")

BEIJING_TZ = timezone(timedelta(hours=8))

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
        "options_cache": {},        # key: f"{ticker}_{epoch}" -> option result dict
        "expiration_cache": {},     # key: ticker -> {"list":[{date,epoch}...], "ts":...}
        "fetch_timestamps": {},
        "active_queue": set(["BTDR", "AAPL", "TSLA", "NVDA"]),
        "lock": threading.Lock(),
        # --- Yahoo crumb/session 复用,避免每次请求都重新鉴权 ---
        "yahoo_session": None,
        "yahoo_crumb": None,
        "crumb_expire_at": 0.0,
    }

GLOBAL_STORE = get_global_data_store()

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
# 🚀 3. 抗崩溃行情数据抓取引擎(个股 K 线 / 宏观指数)
# ==========================================
def fetch_macro_api(symbol, stooq_symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        if res.status_code == 200:
            meta = res.json()['chart']['result'][0]['meta']
            price = meta.get('regularMarketPrice', 0.0)
            prev = meta.get('chartPreviousClose', meta.get('previousClose', price))
            if price > 0:
                return price, ((price / prev - 1) if prev else 0.0)
    except Exception:
        pass
    try:
        s_df = web.DataReader(stooq_symbol, 'stooq').head(2)
        if not s_df.empty and len(s_df) >= 2:
            p1, p2 = s_df['Close'].iloc[0], s_df['Close'].iloc[1]
            return p1, (p1 / p2 - 1)
    except Exception:
        pass
    return 0.0, 0.0


def do_fetch_stock_data(ticker_symbol):
    try:
        now_ts = time.time()
        fetch_time_bj = datetime.fromtimestamp(now_ts, tz=timezone.utc).astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')

        hist = pd.DataFrame()
        info = {}
        source = "Yahoo Finance"

        try:
            tk = yf.Ticker(ticker_symbol)
            hist = tk.history(period="100d", interval="1d")
            try:
                info = tk.info
            except Exception:
                info = {}
        except Exception:
            hist = pd.DataFrame()

        if hist.empty:
            try:
                stooq_code = f"{ticker_symbol}.US" if "^" not in ticker_symbol and "." not in ticker_symbol else ticker_symbol
                hist = web.DataReader(stooq_code, 'stooq').head(100).sort_index()
                if not hist.empty:
                    source = "Stooq (备用源)"
            except Exception:
                pass

        if hist.empty:
            return None

        btc, _ = fetch_macro_api("BTC-USD", "BTCUSD")
        nasdaq, nasdaq_pct = fetch_macro_api("^IXIC", "^NDQ")
        vix, vix_pct = fetch_macro_api("^VIX", "^VIX")

        current_float = PRESET_FLOATS.get(ticker_symbol, info.get('floatShares') or info.get('sharesOutstanding'))

        hist.index = pd.to_datetime(hist.index).date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        std20 = hist['Close'].rolling(20).std()
        hist['Upper'], hist['Lower'] = hist['MA20'] + std20 * 2, hist['MA20'] - std20 * 2
        hist['换手率_raw'] = (hist['Volume'] / current_float) if (current_float and current_float > 0) else np.nan

        tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
        rmf = tp * hist['Volume']
        mfr = pd.Series(np.where(tp > tp.shift(1), rmf, 0)).rolling(14).sum() / pd.Series(np.where(tp < tp.shift(1), rmf, 0)).rolling(14).sum()
        hist['MFI'] = 100 - (100 / (1 + mfr.values))

        avg_vol = hist['Volume'].mean()
        dark = hist[hist['Volume'] > avg_vol * 1.2].tail(8).copy()
        dark['Signal'] = dark.apply(lambda x: "机构吸筹" if x['Close'] > x['Open'] else "大宗派发", axis=1)

        fit_df = hist.dropna()
        X = ((fit_df['Open'] - fit_df['昨收']) / fit_df['昨收']).values.reshape(-1, 1)
        reg_params = {}
        for tag, target in [('h', 'High'), ('l', 'Low')]:
            m = LinearRegression().fit(X, fit_df[target].values / fit_df['昨收'].values - 1)
            reg_params[f's_{tag}'], reg_params[f'i_{tag}'] = m.coef_[0], m.intercept_

        return hist, reg_params, dark, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct,
            'vix': vix, 'vix_pct': vix_pct, 'float': current_float,
            'volume': hist['Volume'].iloc[-1], 'source': source,
            'fetch_time': fetch_time_bj,
            'timestamp': now_ts
        }
    except Exception:
        return None


# ==========================================
# 🎯 4. Yahoo 期权链客户端(重构核心:统一鉴权 + 到期日以接口原始时间戳为准)
# ==========================================
class YahooOptionsError(Exception):
    """期权接口的显式错误,替代原来的静默 except: pass"""
    pass


def _build_new_yahoo_session():
    session = cffi_requests.Session(impersonate="chrome110")
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def _refresh_yahoo_crumb(store):
    """重新建立会话并领取新 crumb,成功后写回全局缓存(约45分钟有效期内复用)。"""
    session = _build_new_yahoo_session()
    session.get("https://fc.yahoo.com", timeout=6)  # 拿反爬 cookie
    resp = session.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=6)
    crumb = (resp.text or "").strip()
    # crumb 正常应该是一小段无空白短字符串,如果拿到的是网页/错误信息就说明鉴权失败
    if not crumb or len(crumb) > 40 or "<" in crumb:
        raise YahooOptionsError(f"获取 Yahoo 鉴权 crumb 失败(可能被反爬拦截): {crumb[:60]!r}")
    with store["lock"]:
        store["yahoo_session"] = session
        store["yahoo_crumb"] = crumb
        store["crumb_expire_at"] = time.time() + 45 * 60
    return session, crumb


def _get_valid_yahoo_crumb(store, force_refresh=False):
    with store["lock"]:
        session = store.get("yahoo_session")
        crumb = store.get("yahoo_crumb")
        expired = time.time() > store.get("crumb_expire_at", 0)
    if force_refresh or not session or not crumb or expired:
        return _refresh_yahoo_crumb(store)
    return session, crumb


def _request_yahoo_option_chain(ticker_symbol, store, date_epoch=None):
    """
    请求 Yahoo 期权链原始接口。
    date_epoch=None 时,Yahoo 会返回“最近一个到期日”的数据,并附带 expirationDates 全量列表——
    这正是我们用来构造到期日下拉框、且保证后续请求“日期→时间戳”100%对应的唯一权威来源。
    """
    session, crumb = _get_valid_yahoo_crumb(store)
    url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker_symbol}"
    params = {"crumb": crumb}
    if date_epoch is not None:
        params["date"] = int(date_epoch)

    resp = session.get(url, params=params, timeout=8)

    # crumb 失效(401/403)时强制刷新一次再重试,而不是直接判定"无数据"
    if resp.status_code in (401, 403):
        session, crumb = _get_valid_yahoo_crumb(store, force_refresh=True)
        params["crumb"] = crumb
        resp = session.get(url, params=params, timeout=8)

    if resp.status_code != 200:
        raise YahooOptionsError(f"Yahoo 期权接口返回状态码 {resp.status_code},当前标的或该时段可能被限流")

    try:
        payload = resp.json()
    except Exception:
        raise YahooOptionsError("Yahoo 期权接口返回了非 JSON 内容(通常是被反爬拦截,而不是真的没数据)")

    chain = payload.get("optionChain", {})
    if chain.get("error"):
        raise YahooOptionsError(f"Yahoo 接口报错: {chain['error']}")

    results = chain.get("result") or []
    if not results:
        raise YahooOptionsError("该标的没有可用的期权链数据(可能本身不支持期权交易)")

    return results[0]


@st.cache_data(ttl=600)  # 到期日列表10分钟内基本不变,减少对 Yahoo 的请求频次
def get_expiration_list(ticker_symbol):
    """
    关键修复点:到期日列表直接来自 Yahoo 接口自带的 expirationDates 数组(epoch 时间戳),
    不再用 yfinance 的另一套会话去猜日期,也不再用 datetime.strptime 自己拼时间戳。
    下拉框展示的每个日期都和它对应的真实 epoch 一一绑定,选中后直接把 epoch 传回去请求,
    彻底消除"选的到期日 A,实际拿到到期日 B 的数据"这个不准的根源。
    """
    result = _request_yahoo_option_chain(ticker_symbol, GLOBAL_STORE, date_epoch=None)
    epochs = result.get("expirationDates") or []
    exp_list = [
        {"date": datetime.fromtimestamp(int(e), tz=timezone.utc).strftime('%Y-%m-%d'), "epoch": int(e)}
        for e in epochs
    ]
    return exp_list


def do_fetch_option_details(ticker_symbol, target_epoch, fallback_current_price):
    """
    三级智能降级模型(保留原有设计):OI 优先 -> 成交量 -> 行权价分布权重。
    重构点:
      1. date 参数直接用调用方传入的、来自 get_expiration_list 的真实 epoch,不再重新计算。
      2. 现价优先用本次期权接口自带的实时报价(quote.regularMarketPrice),比日线收盘价更准。
      3. IV 缺失/异常的行权价从 Gamma Flip 的数值积分里剔除,不再用固定 25% 顶替(避免虚构数据)。
      4. 明确的异常状态返回,UI 层可以区分"接口失败"和"这个到期日确实没有报价"。
    """
    calls_df, puts_df = pd.DataFrame(), pd.DataFrame()
    call_wall, put_wall, gamma_flip, pcr_value = np.nan, np.nan, np.nan, np.nan
    calc_mode = "持仓量 (OI)"
    status_msg = None

    try:
        result = _request_yahoo_option_chain(ticker_symbol, GLOBAL_STORE, date_epoch=target_epoch)

        # 用接口自带的实时报价覆盖“过时的日线收盘价”,提升墙位/Gamma翻转点的定位精度
        quote = result.get("quote") or {}
        live_price = quote.get("regularMarketPrice")
        current_price = live_price if (isinstance(live_price, (int, float)) and live_price > 0) else fallback_current_price
        if pd.isna(current_price) or current_price <= 0:
            current_price = 10.0

        options_block = (result.get("options") or [{}])[0]
        calls = pd.DataFrame(options_block.get("calls", []))
        puts = pd.DataFrame(options_block.get("puts", []))

        if calls.empty and puts.empty:
            status_msg = "该到期日 Yahoo 未返回任何期权报价(可能是刚上市/深度虚值月份流动性太低)"

        for df in (calls, puts):
            if not df.empty:
                for col in ['strike', 'lastPrice', 'openInterest', 'volume', 'impliedVolatility']:
                    if col not in df.columns:
                        df[col] = 0.0
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                # 有效 IV 掩码:严格剔除缺失/异常值,不再拿 25% 顶替去参与 Gamma 计算
                df['iv_valid'] = df['impliedVolatility'] > 0.01
                df['density_weight'] = 1.0 / (1.0 + (df['strike'] - current_price).abs())

        # 行权价合理区间过滤(保留原逻辑)
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

        # 1. 看涨墙(压力位)
        if not calls.empty:
            c_above = calls[calls['strike'] >= current_price]
            c_target = c_above if not c_above.empty else calls
            call_wall = float(c_target.loc[c_target[w_col].idxmax(), 'strike'])

        # 2. 看跌墙(支撑位)
        if not puts.empty:
            p_below = puts[puts['strike'] <= current_price]
            p_target = p_below if not p_below.empty else puts
            put_wall = float(p_target.loc[p_target[w_col].idxmax(), 'strike'])

        if pd.isna(call_wall) and not calls.empty:
            call_wall = float(calls['strike'].max())
        if pd.isna(put_wall) and not puts.empty:
            put_wall = float(puts['strike'].min())

        # 3. Gamma Flip:只用 IV 有效的行权价参与积分,避免用虚构 IV 污染结果
        try:
            exp_date_dt = datetime.fromtimestamp(target_epoch, tz=timezone.utc)
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

        # 表格切片(展示层保留原始 IV,哪怕是0也如实显示,不做美化)
        for df_type, target_df in [('calls', calls), ('puts', puts)]:
            if not target_df.empty:
                idx = (target_df['strike'] - current_price).abs().idxmin()
                slice_df = target_df.iloc[max(0, idx - 4): min(len(target_df), idx + 5)]
                if df_type == 'calls':
                    calls_df = slice_df
                else:
                    puts_df = slice_df

    except YahooOptionsError as e:
        status_msg = str(e)
        calc_mode = "接口异常"
    except Exception as e:
        status_msg = f"未预期的异常: {e}"
        calc_mode = "接口异常"

    return calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_value, calc_mode, status_msg


# ==========================================
# 🔄 5. 永不崩溃的后台守护线程
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

                        for exp_item in exp_list[:2]:  # 预热最近两个到期日
                            try:
                                opt_key = f"{ticker}_{exp_item['epoch']}"
                                opt_data = do_fetch_option_details(ticker, exp_item['epoch'], last_price)
                                with GLOBAL_STORE["lock"]:
                                    GLOBAL_STORE["options_cache"][opt_key] = opt_data
                                time.sleep(1.0)
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
# 🖥️ 6. 纯前端 UI 渲染层
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
    st.error("⚠️ 当前股票数据拉取失败,可能是股票代码不匹配或数据源暂时中断。")
else:
    hist_df, reg, dark_df, mkt = stock_data
    last = hist_df.iloc[-1]

    st.caption(f"🟢 数据解耦防护中 | **{ticker}** 数据抓取于 (北京时间): **{mkt['fetch_time']}** | 数据源: **{mkt['source']}**")

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
            turnover_rate = (mkt['volume'] / mkt['float']) * 100
            st.write(f"实时换手: **{turnover_rate:.2f}%**")
        else:
            st.write("实时换手: **N/A (无股本数据)**")
        st.write(f"BOLL 高/低: **{last['Upper']:.2f} / {last['Lower']:.2f}**")
        st.write(f"资金 MFI: **{last['MFI']:.2f}**")
    with c2:
        st.subheader("📍 场景回归预测")
        ratio_o = (last['Open'] - last['昨收']) / last['昨收'] if last['昨收'] > 0 else 0
        p_h = last['昨收'] * (1 + (reg['i_h'] + reg['s_h'] * ratio_o))
        p_l = last['昨收'] * (1 + (reg['i_l'] + reg['s_l'] * ratio_o))

        st.table(pd.DataFrame({
            "场景": ["乐观", "中性", "悲观"],
            "压力参考": [p_h * 1.06, p_h, p_h * 0.94],
            "支撑参考": [p_l * 1.06, p_l, p_l * 0.94]
        }).style.format(precision=2))

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
        except YahooOptionsError as e:
            exp_list = []
            st.warning(f"⚠️ 到期日列表获取失败: {e}")
        except Exception as e:
            exp_list = []
            st.warning(f"⚠️ 到期日列表获取异常: {e}")

        if exp_list:
            date_to_epoch = {item['date']: item['epoch'] for item in exp_list}
            date_options = list(date_to_epoch.keys())

            today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            default_exp_idx = 0
            for idx, d in enumerate(date_options):
                if d > today_str:
                    default_exp_idx = idx
                    break

            selected_exp = st.selectbox("📅 选择期权到期日", options=date_options, index=default_exp_idx)
            selected_epoch = date_to_epoch[selected_exp]  # 关键:直接用接口原始 epoch,不做二次转换

            opt_key = f"{ticker}_{selected_epoch}"
            opt_data = GLOBAL_STORE["options_cache"].get(opt_key)
            if not opt_data or is_expired:
                opt_data = do_fetch_option_details(ticker, selected_epoch, last['Close'])
                if opt_data:
                    with GLOBAL_STORE["lock"]:
                        GLOBAL_STORE["options_cache"][opt_key] = opt_data

            if opt_data:
                calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_val, calc_mode, status_msg = opt_data

                if status_msg:
                    st.info(f"ℹ️ {status_msg}")

                st.caption(f"ℹ️ 权重维度说明:当前建模基于 **{calc_mode}** 动态推算 | 到期日 (UTC): **{selected_exp}**")

                q1, q2, q3, q4 = st.columns(4)
                q1.metric("🧱 看涨墙 (Call Wall)", f"${call_wall:.2f}" if pd.notnull(call_wall) else "N/A")
                q2.metric("🧱 看跌墙 (Put Wall)", f"${put_wall:.2f}" if pd.notnull(put_wall) else "N/A")
                q3.metric("🌀 伽马翻转点", f"${gamma_flip:.2f}" if pd.notnull(gamma_flip) else "N/A")
                q4.metric("📊 Put/Call Ratio", f"{pcr_val:.2f}" if pd.notnull(pcr_val) else "N/A")

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
            st.info("💡 当前标的暂无期权链交易,或接口暂时不可用。")

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
