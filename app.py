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

# ==========================================
# 0. 页面全局配置与时区定义
# ==========================================
st.set_page_config(layout="wide", page_title="专业量化决策终端")

BEIJING_TZ = timezone(timedelta(hours=8))

# 全局最小请求间隔(秒):所有打向 Yahoo 的请求(不管前台点击还是后台线程)共享同一个节流器,
# 目的是"别去触发限流",而不是"触发了之后想办法绕过去"。
GLOBAL_RATE_INTERVAL = 1.5

# ==========================================
# 🧠 1. 全局解耦内存存储中心
# ==========================================
@st.cache_resource
def get_global_data_store():
    return {
        "stock_cache": {},
        "options_cache": {},     # key: f"{ticker}_{expiration}" -> option result tuple
        "fetch_timestamps": {},
        "active_queue": set(["BTDR", "AAPL", "TSLA", "NVDA"]),
        "lock": threading.Lock(),
        "last_yahoo_call_ts": 0.0,
    }

GLOBAL_STORE = get_global_data_store()


def _throttled_yahoo_call(func, max_retries=1):
    """
    所有对 Yahoo(不管是走 yfinance 库,还是直接 requests 打 Yahoo 的公开接口)的网络请求统一走这里:
      - 全局节流:整个应用共享一个最小请求间隔,避免并发/高频触发限流。
      - 遇到 429/Too Many Requests 就退避重试一次,超过重试次数就老实报错并提示稍后再试。
      - 不做代理轮换、身份伪装、伪造 User-Agent 池等任何"绕过限制"的操作。
    """
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
                time.sleep(8 + attempt * 7)  # 老实退避,不做规避
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
# 📐 3. 股本数据(换手率分母)
# ==========================================
@st.cache_data(ttl=86400)
def get_share_stats(ticker_symbol):
    """
    换手率分母修复:
      - 不再用硬编码字典(会随增发/回购/拆股过时)。
      - 优先用"流通股 floatShares",拿不到才退回"总股本 sharesOutstanding"。
      - 明确返回用的是哪种口径,UI上标注出来,避免两种不同含义的数字被当成同一件事看。
      - 每天刷新一次,不绑死在行情缓存的生命周期里。
    """
    try:
        info = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).info)
    except Exception:
        info = {}

    float_shares = info.get('floatShares')
    shares_out = info.get('sharesOutstanding')

    if float_shares and float_shares > 0:
        return float_shares, "流通股(Float)"
    if shares_out and shares_out > 0:
        return shares_out, "总股本(Outstanding·近似)"
    return None, "无股本数据"


def get_effective_float(ticker_symbol):
    """支持侧边栏手动修正:如果你对某个标的的股本有更准的数据源,可以自己填,不用改代码。"""
    override_key = f"float_override_{ticker_symbol}"
    override_val = st.session_state.get(override_key)
    if override_val and override_val > 0:
        return float(override_val), "手动修正"
    return get_share_stats(ticker_symbol)


# ==========================================
# 🚀 4. 抗崩溃行情数据抓取引擎(个股 K 线 / 宏观指数)
# ==========================================
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
        s_df = web.DataReader(stooq_symbol, 'stooq').head(2)
        if not s_df.empty and len(s_df) >= 2:
            p1, p2 = s_df['Close'].iloc[0], s_df['Close'].iloc[1]
            return p1, (p1 / p2 - 1)
    except Exception:
        pass
    return 0.0, 0.0


def get_live_quote(ticker_symbol):
    """实时(尽力而为)现价与成交量,来自 yfinance 的 fast_info,比日线K线更贴近当前时刻。"""
    try:
        tk = yf.Ticker(ticker_symbol)
        fi = _throttled_yahoo_call(lambda: tk.fast_info)
        price = fi.get('last_price') or fi.get('lastPrice')
        volume = fi.get('last_volume') or fi.get('lastVolume') or fi.get('regular_market_volume')
        return price, volume
    except Exception:
        return None, None


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

        current_float, float_label = get_effective_float(ticker_symbol)

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

        live_price, live_volume = get_live_quote(ticker_symbol)
        volume_for_turnover = live_volume if (live_volume and live_volume > 0) else hist['Volume'].iloc[-1]

        return hist, reg_params, dark, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct,
            'vix': vix, 'vix_pct': vix_pct,
            'float': current_float, 'float_label': float_label,
            'volume': volume_for_turnover, 'source': source,
            'fetch_time': fetch_time_bj,
            'timestamp': now_ts
        }
    except Exception:
        return None


# ==========================================
# 🎯 5. 期权链(改用 yfinance 自带方法,不再手动拼 crumb/时间戳)
# ==========================================
def get_expiration_list(ticker_symbol):
    """
    到期日直接用 yfinance 的 tk.options,库内部已经处理好了和 Yahoo 的认证与日期映射,
    不需要我们自己算时间戳——这正是之前"选的到期日A、实际拿到B"这个bug的根源,现在直接消失。
    """
    tk = yf.Ticker(ticker_symbol)
    dates = _throttled_yahoo_call(lambda: list(tk.options))
    return dates


def do_fetch_option_details(ticker_symbol, expiration_date, fallback_current_price):
    """
    三级智能降级模型(保留原有设计):OI 优先 -> 成交量 -> 行权价分布权重。
    重构点:
      1. 期权原始数据来自 yfinance 的 tk.option_chain(date),不再手写反爬请求。
      2. 所有网络请求走全局节流器,失败(429)老实退避重试一次,不做任何绕过限制的操作。
      3. 现价优先用 fast_info 的实时价,而不是可能滞后的日线收盘价。
      4. IV 缺失/异常的行权价从 Gamma Flip 的数值积分里剔除,不再用固定值顶替(避免虚构数据)。
      5. 明确的异常状态返回,区分"接口失败/被限流"和"这个到期日确实没有报价"。
    """
    calls_df, puts_df = pd.DataFrame(), pd.DataFrame()
    call_wall, put_wall, gamma_flip, pcr_value = np.nan, np.nan, np.nan, np.nan
    calc_mode = "持仓量 (OI)"
    status_msg = None

    try:
        tk = yf.Ticker(ticker_symbol)
        opt = _throttled_yahoo_call(lambda: tk.option_chain(expiration_date), max_retries=1)
        calls, puts = opt.calls.copy(), opt.puts.copy()

        if calls.empty and puts.empty:
            status_msg = "该到期日 Yahoo 未返回任何期权报价(可能流动性太低)"

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
                df['iv_valid'] = df['impliedVolatility'] > 0.01  # 无效IV直接从Gamma计算里剔除,不用固定值顶替
                df['density_weight'] = 1.0 / (1.0 + (df['strike'] - current_price).abs())

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
        msg = str(e)
        if "429" in msg or "Too Many Requests" in msg:
            status_msg = "触发 Yahoo 限流(429),请等几分钟再试,或降低刷新频率(不建议强行重试)"
        else:
            status_msg = f"期权数据获取失败: {msg}"
        calc_mode = "接口异常"

    return calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_value, calc_mode, status_msg


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

                        for exp_date in exp_list[:2]:  # 只预热最近两个到期日,减少总请求量
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

    st.caption(f"📡 期权数据源: Yahoo Finance(经 yfinance) | 全局节流间隔: {GLOBAL_RATE_INTERVAL}s")

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
            turnover_rate = (mkt['volume'] / mkt['float']) * 100
            st.write(f"实时换手 ({mkt['float_label']}): **{turnover_rate:.2f}%**")
            st.caption(f"分母股本: {mkt['float']:,.0f} 股 | 分子成交量: {mkt['volume']:,.0f} 股")
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
        except Exception as e:
            exp_list = []
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                st.warning("⚠️ 到期日列表获取被限流(429),请等几分钟再试。")
            else:
                st.warning(f"⚠️ 到期日列表获取异常: {e}")

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
                calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_val, calc_mode, status_msg = opt_data

                if status_msg:
                    st.info(f"ℹ️ {status_msg}")

                st.caption(f"ℹ️ 权重维度: **{calc_mode}** | 到期日: **{selected_exp}**")

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
