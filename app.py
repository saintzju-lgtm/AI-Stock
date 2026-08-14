import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import threading
import requests
import pandas_datareader.data as web

# ==========================================
# 0. 页面全局配置 (必须放在第一行)
# ==========================================
st.set_page_config(layout="wide", page_title="专业量化决策终端")

# ==========================================
# 🛡️ 1. 零成本终极防御模块 (全局锁 + 恶意惩罚)
# ==========================================
@st.cache_resource
def get_global_api_state():
    return {
        "last_real_request_time": 0.0,
        "lock": threading.Lock() 
    }

global_api_state = get_global_api_state()

if 'malicious_strikes' not in st.session_state:
    st.session_state.malicious_strikes = 0
if 'user_penalty_until' not in st.session_state:
    st.session_state.user_penalty_until = 0.0
if 'last_user_click' not in st.session_state:
    st.session_state.last_user_click = 0.0
if 'current_ticker' not in st.session_state: 
    st.session_state.current_ticker = "BTDR"

def verify_and_lock_request():
    """核准用户请求，拦截恶意高频与多用户并发"""
    now = time.time()
    
    if now < st.session_state.user_penalty_until:
        remaining = int(st.session_state.user_penalty_until - now)
        st.error(f"🛑 检测到恶意高频刷新！您的访问已被锁定，请等待 {remaining} 秒。")
        return False

    time_since_last_click = now - st.session_state.last_user_click
    st.session_state.last_user_click = now
    
    if time_since_last_click < 3.0: 
        st.session_state.malicious_strikes += 1
        if st.session_state.malicious_strikes >= 3:
            st.session_state.user_penalty_until = now + 60.0
            st.session_state.malicious_strikes = 0
            st.error("🛑 警告：操作过于频繁，触发 60 秒惩罚锁定！")
            return False
        else:
            st.warning("⚠️ 请勿频繁点击，系统处理中...")
            return False
    else:
        st.session_state.malicious_strikes = max(0, st.session_state.malicious_strikes - 1)

    with global_api_state["lock"]:
        if now - global_api_state["last_real_request_time"] < 5.0:
            st.info("⏳ 系统正在处理其他用户的排队请求，请 5 秒后再试。")
            return False
        global_api_state["last_real_request_time"] = now
        return True

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
        return custom_library + [
            "AMZN - Amazon", "META - Meta Platforms", "BABA - Alibaba", 
            "TCEHY - Tencent", "BILI - Bilibili"
        ]

STOCK_LIBRARY = load_us_stock_library()

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
# 🚀 3. 双源数据引擎 (主用 Yahoo / 备用 Stooq)
# ==========================================
def fetch_macro_with_fallback(symbol, stooq_symbol):
    """宏观数据双源兜底抓取"""
    # 优先 Yahoo API
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=2)
        if res.status_code == 200:
            meta = res.json()['chart']['result'][0]['meta']
            price = meta.get('regularMarketPrice', 0.0)
            prev = meta.get('chartPreviousClose', meta.get('previousClose', price))
            if price > 0:
                return price, ((price / prev - 1) if prev else 0.0)
    except Exception:
        pass

    # 备用 Stooq
    try:
        s_df = web.DataReader(stooq_symbol, 'stooq').head(2)
        if not s_df.empty and len(s_df) >= 2:
            p_last = s_df['Close'].iloc[0]
            p_prev = s_df['Close'].iloc[1]
            return p_last, (p_last / p_prev - 1)
    except Exception:
        pass

    return 0.0, 0.0

@st.cache_data(ttl=300)
def get_enhanced_market_data(ticker_symbol):
    try:
        hist = pd.DataFrame()
        info = {}
        data_source_used = "Yahoo Finance"
        
        # --- 1. K 线主数据源：Yahoo ---
        try:
            tk = yf.Ticker(ticker_symbol)
            hist = tk.history(period="100d", interval="1d")
            try: info = tk.info
            except: info = {}
        except Exception:
            hist = pd.DataFrame()

        # --- 2. K 线备用数据源：Stooq (当 Yahoo 429 或返回空时自动触发) ---
        if hist.empty or len(hist) < 5:
            try:
                stooq_code = f"{ticker_symbol}.US" if "^" not in ticker_symbol and "." not in ticker_symbol else ticker_symbol
                hist = web.DataReader(stooq_code, 'stooq').head(100)
                hist = hist.sort_index()
                if not hist.empty:
                    data_source_used = "Stooq (备用源)"
            except Exception:
                pass

        if hist.empty:
            return "所有数据源（Yahoo与Stooq）均访问失败，请检查网络或稍后再试。"

        # --- 3. 宏观数据抓取（双源保活） ---
        btc, _ = fetch_macro_with_fallback("BTC-USD", "BTCUSD")
        nasdaq, nasdaq_pct = fetch_macro_with_fallback("^IXIC", "^NDQ")
        vix, vix_pct = fetch_macro_with_fallback("^VIX", "^VIX")

        # --- 4. 期权到期日抓取 ---
        exp_dates = []
        if data_source_used == "Yahoo Finance":
            try:
                tk_opt = yf.Ticker(ticker_symbol)
                exp_dates = list(tk_opt.options)
            except Exception:
                exp_dates = []

        current_float = None
        if ticker_symbol in PRESET_FLOATS:
            current_float = PRESET_FLOATS[ticker_symbol]
        elif info:
            current_float = info.get('floatShares') or info.get('sharesOutstanding')

        hist.index = pd.to_datetime(hist.index).date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        std20 = hist['Close'].rolling(20).std()
        hist['Upper'], hist['Lower'] = hist['MA20'] + std20*2, hist['MA20'] - std20*2
        
        if current_float and current_float > 0:
            hist['换手率_raw'] = (hist['Volume'] / current_float)
        else:
            hist['换手率_raw'] = np.nan
        
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

        return hist, reg_params, dark, exp_dates, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct, 
            'vix': vix, 'vix_pct': vix_pct, 'float': current_float, 
            'volume': hist['Volume'].iloc[-1] if not hist.empty else 0,
            'data_source': data_source_used
        }
    except Exception as e:
        return f"系统核心异常: {str(e)}"

# -------------------------------------------------------------
# 期权链计算逻辑
# -------------------------------------------------------------
@st.cache_data(ttl=300)
def get_option_chain_details(ticker_symbol, selected_exp, current_price):
    calls_df, puts_df = pd.DataFrame(), pd.DataFrame()
    call_wall, put_wall, gamma_flip, pcr_value = np.nan, np.nan, np.nan, np.nan
    
    try:
        tk_opt = yf.Ticker(ticker_symbol)
        opt_data = tk_opt.option_chain(selected_exp)
        calls = opt_data.calls
        puts = opt_data.puts

        total_calls_oi = calls['openInterest'].sum() if not calls.empty else 0
        total_puts_oi = puts['openInterest'].sum() if not puts.empty else 0
        if total_calls_oi > 0:
            pcr_value = total_puts_oi / total_calls_oi

        if not calls.empty and calls['openInterest'].max() > 0:
            call_wall = calls.loc[calls['openInterest'].idxmax()]['strike']
        if not puts.empty and puts['openInterest'].max() > 0:
            put_wall = puts.loc[puts['openInterest'].idxmax()]['strike']

        try:
            exp_date = datetime.strptime(selected_exp, '%Y-%m-%d')
            days = (exp_date - datetime.now()).days
            T = max(days, 1) / 365.0
            
            s_range = np.linspace(current_price * 0.6, current_price * 1.4, 150)
            net_gammas = []

            for s_test in s_range:
                tot_g = 0.0
                for _, row in calls.iterrows():
                    k, oi, iv = row['strike'], row['openInterest'], row.get('impliedVolatility', 0.2)
                    if oi > 0 and k > 0 and iv > 0.01:
                        d1 = (np.log(s_test / k) + 0.5 * (iv**2) * T) / (iv * np.sqrt(T))
                        gamma = np.exp(-0.5 * d1**2) / (s_test * iv * np.sqrt(2 * np.pi * T))
                        tot_g += oi * gamma
                for _, row in puts.iterrows():
                    k, oi, iv = row['strike'], row['openInterest'], row.get('impliedVolatility', 0.2)
                    if oi > 0 and k > 0 and iv > 0.01:
                        d1 = (np.log(s_test / k) + 0.5 * (iv**2) * T) / (iv * np.sqrt(T))
                        gamma = np.exp(-0.5 * d1**2) / (s_test * iv * np.sqrt(2 * np.pi * T))
                        tot_g -= oi * gamma
                net_gammas.append(tot_g)

            net_gammas = np.array(net_gammas)
            zero_crossings = np.where(np.diff(np.sign(net_gammas)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                y1, y2 = net_gammas[idx], net_gammas[idx+1]
                x1, x2 = s_range[idx], s_range[idx+1]
                gamma_flip = x1 - y1 * (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else x1
            else:
                tot_oi = total_calls_oi + total_puts_oi
                if tot_oi > 0:
                    gamma_flip = ((calls['strike']*calls['openInterest']).sum() + (puts['strike']*puts['openInterest']).sum()) / tot_oi
        except Exception:
            pass

        for df_type, target_df in [('calls', calls), ('puts', puts)]:
            if not target_df.empty:
                idx = (target_df['strike'] - current_price).abs().idxmin()
                slice_df = target_df.iloc[max(0, idx-4) : min(len(target_df), idx+5)]
                if df_type == 'calls': calls_df = slice_df
                else: puts_df = slice_df

    except Exception:
        pass

    return calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_value

# ==========================================
# 🖥️ 4. 终端 UI 渲染层
# ==========================================
st.markdown("""<style> .main { background-color: #FFFFFF !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; } </style>""", unsafe_allow_html=True)

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
        if verify_and_lock_request():
            st.session_state.current_ticker = new_tk
            st.rerun()
            
    st.divider()
    auto_refresh = st.checkbox("开启 5分钟自动刷新", value=False)

ticker = st.session_state.current_ticker
st.title(f"🎯 {ticker} 专业量化决策终端")

data = get_enhanced_market_data(ticker)

if isinstance(data, str):
    st.error(f"⚠️ {data}")
elif data:
    hist_df, reg, dark_df, exp_dates, mkt = data
    last = hist_df.iloc[-1]
    
    # 🟢 明确提示当前正处于哪个数据源（让切换完全透明可见）
    if "Stooq" in mkt['data_source']:
        st.warning(f"🟡 注意：主数据源 (Yahoo) 响应异常，当前已自动降级并切换至 **{mkt['data_source']}**。(注：备用源仅支持价格/K线，不提供期权链数据)")
    else:
        st.caption(f"🟢 数据链路正常 | 当前主用数据源: **{mkt['data_source']}**")
    
    # 全球宏观看板
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bitcoin", f"${mkt['btc']:,.0f}" if mkt['btc'] > 0 else "N/A")
    m2.metric("Nasdaq", f"{mkt['nasdaq']:,.2f}" if mkt['nasdaq'] > 0 else "N/A", f"{mkt['nasdaq_pct']:.2%}" if mkt['nasdaq'] > 0 else "N/A")
    m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}" if mkt['vix'] > 0 else "N/A", f"{mkt['vix_pct']:.2%}" if mkt['vix'] > 0 else "N/A", delta_color="inverse")
    m4.metric(f"{ticker} 现价", f"${last['Close']:.2f}", f"{(last['Close']/last['昨收']-1):.2%}" if pd.notnull(last['昨收']) else "N/A")

    st.divider()
    
    # 实时指标与场景回归
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时指标")
        if mkt['float'] and mkt['float'] > 0:
            turnover_rate = (mkt['volume']/mkt['float'])*100
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
            "压力参考": [p_h*1.06, p_h, p_h*0.94], 
            "支撑参考": [p_l*1.06, p_l, p_l*0.94]
        }).style.format(precision=2))

    # K 线与成交量图表
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
        fig.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_raw']*100, marker_color=colors, name="换手%"), row=2, col=1)
    else:
        fig.add_trace(go.Bar(x=p_df['label'], y=p_df['Volume']/10000, marker_color=colors, name="成交量(万股)"), row=2, col=1)
    
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type='category', tickmode='linear', dtick=1, tickangle=-90)
    st.plotly_chart(fig, use_container_width=True)

    # 期权与大宗交易模块
    st.divider()
    o_col, d_col = st.columns([1.6, 1])
    
    with o_col:
        st.subheader("🕯️ 全景期权决策分析")
        
        if exp_dates:
            today_str = datetime.now().strftime('%Y-%m-%d')
            default_exp_idx = 0
            for idx, ed in enumerate(exp_dates):
                if ed >= today_str:
                    default_exp_idx = idx
                    break
            
            selected_exp = st.selectbox("📅 选择期权到期日", options=exp_dates, index=default_exp_idx)
            calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_val = get_option_chain_details(ticker, selected_exp, last['Close'])
            
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("🧱 看涨墙 (Call Wall)", f"${call_wall:.2f}" if pd.notnull(call_wall) else "N/A")
            q2.metric("🧱 看跌墙 (Put Wall)", f"${put_wall:.2f}" if pd.notnull(put_wall) else "N/A")
            q3.metric("🌀 伽马翻转点", f"${gamma_flip:.2f}" if pd.notnull(gamma_flip) else "N/A")
            q4.metric("📊 Put/Call Ratio", f"{pcr_val:.2f}" if pd.notnull(pcr_val) else "N/A")

            t1, t2 = st.tabs(["📈 看涨 (Calls)", "📉 看跌 (Puts)"])
            with t1: 
                if not calls_df.empty: 
                    st.dataframe(calls_df[['strike','lastPrice','openInterest','impliedVolatility']].style.format({'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}), use_container_width=True)
                else: 
                    st.info("该到期日暂无看涨期权数据或受到接口频控")
            with t2: 
                if not puts_df.empty: 
                    st.dataframe(puts_df[['strike','lastPrice','openInterest','impliedVolatility']].style.format({'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}), use_container_width=True)
                else: 
                    st.info("该到期日暂无看跌期权数据或受到接口频控")
        else:
            st.info("💡 当前处于 Stooq 备用源模式（无期权链）或该股票暂无期权交易。")
            
    with d_col:
        st.subheader("🌑 大宗异动打印 (Dark Pool)")
        if not dark_df.empty: 
            st.table(dark_df[['Volume', 'Signal']])
        else: 
            st.info("近期无显著异动")

    # 历史数据表
    st.subheader("📋 历史明细")
    hist_show = hist_df.tail(15).copy()
    if hist_show['换手率_raw'].notnull().any():
        hist_show['换手'] = (hist_show['换手率_raw'] * 100).map('{:.2f}%'.format)
    else:
        hist_show['换手'] = "N/A"
    st.dataframe(hist_show[['Open','High','Low','Close','换手','MFI','MA20','MA5']].style.format(precision=2), use_container_width=True)

    if auto_refresh:
        time.sleep(300)
        st.rerun()
