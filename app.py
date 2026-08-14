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
# 0. 全局配置
# ==========================================
st.set_page_config(layout="wide", page_title="专业量化决策终端")

# 预设核心监控池（后台线程会自动轮询更新这些股票）
WATCHLIST = ["BTDR", "AAPL", "TSLA", "NVDA", "MSFT", "GOOG", "QQQ"]

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
# 🧠 1. 全局数据存储中心 (解耦核心)
# ==========================================
@st.cache_resource
def get_global_data_store():
    """全局共享内存：所有用户的终端访问都只从这里取数据"""
    return {
        "market_data": {},     # 结构: { "BTDR": (hist, reg, dark, exp_dates, mkt), ... }
        "options_data": {},    # 结构: { "BTDR_2026-08-21": (calls, puts, wall_c, wall_p, flip, pcr), ... }
        "last_updated": 0,
        "lock": threading.Lock()
    }

GLOBAL_STORE = get_global_data_store()

# ==========================================
# ⚙️ 2. 底层数据抓取引擎 (仅后台线程可调用)
# ==========================================
def fetch_macro_api(symbol, stooq_symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        if res.status_code == 200:
            meta = res.json()['chart']['result'][0]['meta']
            price = meta.get('regularMarketPrice', 0.0)
            prev = meta.get('chartPreviousClose', meta.get('previousClose', price))
            if price > 0: return price, ((price / prev - 1) if prev else 0.0)
    except Exception: pass
    
    try:
        s_df = web.DataReader(stooq_symbol, 'stooq').head(2)
        if not s_df.empty and len(s_df) >= 2:
            p1, p2 = s_df['Close'].iloc[0], s_df['Close'].iloc[1]
            return p1, (p1 / p2 - 1)
    except Exception: pass
    return 0.0, 0.0

def do_fetch_stock_data(ticker_symbol):
    """单次抓取股票与指标数据"""
    try:
        hist = pd.DataFrame()
        info = {}
        source = "Yahoo Finance"
        
        try:
            tk = yf.Ticker(ticker_symbol)
            hist = tk.history(period="100d", interval="1d")
            try: info = tk.info
            except: pass
        except Exception: pass

        if hist.empty:
            try:
                stooq_code = f"{ticker_symbol}.US" if "^" not in ticker_symbol and "." not in ticker_symbol else ticker_symbol
                hist = web.DataReader(stooq_code, 'stooq').head(100).sort_index()
                if not hist.empty: source = "Stooq (备用源)"
            except Exception: pass

        if hist.empty: return None

        btc, _ = fetch_macro_api("BTC-USD", "BTCUSD")
        nasdaq, nasdaq_pct = fetch_macro_api("^IXIC", "^NDQ")
        vix, vix_pct = fetch_macro_api("^VIX", "^VIX")

        exp_dates = []
        if source == "Yahoo Finance":
            try: exp_dates = list(yf.Ticker(ticker_symbol).options)
            except Exception: pass

        current_float = PRESET_FLOATS.get(ticker_symbol, info.get('floatShares') or info.get('sharesOutstanding'))

        hist.index = pd.to_datetime(hist.index).date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        std20 = hist['Close'].rolling(20).std()
        hist['Upper'], hist['Lower'] = hist['MA20'] + std20*2, hist['MA20'] - std20*2
        hist['换手率_raw'] = (hist['Volume'] / current_float) if current_float else np.nan
        
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
            'volume': hist['Volume'].iloc[-1], 'source': source
        }
    except Exception:
        return None

# ==========================================
# 🔄 3. 异步后台守护轮询线程
# ==========================================
def background_updater_loop():
    """后台独立守护线程：5分钟一次全量轮询，与前端完全隔离"""
    while True:
        for ticker in WATCHLIST:
            data = do_fetch_stock_data(ticker)
            if data:
                with GLOBAL_STORE["lock"]:
                    GLOBAL_STORE["market_data"][ticker] = data
            time.sleep(3) # 滴水式优雅节流，每次请求间隔3秒，极度安全
        
        with GLOBAL_STORE["lock"]:
            GLOBAL_STORE["last_updated"] = time.time()
            
        time.sleep(300) # 休眠 5 分钟后进行下一轮更新

@st.cache_resource
def start_background_thread():
    """确保守护线程只启动一次"""
    t = threading.Thread(target=background_updater_loop, daemon=True)
    t.start()
    return True

# 启动后台引擎
start_background_thread()

# ==========================================
# 🖥️ 4. 纯前端渲染层 (零外部网络请求)
# ==========================================
st.markdown("""<style> .main { background-color: #FFFFFF !important; } </style>""", unsafe_allow_html=True)

with st.sidebar:
    selected_ticker = st.selectbox("自选监控池", options=WATCHLIST, index=0)

st.title(f"🎯 {selected_ticker} 专业量化决策终端")

# 🔒 【核心解耦点】：前端只从内存提取，0 次外部网络请求
data = GLOBAL_STORE["market_data"].get(selected_ticker)

if not data:
    st.info("⏳ 后台引擎正在首次初始化抓取数据，请等待 10 秒后刷新页面...")
else:
    hist_df, reg, dark_df, exp_dates, mkt = data
    last = hist_df.iloc[-1]
    
    # 显示更新时间与数据状态
    updated_str = datetime.fromtimestamp(GLOBAL_STORE["last_updated"]).strftime('%H:%M:%S')
    st.caption(f"🟢 数据解耦隔离中 | 内存缓存更新于: **{updated_str}** | 来源: **{mkt['source']}**")
    
    # 顶部看板
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bitcoin", f"${mkt['btc']:,.0f}" if mkt['btc'] > 0 else "N/A")
    m2.metric("Nasdaq", f"{mkt['nasdaq']:,.2f}" if mkt['nasdaq'] > 0 else "N/A", f"{mkt['nasdaq_pct']:.2%}")
    m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}" if mkt['vix'] > 0 else "N/A", f"{mkt['vix_pct']:.2%}", delta_color="inverse")
    m4.metric(f"{selected_ticker} 现价", f"${last['Close']:.2f}", f"{(last['Close']/last['昨收']-1):.2%}")

    st.divider()

    # 实时指标与预测
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时指标")
        if mkt['float']:
            st.write(f"实时换手: **{(mkt['volume']/mkt['float'])*100:.2f}%**")
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

    # K线主图
    st.divider()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(40).copy()
    p_df['label'] = pd.to_datetime(p_df.index).strftime('%m-%d')
    
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Upper'], line=dict(color='rgba(0,102,204,0.3)'), name="Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Lower'], line=dict(color='rgba(0,102,204,0.3)'), fill='tonexty', name="Lower"), row=1, col=1)
    fig.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], line=dict(color='#FF9800'), name="MA5"), row=1, col=1)
    
    colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df['label'], y=p_df['Volume']/10000, marker_color=colors, name="成交量(万股)"), row=2, col=1)
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    st.plotly_chart(fig, use_container_width=True)
