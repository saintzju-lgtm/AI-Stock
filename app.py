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
from curl_cffi import requests as cffi_requests

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
        headers = {'User-Agent': 'Quant-Terminal-App/1.0 (admin@example.com)'}
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

# ==========================================
# ⚙️ 3. 核心量化引擎 (双源智能切换 + 防封锁)
# ==========================================
@st.cache_data(ttl=300)
def get_enhanced_market_data(ticker_symbol):
    try:
        hist = pd.DataFrame()
        info = {}
        
        # --- 抓取尝试 1: Yahoo Finance + TLS 指纹伪造 ---
        try:
            session = cffi_requests.Session(impersonate="chrome110")
            tk = yf.Ticker(ticker_symbol, session=session)
            hist = tk.history(period="100d", interval="1d")
            try: info = tk.info
            except: info = {}
        except Exception:
            hist = pd.DataFrame()

        # --- 抓取尝试 2: 兜底数据源 Stooq (对云端 IP 极友好) ---
        if hist.empty:
            try:
                stooq_code = f"{ticker_symbol}.US" if "^" not in ticker_symbol and "." not in ticker_symbol else ticker_symbol
                hist = web.DataReader(stooq_code, 'stooq').head(100)
                hist = hist.sort_index()
            except Exception:
                pass

        if hist.empty:
            return "所有数据源访问均失败，可能触发频繁限制，请稍后再试。"

        # 宏观指标辅助函数
        def safe_get_macro(sym):
            try:
                m_df = web.DataReader(f"{sym}.US" if sym=="^IXIC" else sym, 'stooq').head(2)
                if not m_df.empty and len(m_df) >= 2:
                    p = m_df['Close'].iloc[-1]
                    p_prev = m_df['Close'].iloc[-2]
                    return p, (p / p_prev - 1)
            except Exception:
                pass
            return 0.0, 0.0

        btc, _ = safe_get_macro("BTC-USD")
        nasdaq, nasdaq_pct = safe_get_macro("^IXIC")
        vix, vix_pct = safe_get_macro("^VIX")

        # 期权数据抓取 (若被封自动容错置空)
        calls_df, puts_df = pd.DataFrame(), pd.DataFrame()
        current_exp, pcr_value = "N/A", "N/A"
        try:
            session = cffi_requests.Session(impersonate="chrome110")
            tk_opt = yf.Ticker(ticker_symbol, session=session)
            exp_dates = tk_opt.options
            if exp_dates:
                today_str = datetime.now().strftime('%Y-%m-%d')
                current_exp = exp_dates[1] if (exp_dates[0] <= today_str and len(exp_dates) > 1) else exp_dates[0]
                opt_data = tk_opt.option_chain(current_exp)
                curr_p = hist['Close'].iloc[-1]
                
                try:
                    total_calls = opt_data.calls['openInterest'].sum()
                    total_puts = opt_data.puts['openInterest'].sum()
                    if total_calls > 0: pcr_value = total_puts / total_calls
                except: pass

                for df_type in ['calls', 'puts']:
                    df = getattr(opt_data, df_type)
                    if not df.empty:
                        idx = (df['strike'] - curr_p).abs().idxmin()
                        slice_df = df.iloc[max(0, idx-4) : min(len(df), idx+5)]
                        if df_type == 'calls': calls_df = slice_df
                        else: puts_df = slice_df
        except Exception:
            pass

        # 流通股矫正
        if ticker_symbol in ["BTDR", "比特小鹿"]:
            current_float = 123715025
        else:
            current_float = info.get('floatShares') if info else 100000000
            if not current_float: current_float = 100000000

        # 数据清洗与指标计算
        hist.index = pd.to_datetime(hist.index).date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        std20 = hist['Close'].rolling(20).std()
        hist['Upper'], hist['Lower'] = hist['MA20'] + std20*2, hist['MA20'] - std20*2
        hist['换手率_raw'] = (hist['Volume'] / current_float)
        
        # MFI 资金流
        tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
        rmf = tp * hist['Volume']
        mfr = pd.Series(np.where(tp > tp.shift(1), rmf, 0)).rolling(14).sum() / pd.Series(np.where(tp < tp.shift(1), rmf, 0)).rolling(14).sum()
        hist['MFI'] = 100 - (100 / (1 + mfr.values))

        # 暗池/大宗信号
        avg_vol = hist['Volume'].mean()
        dark = hist[hist['Volume'] > avg_vol * 1.2].tail(8).copy()
        dark['Signal'] = dark.apply(lambda x: "机构吸筹" if x['Close'] > x['Open'] else "大宗派发", axis=1)

        # 线性回归预测
        fit_df = hist.dropna()
        X = ((fit_df['Open'] - fit_df['昨收']) / fit_df['昨收']).values.reshape(-1, 1)
        reg_params = {}
        for tag, target in [('h', 'High'), ('l', 'Low')]:
            m = LinearRegression().fit(X, fit_df[target].values / fit_df['昨收'].values - 1)
            reg_params[f's_{tag}'], reg_params[f'i_{tag}'] = m.coef_[0], m.intercept_

        return hist, reg_params, calls_df, puts_df, dark, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct, 
            'vix': vix, 'vix_pct': vix_pct, 'float': current_float, 
            'volume': hist['Volume'].iloc[-1] if not hist.empty else 0, 
            'exp': current_exp, 'pcr': pcr_value
        }
    except Exception as e:
        return f"系统核心异常: {str(e)}"

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
    hist_df, reg, calls_df, puts_df, dark_df, mkt = data
    last = hist_df.iloc[-1]
    
    # 全球宏观看板
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bitcoin", f"${mkt['btc']:,.0f}" if mkt['btc'] > 0 else "N/A")
    m2.metric("Nasdaq", f"{mkt['nasdaq']:,.2f}" if mkt['nasdaq'] > 0 else "N/A", f"{mkt['nasdaq_pct']:.2%}")
    m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}" if mkt['vix'] > 0 else "N/A", f"{mkt['vix_pct']:.2%}", delta_color="inverse")
    m4.metric(f"{ticker} 现价", f"${last['Close']:.2f}", f"{(last['Close']/last['昨收']-1):.2%}" if pd.notnull(last['昨收']) else "N/A")

    st.divider()
    
    # 实时指标与场景回归
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时指标")
        turnover_rate = (mkt['volume']/mkt['float'])*100 if mkt['float'] > 0 else 0
        st.write(f"实时换手: **{turnover_rate:.2f}%**")
        st.write(f"BOLL 高/低: **{last['Upper']:.2f} / {last['Lower']:.2f}**")
        st.write(f"资金 MFI: **{last['MFI']:.2f}**")
    with c2:
        st.subheader("📍 场景回归预测")
        ratio_o = (last['Open'] - last['昨收']) / last['昨收'] if last['昨收'] > 0 else 0
        p_h = last['昨收'] * (1 + (reg['i_h'] + reg['s_h'] * ratio_o))
        p_l = last['昨收'] * (1 + (reg['i_l'] + reg['s_l'] * ratio_o))
        st.table(pd.DataFrame({
            "场景": ["看空失效", "中性回归", "支撑测试"], 
            "压力参考": [p_h*1.06, p_h, p_h*0.94], 
            "支撑参考": [p_l*1.06, p_l, p_l*0.94]
        }).style.format(precision=2))

    # K 线与成交量图表
    st.divider()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(40).copy()
    
    # 【已修复】针对 Index 类型正确的 strftime 转换格式
    p_df['label'] = pd.to_datetime(p_df.index).strftime('%m-%d')
    
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Upper'], line=dict(color='rgba(0,102,204,0.3)'), name=f"High:{last['Upper']:.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Lower'], line=dict(color='rgba(0,102,204,0.3)'), fill='tonexty', name=f"Low:{last['Lower']:.2f}"), row=1, col=1)
    fig.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], line=dict(color='#FF9800'), name=f"MA5:{last['MA5']:.2f}"), row=1, col=1)
    
    colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_raw']*100, marker_color=colors, name="换手%"), row=2, col=1)
    
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type='category', tickmode='linear', dtick=1, tickangle=-90)
    st.plotly_chart(fig, use_container_width=True)

    # 期权与大宗交易
    st.divider()
    o_col, d_col = st.columns(2)
    with o_col:
        pcr_display = f" | PCR: {mkt['pcr']:.2f}" if isinstance(mkt.get('pcr'), (float, int)) else ""
        st.subheader(f"🕯️ 全景期权 (到期:{mkt['exp']}{pcr_display})")
        t1, t2 = st.tabs(["📈 看涨 (Calls)", "📉 看跌 (Puts)"])
        
        with t1: 
            if not calls_df.empty: 
                st.dataframe(calls_df[['strike','lastPrice','openInterest','impliedVolatility']].style.format({'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}), use_container_width=True)
            else: 
                st.info("期权接口受云端限流保护，暂不显示")
        with t2: 
            if not puts_df.empty: 
                st.dataframe(puts_df[['strike','lastPrice','openInterest','impliedVolatility']].style.format({'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}), use_container_width=True)
            else: 
                st.info("期权接口受云端限流保护，暂不显示")
            
    with d_col:
        st.subheader("🌑 大宗异动打印 (Dark Pool)")
        if not dark_df.empty: 
            st.table(dark_df[['Volume', 'Signal']])
        else: 
            st.info("近期无显著异动")

    # 历史数据表
    st.subheader("📋 历史明细")
    hist_show = hist_df.tail(15).copy()
    hist_show['换手'] = (hist_show['换手率_raw'] * 100).map('{:.2f}%'.format)
    st.dataframe(hist_show[['Open','High','Low','Close','换手','MFI','MA20','MA5']].style.format(precision=2), use_container_width=True)

    if auto_refresh:
        time.sleep(300)
        st.rerun()
