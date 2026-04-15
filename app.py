import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# --- 0. 页面全局配置 ---
st.set_page_config(layout="wide", page_title="专业量化决策终端")

# --- 智能中英文模糊匹配引擎 ---
TICKER_MAP = {
    "苹果": "AAPL", "特斯拉": "TSLA", "英伟达": "NVDA", "微软": "MSFT",
    "谷歌": "GOOG", "亚马逊": "AMZN", "脸书": "META", "比特小鹿": "BTDR",
    "阿里巴巴": "BABA", "腾讯": "TCEHY", "哔哩哔哩": "BILI", "B站": "BILI",
    "比特": "BTC-USD", "纳指": "QQQ", "恐慌指数": "^VIX"
}

def fuzzy_match_ticker(query):
    query = query.strip().upper()
    if not query: return "BTDR"
    if query in TICKER_MAP.values(): return query
    if query in TICKER_MAP: return TICKER_MAP[query]
    if len(query) >= 2:
        for name, ticker in TICKER_MAP.items():
            if query in name: return ticker
    return query

# --- 1. 核心量化引擎 (依赖 YF 自带底层反爬) ---
@st.cache_data(ttl=300) 
def get_enhanced_market_data(ticker_symbol):
    try:
        time.sleep(1) # 轻微延迟，防止一瞬间并发
        tk = yf.Ticker(ticker_symbol)
        info = tk.info
        hist = tk.history(period="100d", interval="1d")
        
        if hist.empty: return "数据源返回为空，请检查代码或等待接口恢复。"

        # 容错获取宏观数据 (移除自定义 session)
        def safe_get_macro(sym):
            try:
                t = yf.Ticker(sym)
                p = t.fast_info['last_price']
                return p, (p / t.fast_info['previous_close'] - 1)
            except: return 0.0, 0.0

        btc, _ = safe_get_macro("BTC-USD")
        nasdaq, nasdaq_pct = safe_get_macro("^IXIC")
        vix, vix_pct = safe_get_macro("^VIX")

        # 期权链处理
        calls_df, puts_df = pd.DataFrame(), pd.DataFrame()
        current_exp = "N/A"
        try:
            exp_dates = tk.options
            if exp_dates:
                today_str = datetime.now().strftime('%Y-%m-%d')
                current_exp = exp_dates[1] if (exp_dates[0] <= today_str and len(exp_dates) > 1) else exp_dates[0]
                opt_data = tk.option_chain(current_exp)
                curr_p = hist['Close'].iloc[-1]
                # ATM 切片
                for df_type in ['calls', 'puts']:
                    df = getattr(opt_data, df_type)
                    idx = (df['strike'] - curr_p).abs().idxmin()
                    slice_df = df.iloc[max(0, idx-4) : min(len(df), idx+5)]
                    if df_type == 'calls': calls_df = slice_df
                    else: puts_df = slice_df
        except: pass

        # 指标计算
        current_float = info.get('floatShares') or info.get('shares') or 118500000
        hist.index = hist.index.date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        std20 = hist['Close'].rolling(20).std()
        hist['Upper'], hist['Lower'] = hist['MA20'] + std20*2, hist['MA20'] - std20*2
        hist['换手率_raw'] = (hist['Volume'] / current_float)
        
        # MFI
        tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
        rmf = tp * hist['Volume']
        mfr = pd.Series(np.where(tp > tp.shift(1), rmf, 0)).rolling(14).sum() / pd.Series(np.where(tp < tp.shift(1), rmf, 0)).rolling(14).sum()
        hist['MFI'] = 100 - (100 / (1 + mfr.values))

        # 大宗识别
        avg_vol = hist['Volume'].mean()
        dark = hist[hist['Volume'] > avg_vol * 1.2].tail(8).copy()
        dark['Signal'] = dark.apply(lambda x: "机构吸筹" if x['Close'] > x['Open'] else "大宗派发", axis=1)

        # 回归模型
        fit_df = hist.dropna()
        X = ((fit_df['Open'] - fit_df['昨收']) / fit_df['昨收']).values.reshape(-1, 1)
        reg_params = {}
        for tag, target in [('h', 'High'), ('l', 'Low')]:
            m = LinearRegression().fit(X, fit_df[target].values / fit_df['昨收'].values - 1)
            reg_params[f's_{tag}'], reg_params[f'i_{tag}'] = m.coef_[0], m.intercept_

        return hist, reg_params, calls_df, puts_df, dark, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct, 
            'vix': vix, 'vix_pct': vix_pct, 'float': current_float, 'volume': info.get('regularMarketVolume', 0), 'exp': current_exp
        }
    except Exception as e:
        return f"系统核心异常: {str(e)}"

# --- 2. UI 渲染 ---
st.markdown("""<style> .main { background-color: #FFFFFF !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; } </style>""", unsafe_allow_html=True)

if 'current_ticker' not in st.session_state: st.session_state.current_ticker = "BTDR"
if 'last_q_time' not in st.session_state: st.session_state.last_q_time = 0.0

with st.sidebar:
    st.markdown("### 🔍 切换股票")
    raw_input = st.text_input("支持中英文模糊搜索", value=st.session_state.current_ticker)
    new_tk = fuzzy_match_ticker(raw_input)
    
    if new_tk and new_tk != st.session_state.current_ticker:
        # 防刷锁：30 秒内禁止连续查询新股票
        if (time.time() - st.session_state.last_q_time) < 30.0:
            st.error(f"⏳ 防刷锁生效: 请等待 {int(30 - (time.time()-st.session_state.last_q_time))} 秒再切换")
        else:
            st.session_state.current_ticker = new_tk
            st.session_state.last_q_time = time.time()
            st.rerun()
            
    st.divider()
    auto_refresh = st.checkbox("开启 5分钟自动无感刷新", value=True)

ticker = st.session_state.current_ticker
data = get_enhanced_market_data(ticker)

if isinstance(data, str):
    st.error(f"⚠️ {data}")
elif data:
    hist_df, reg, calls_df, puts_df, dark_df, mkt = data
    last = hist_df.iloc[-1]
    
    # 宏观看板
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bitcoin", f"${mkt['btc']:,.0f}" if mkt['btc'] > 0 else "N/A")
    m2.metric("Nasdaq", f"{mkt['nasdaq']:,.2f}" if mkt['nasdaq'] > 0 else "N/A", f"{mkt['nasdaq_pct']:.2%}")
    m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}" if mkt['vix'] > 0 else "N/A", f"{mkt['vix_pct']:.2%}", delta_color="inverse")
    m4.metric(f"{ticker} 现价", f"${last['Close']:.2f}", f"{(last['Close']/last['昨收']-1):.2%}")

    st.divider()
    
    # 指标与回归
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时指标")
        # 换手率容错，防 0 做分母
        turnover_rate = (mkt['volume']/mkt['float'])*100 if mkt['float'] > 0 else 0
        st.write(f"实时换手: **{turnover_rate:.2f}%**")
        st.write(f"BOLL 高/低: **{last['Upper']:.2f} / {last['Lower']:.2f}**")
        st.write(f"资金 MFI: **{last['MFI']:.2f}**")
    with c2:
        st.subheader("📍 场景回归预测")
        ratio_o = (last['Open'] - last['昨收']) / last['昨收']
        p_h = last['昨收'] * (1 + (reg['i_h'] + reg['s_h'] * ratio_o))
        p_l = last['昨收'] * (1 + (reg['i_l'] + reg['s_l'] * ratio_o))
        st.table(pd.DataFrame({"场景": ["看空失效", "中性回归", "支撑测试"], "压力参考": [p_h*1.06, p_h, p_h*0.94], "支撑参考": [p_l*1.06, p_l, p_l*0.94]}).style.format(precision=2))

    # 主图
    st.divider()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(40).copy()
    p_df['label'] = pd.to_datetime(p_df.index).strftime('%m-%d')
    
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Upper'], line=dict(color='rgba(0,102,204,0.3)'), name=f"High:{last['Upper']:.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Lower'], line=dict(color='rgba(0,102,204,0.3)'), fill='tonexty', name=f"Low:{last['Lower']:.2f}"), row=1, col=1)
    fig.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], line=dict(color='#FF9800'), name=f"MA5:{last['MA5']:.2f}"), row=1, col=1)
    
    colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]
    fig.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_raw']*100, marker_color=colors, name="换手%"), row=2, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
    fig.update_xaxes(type='category', tickmode='linear', dtick=1, tickangle=-90)
    st.plotly_chart(fig, use_container_width=True)

    # 期权与大宗
    st.divider()
    o_col, d_col = st.columns(2)
    with o_col:
        st.subheader(f"🕯️ 全景期权 (到期:{mkt['exp']})")
        t1, t2 = st.tabs(["📈 看涨 (Calls)", "📉 看跌 (Puts)"])
        with t1: 
            if not calls_df.empty: st.dataframe(calls_df[['strike','lastPrice','openInterest','impliedVolatility']].style.format({'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}), use_container_width=True)
            else: st.info("无数据")
        with t2: 
            if not puts_df.empty: st.dataframe(puts_df[['strike','lastPrice','openInterest','impliedVolatility']].style.format({'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}), use_container_width=True)
            else: st.info("无数据")
            
    with d_col:
        st.subheader("🌑 大宗打印 (Dark Pool)")
        if not dark_df.empty: st.table(dark_df[['Volume', 'Signal']])
        else: st.info("近期无显著异动")

    # 历史明细
    st.subheader("📋 历史明细")
    hist_show = hist_df.tail(15).copy()
    hist_show['换手'] = (hist_show['换手率_raw'] * 100).map('{:.2f}%'.format)
    st.dataframe(hist_show[['Open','High','Low','Close','换手','MFI','MA20','MA5']].style.format(precision=2), use_container_width=True)

    # 自动刷新逻辑
    if auto_refresh:
        time.sleep(300) # 5 分钟无感更新
        st.rerun()
