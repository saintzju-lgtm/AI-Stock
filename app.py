import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 0. 授权验证 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if not st.session_state.password_correct:
        st.set_page_config(layout="wide", page_title="BTDR Quant Pro")
        st.title("🎯 BTDR 专业量化决策终端")
        pwd = st.text_input("输入访问码", type="password")
        if st.button("进入系统"):
            if pwd == st.secrets.get("ACCESS_PASSWORD", "123456"):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("访问受限")
        st.stop()
        return False
    return True

# --- 1. 增强版量化引擎 ---
@st.cache_data(ttl=3600)
def get_pro_market_data():
    try:
        time.sleep(1) # 频率保护
        tk = yf.Ticker("BTDR")
        info = tk.info
        hist = tk.history(period="100d", interval="1d")
        
        # 1. 市场锚点
        btc = yf.Ticker("BTC-USD").fast_info['last_price']
        nasdaq = yf.Ticker("^IXIC").fast_info['last_price']
        nasdaq_pct = (nasdaq / yf.Ticker("^IXIC").fast_info['previous_close'] - 1)
        
        # 2. 期权链数据 (获取最近到期日)
        exp_dates = tk.options
        opt_chain = None
        if exp_dates:
            opt_chain = tk.option_chain(exp_dates[0])
        
        # 3. 基础处理与 BOLL/MFI (保持原逻辑)
        current_float = info.get('floatShares') or info.get('shares') or 118500000
        rt_v = info.get('regularMarketVolume', 0)
        hist.index = hist.index.date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['Upper'] = hist['MA20'] + (hist['Close'].rolling(20).std() * 2)
        hist['Lower'] = hist['MA20'] - (hist['Close'].rolling(20).std() * 2)
        
        tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
        rmf = tp * hist['Volume']
        pos_flow = np.where(tp > tp.shift(1), rmf, 0)
        neg_flow = np.where(tp < tp.shift(1), rmf, 0)
        mfr = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
        hist['MFI'] = 100 - (100 / (1 + mfr.values))

        # 4. 暗池/大宗交易模拟 (识别成交量激增点)
        avg_vol = hist['Volume'].mean()
        dark_prints = hist[hist['Volume'] > avg_vol * 1.5].tail(5).copy()
        dark_prints['Signal'] = dark_prints.apply(lambda x: "机构吸筹" if x['Close'] > x['Open'] else "大宗派发", axis=1)

        # 5. 回归模型
        hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
        fit_df = hist.dropna()
        X = fit_df[['今开比例']].values
        m_h = LinearRegression().fit(X, fit_df['High'].values / fit_df['昨收'].values - 1)
        m_l = LinearRegression().fit(X, fit_df['Low'].values / fit_df['昨收'].values - 1)
        reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}

        return hist, reg_params, opt_chain, dark_prints, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct, 'float': current_float, 'volume': rt_v, 'exp': exp_dates[0] if exp_dates else "N/A"
        }
    except:
        return None

# --- 2. 界面渲染 ---
if check_password():
    st.markdown("""<style> .main { background-color: #FFFFFF !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; } 
    .darkpool-box { background-color: #F4F4F9; padding: 10px; border-radius: 5px; border-left: 5px solid #1A237E; } </style>""", unsafe_allow_html=True)

    data = get_pro_market_data()
    
    if data:
        hist_df, reg, opt, dark, mkt = data
        last_h = hist_df.iloc[-1]
        curr_p = last_h['Close']
        
        # --- 顶部大盘 (原逻辑) ---
        st.subheader("🌐 全球市场锚点")
        m1, m2, m3 = st.columns(3)
        m1.metric("Bitcoin (BTC)", f"${mkt['btc']:,.0f}")
        m2.metric("Nasdaq Index", f"{mkt['nasdaq']:,.2f}", f"{mkt['nasdaq_pct']:.2%}")
        m3.metric("BTDR 现价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")

        st.divider()

        # --- 看板与回归 (原逻辑) ---
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("📊 核心指标")
            st.write(f"实时换手: **{(mkt['volume']/mkt['float'])*100:.2f}%**")
            st.write(f"BOLL 高/低: **{last_h['Upper']:.2f} / {last_h['Lower']:.2f}**")
            st.write(f"资金 MFI: **{last_h['MFI']:.2f}**")
        with c2:
            ratio_o = (last_h['Open'] - last_h['昨收']) / last_h['昨收']
            p_h = last_h['昨收'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
            p_l = last_h['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))
            st.subheader("📍 场景回归预测")
            st.table(pd.DataFrame({"场景": ["看空失效", "中性回归", "支撑测试"], "压力参考": [p_h*1.06, p_h, p_h*0.94], "支撑参考": [p_l*1.06, p_l, p_l*0.94]}).style.format(precision=2))

        # --- 走势主图 (原逻辑) ---
        st.divider()
        fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        p_df = hist_df.tail(40).copy()
        fig_k.add_trace(go.Scatter(x=p_df.index, y=p_df['Upper'], line=dict(color='rgba(0,102,204,0.4)'), name="BOLL High"), row=1, col=1)
        fig_k.add_trace(go.Scatter(x=p_df.index, y=p_df['Lower'], line=dict(color='rgba(0,102,204,0.4)'), fill='tonexty', fillcolor='rgba(0,102,204,0.1)', name="BOLL Low"), row=1, col=1)
        fig_k.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
        fig_k.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name="MA5", line=dict(color='#FF9800', width=2)), row=1, col=1)
        fig_k.add_trace(go.Bar(x=p_df.index, y=(p_df['Volume']/mkt['float'])*100, name="换手率%", marker_color='gray'), row=2, col=1)
        fig_k.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig_k, use_container_width=True)

        # --- 新增模块：期权链 & 暗池流向 ---
        st.divider()
        o_col, d_col = st.columns(2)
        
        with o_col:
            st.subheader(f"🕯️ 近期期权链 (到期: {mkt['exp']})")
            if opt is not None:
                calls = opt.calls[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']].tail(5)
                calls.columns = ['行权价', '最新价', '未平仓', '隐波']
                st.write("**看涨期权 (Calls) 核心档位**")
                st.dataframe(calls.style.format({'隐波': '{:.2%}'}), use_container_width=True)
            else:
                st.info("当前无可用期权数据")

        with d_col:
            st.subheader("🌑 暗池/大宗交易监测 (模拟)")
            if not dark.empty:
                dark_show = dark[['Volume', 'Signal']].copy()
                dark_show.columns = ['成交量', '流向性质']
                st.write("**近期异常大宗异动打印 (Dark Pool Print)**")
                st.table(dark_show)
                st.caption("注：基于成交量异常偏离均值 150% 以上的机构行为识别。")
            else:
                st.info("近期无显著大宗异动")

        # --- 历史明细 (原逻辑) ---
        st.divider()
        st.subheader("📋 历史数据明细")
        show_df = hist_df.tail(10).copy()
        show_df['换手'] = ((show_df['Volume']/mkt['float'])*100).map('{:.2f}%'.format)
        st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手', 'MFI']].style.format(precision=2), use_container_width=True)

    else:
        st.error("数据接口被封锁，请检查网络或更换 VPN 节点。")
