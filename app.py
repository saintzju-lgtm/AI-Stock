import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

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

# --- 1. 核心量化引擎 ---
@st.cache_data(ttl=3600)
def get_enhanced_market_data():
    try:
        time.sleep(1.5) 
        tk = yf.Ticker("BTDR")
        info = tk.info
        hist = tk.history(period="100d", interval="1d")
        if hist.empty: return None

        # 1. 抓取宏观与锚点数据
        btc = yf.Ticker("BTC-USD").fast_info['last_price']
        nas_tk = yf.Ticker("^IXIC")
        nasdaq = nas_tk.fast_info['last_price']
        nasdaq_pct = (nasdaq / nas_tk.fast_info['previous_close'] - 1)
        
        vix_tk = yf.Ticker("^VIX")
        vix = vix_tk.fast_info['last_price']
        vix_pct = (vix / vix_tk.fast_info['previous_close'] - 1)

        # 2. 期权链处理 (优化：自动避开到期日归零合约)
        exp_dates = tk.options
        calls_df = pd.DataFrame()
        current_exp = "N/A"
        
        if exp_dates:
            # 逻辑：如果最近的到期日是今天，则抓取下一个到期日以获得有效数据
            today_str = datetime.now().strftime('%Y-%m-%d')
            if exp_dates[0] <= today_str and len(exp_dates) > 1:
                current_exp = exp_dates[1] # 取下周或下个月
            else:
                current_exp = exp_dates[0]
                
            try:
                curr_p = hist['Close'].iloc[-1]
                opt_data = tk.option_chain(current_exp)
                all_calls = opt_data.calls
                # ATM 中心化切片
                atm_idx = (all_calls['strike'] - curr_p).abs().idxmin()
                start_idx = max(0, atm_idx - 4)
                end_idx = min(len(all_calls), atm_idx + 5)
                calls_df = all_calls.iloc[start_idx:end_idx]
            except: pass

        # 3. 基础指标计算
        current_float = info.get('floatShares') or info.get('shares') or 118500000
        rt_v = info.get('regularMarketVolume', 0)
        
        hist.index = hist.index.date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['Upper'] = hist['MA20'] + (hist['Close'].rolling(20).std() * 2)
        hist['Lower'] = hist['MA20'] - (hist['Close'].rolling(20).std() * 2)
        hist['换手率_raw'] = (hist['Volume'] / current_float)
        
        # MFI 计算
        tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
        rmf = tp * hist['Volume']
        pos_flow = np.where(tp > tp.shift(1), rmf, 0)
        neg_flow = np.where(tp < tp.shift(1), rmf, 0)
        mfr = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
        hist['MFI'] = 100 - (100 / (1 + mfr.values))

        # 4. 暗池/大宗打印 (1.2倍偏离)
        avg_vol = hist['Volume'].mean()
        dark_prints = hist[hist['Volume'] > avg_vol * 1.2].tail(8).copy()
        dark_prints['Signal'] = dark_prints.apply(lambda x: "机构吸筹" if x['Close'] > x['Open'] else "大宗派发", axis=1)

        # 5. 场景回归模型
        hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
        fit_df = hist.dropna()
        X = fit_df[['今开比例']].values
        m_h = LinearRegression().fit(X, fit_df['High'].values / fit_df['昨收'].values - 1)
        m_l = LinearRegression().fit(X, fit_df['Low'].values / fit_df['昨收'].values - 1)
        reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}

        return hist, reg_params, calls_df, dark_prints, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct, 
            'vix': vix, 'vix_pct': vix_pct,
            'float': current_float, 'volume': rt_v, 'exp': current_exp
        }
    except:
        return None

# --- 2. UI 渲染 ---
if check_password():
    st.markdown("""<style> .main { background-color: #FFFFFF !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; padding-bottom: 5px; } </style>""", unsafe_allow_html=True)

    data = get_enhanced_market_data()
    
    if data:
        hist_df, reg, calls_df, dark_df, mkt = data
        last_h = hist_df.iloc[-1]
        curr_p = last_h['Close']

        # 🌐 全球市场锚点
        st.subheader("🌐 宏观风险防御看板")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bitcoin (BTC)", f"${mkt['btc']:,.0f}")
        m2.metric("Nasdaq Index", f"{mkt['nasdaq']:,.2f}", f"{mkt['nasdaq_pct']:.2%}")
        m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}", f"{mkt['vix_pct']:.2%}", delta_color="inverse")
        m4.metric("BTDR 现价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")

        st.divider()

        # 看板与回归
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("📊 实时指标")
            st.write(f"实时换手: **{(mkt['volume']/mkt['float'])*100:.2f}%**")
            st.write(f"BOLL 高/低: **{last_h['Upper']:.2f} / {last_h['Lower']:.2f}**")
            st.write(f"资金 MFI: **{last_h['MFI']:.2f}**")
        with c2:
            st.subheader("📍 场景回归预测")
            ratio_o = (last_h['Open'] - last_h['昨收']) / last_h['昨收']
            p_h = last_h['昨收'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
            p_l = last_h['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))
            st.table(pd.DataFrame({"场景": ["看空失效", "中性回归", "支撑测试"], "压力参考": [p_h*1.06, p_h, p_h*0.94], "支撑参考": [p_l*1.06, p_l, p_l*0.94]}).style.format(precision=2))

        # 走势主图
        st.divider()
        fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        p_df = hist_df.tail(40).copy()
        fig_k.add_trace(go.Scatter(x=p_df.index, y=p_df['Upper'], line=dict(color='rgba(0,102,204,0.5)', width=1.5), name=f"BOLL High: {last_h['Upper']:.2f}"), row=1, col=1)
        fig_k.add_trace(go.Scatter(x=p_df.index, y=p_df['Lower'], line=dict(color='rgba(0,102,204,0.5)', width=1.5), fill='tonexty', fillcolor='rgba(0,102,204,0.1)', name=f"BOLL Low: {last_h['Lower']:.2f}"), row=1, col=1)
        fig_k.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
        fig_k.add_trace(go.Scatter(x=p_df.index, y=p_df['MA5'], name=f"MA5: {last_h['MA5']:.2f}", line=dict(color='#FF9800', width=2)), row=1, col=1)
        
        vol_colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]
        fig_k.add_trace(go.Bar(x=p_df.index, y=p_df['换手率_raw']*100, name="换手率%", marker_color=vol_colors), row=2, col=1)
        fig_k.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_k, use_container_width=True)

        # 期权链与暗池 (显示非归零合约)
        st.divider()
        o_col, d_col = st.columns(2)
        with o_col:
            st.subheader(f"🕯️ 活跃期权链 (到期: {mkt['exp']})")
            if not calls_df.empty:
                display_calls = calls_df[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']]
                display_calls.columns = ['行权价', '最新价', '未平仓', '隐波']
                st.dataframe(display_calls.style.format({'隐波': '{:.2%}', '最新价': '{:.2f}', '行权价': '{:.2f}'}), use_container_width=True)
            else:
                st.info("当前无可用期权数据")
        with d_col:
            st.subheader("🌑 大宗异动打印 (Dark Pool Print)")
            if not dark_df.empty:
                dark_show = dark_df[['Volume', 'Signal']].copy()
                dark_show.columns = ['成交量', '流向性质']
                st.table(dark_show)
            else:
                st.info("近期无显著大宗异动")

        # 历史明细
        st.subheader("📋 历史明细 (集成 MFI)")
        hist_show = hist_df.tail(15).copy()
        hist_show['换手'] = (hist_show['换手率_raw'] * 100).map('{:.2f}%'.format)
        st.dataframe(hist_show[['Open', 'High', 'Low', 'Close', '换手', 'MFI', 'MA20']].style.format(precision=2), use_container_width=True)

    else:
        st.error("⚠️ 数据加载失败。请更换 VPN 节点或刷新页面。")
