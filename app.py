import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# --- 0. 页面全局配置 (必须放在第一行) ---
st.set_page_config(layout="wide", page_title="专业量化决策终端")

# --- 1. 核心量化引擎 (缓存改为 60 秒，配合 1 分钟自动刷新) ---
@st.cache_data(ttl=60)
def get_enhanced_market_data(ticker_symbol):
    try:
        time.sleep(1) # 模拟延迟，防止高频触发封锁
        tk = yf.Ticker(ticker_symbol)
        info = tk.info
        hist = tk.history(period="100d", interval="1d")
        
        if hist.empty: 
            return "股票历史数据获取为空，请检查股票代码是否正确或网络状况。"

        # --- 容错获取宏观数据 ---
        def safe_get_macro(ticker_sym):
            try:
                t = yf.Ticker(ticker_sym)
                last_p = t.fast_info['last_price']
                prev_p = t.fast_info['previous_close']
                return last_p, (last_p / prev_p - 1)
            except:
                return 0.0, 0.0 

        btc, _ = safe_get_macro("BTC-USD")
        nasdaq, nasdaq_pct = safe_get_macro("^IXIC")
        vix, vix_pct = safe_get_macro("^VIX")

        # --- 容错获取期权链 ---
        calls_df, puts_df = pd.DataFrame(), pd.DataFrame()
        current_exp = "N/A"
        try:
            exp_dates = tk.options
            if exp_dates:
                today_str = datetime.now().strftime('%Y-%m-%d')
                if exp_dates[0] <= today_str and len(exp_dates) > 1:
                    current_exp = exp_dates[1] 
                else:
                    current_exp = exp_dates[0]
                    
                curr_p = hist['Close'].iloc[-1]
                opt_data = tk.option_chain(current_exp)
                
                # Calls 切片
                all_calls = opt_data.calls
                atm_idx_c = (all_calls['strike'] - curr_p).abs().idxmin()
                calls_df = all_calls.iloc[max(0, atm_idx_c - 4) : min(len(all_calls), atm_idx_c + 5)]
                
                # Puts 切片
                all_puts = opt_data.puts
                atm_idx_p = (all_puts['strike'] - curr_p).abs().idxmin()
                puts_df = all_puts.iloc[max(0, atm_idx_p - 4) : min(len(all_puts), atm_idx_p + 5)]
        except Exception as e:
            pass 

        # 3. 基础处理与技术指标
        current_float = info.get('floatShares') or info.get('shares') or 118500000
        rt_v = info.get('regularMarketVolume', 0)
        
        hist.index = hist.index.date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['Upper'] = hist['MA20'] + (hist['Close'].rolling(20).std() * 2)
        hist['Lower'] = hist['MA20'] - (hist['Close'].rolling(20).std() * 2)
        hist['换手率_raw'] = (hist['Volume'] / current_float)
        
        # MFI 资金流
        tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
        rmf = tp * hist['Volume']
        pos_flow = np.where(tp > tp.shift(1), rmf, 0)
        neg_flow = np.where(tp < tp.shift(1), rmf, 0)
        mfr = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
        hist['MFI'] = 100 - (100 / (1 + mfr.values))

        # 4. 暗池/大宗打印 (1.2倍均量偏离识别)
        avg_vol = hist['Volume'].mean()
        dark_prints = hist[hist['Volume'] > avg_vol * 1.2].tail(8).copy()
        dark_prints['Signal'] = dark_prints.apply(lambda x: "机构吸筹" if x['Close'] > x['Open'] else "大宗派发", axis=1)

        # 5. 回归模型
        hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
        fit_df = hist.dropna()
        X = fit_df[['今开比例']].values
        m_h = LinearRegression().fit(X, fit_df['High'].values / fit_df['昨收'].values - 1)
        m_l = LinearRegression().fit(X, fit_df['Low'].values / fit_df['昨收'].values - 1)
        reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}

        return hist, reg_params, calls_df, puts_df, dark_prints, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct, 
            'vix': vix, 'vix_pct': vix_pct,
            'float': current_float, 'volume': rt_v, 'exp': current_exp
        }
    except Exception as e:
        return f"系统核心异常: {str(e)}"

# --- 2. UI 渲染 (免密码直接加载) ---
st.markdown("""<style> .main { background-color: #FFFFFF !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; padding-bottom: 5px; } </style>""", unsafe_allow_html=True)

# 侧边栏全局配置
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = "BTDR"

with st.sidebar:
    st.markdown("### 添加新股票")
    new_ticker = st.text_input("", value=st.session_state.current_ticker, label_visibility="collapsed")
    if new_ticker and new_ticker.upper() != st.session_state.current_ticker:
        st.session_state.current_ticker = new_ticker.upper()
        st.rerun() # 更改代码后立即刷新
    
    st.divider()
    # 默认开启 1 分钟自动刷新
    auto_refresh = st.checkbox("开启 1分钟自动无感刷新", value=True)

ticker = st.session_state.current_ticker
st.title(f"🎯 {ticker} 专业量化决策终端")

# 传入动态 Ticker
data = get_enhanced_market_data(ticker)

if isinstance(data, str):
    st.error(f"⚠️ 数据加载失败。详细错误: {data}")
elif data:
    hist_df, reg, calls_df, puts_df, dark_df, mkt = data
    last_h = hist_df.iloc[-1]
    curr_p = last_h['Close']

    # --- 第一阶段：大盘与宏观 ---
    st.subheader("🌐 宏观风险防御看板")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bitcoin (BTC)", f"${mkt['btc']:,.0f}" if mkt['btc'] > 0 else "N/A")
    m2.metric("Nasdaq Index", f"{mkt['nasdaq']:,.2f}" if mkt['nasdaq'] > 0 else "N/A", f"{mkt['nasdaq_pct']:.2%}")
    m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}" if mkt['vix'] > 0 else "N/A", f"{mkt['vix_pct']:.2%}", delta_color="inverse")
    m4.metric(f"{ticker} 现价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")

    st.divider()

    # --- 第二阶段：指标与回归 ---
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
        st.table(pd.DataFrame({
            "场景": ["看空失效", "中性回归", "支撑测试"],
            "压力参考": [p_h*1.06, p_h, p_h*0.94],
            "支撑参考": [p_l*1.06, p_l, p_l*0.94]
        }).style.format(precision=2))

    # --- 第三阶段：走势主图 ---
    st.divider()
    st.subheader("🕒 走势主图 (图例集成实时数值)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(40).copy()
    
    p_df['label'] = pd.to_datetime(p_df.index).strftime('%m-%d')
    
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['Upper'], line=dict(color='rgba(0,102,204,0.5)', width=1.5), name=f"BOLL High: {last_h['Upper']:.2f}"), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['Lower'], line=dict(color='rgba(0,102,204,0.5)', width=1.5), fill='tonexty', fillcolor='rgba(0,102,204,0.1)', name=f"BOLL Low: {last_h['Lower']:.2f}"), row=1, col=1)
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name=f"MA5: {last_h['MA5']:.2f}", line=dict(color='#FF9800', width=2)), row=1, col=1)
    
    vol_colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_raw']*100, name="换手率%", marker_color=vol_colors), row=2, col=1)
    
    fig_k.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_k.update_xaxes(type='category', tickmode='linear', dtick=1, tickangle=-90)
    st.plotly_chart(fig_k, use_container_width=True)

    # --- 第四阶段：期权与大宗 ---
    st.divider()
    o_col, d_col = st.columns(2)
    with o_col:
        st.subheader(f"🕯️ 全景期权链 (到期: {mkt['exp']})")
        tab1, tab2 = st.tabs(["📈 看涨 (Calls)", "📉 看跌 (Puts)"])
        with tab1:
            if not calls_df.empty:
                display_calls = calls_df[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']].copy()
                display_calls.columns = ['行权价', '最新价', '未平仓', '隐波']
                st.dataframe(display_calls.style.format({'隐波': '{:.2%}', '最新价': '{:.2f}', '行权价': '{:.2f}', '未平仓': '{:,.0f}'}), use_container_width=True)
            else:
                st.info("当前无看涨期权数据")
        with tab2:
            if not puts_df.empty:
                display_puts = puts_df[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']].copy()
                display_puts.columns = ['行权价', '最新价', '未平仓', '隐波']
                st.dataframe(display_puts.style.format({'隐波': '{:.2%}', '最新价': '{:.2f}', '行权价': '{:.2f}', '未平仓': '{:,.0f}'}), use_container_width=True)
            else:
                st.info("当前无看跌期权数据")
            
    with d_col:
        st.subheader("🌑 大宗异动打印 (Dark Pool Print)")
        if not dark_df.empty:
            dark_show = dark_df[['Volume', 'Signal']].copy()
            dark_show.columns = ['成交量', '流向性质']
            st.table(dark_show)
        else:
            st.info("近期无显著大宗异动")

    # --- 第五阶段：历史明细 ---
    st.subheader("📋 历史明细 (集成 MFI)")
    hist_show = hist_df.tail(15).copy()
    hist_show['换手'] = (hist_show['换手率_raw'] * 100).map('{:.2f}%'.format)
    
    cols_to_show = ['Open', 'High', 'Low', 'Close', '换手', 'MFI', 'MA20', 'MA5']
    st.dataframe(hist_show[cols_to_show].style.format(
        subset=['Open', 'High', 'Low', 'Close', 'MFI', 'MA20', 'MA5'], precision=2
    ), use_container_width=True)

    # --- 自动刷新控制逻辑 ---
    if auto_refresh:
        time.sleep(60) # 等待 60 秒
        st.rerun()     # 自动重新运行脚本实现无感刷新

else:
    st.error("⚠️ 发生未知错误，数据返回为空。")
