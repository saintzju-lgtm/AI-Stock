import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
from io import BytesIO

# --- 1. 页面与样式配置 ---
st.set_page_config(page_title="A股全能量化(免Token修复版)", layout="wide", page_icon="📈")

# 注入 CSS 优化界面
st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    .stProgress > div > div > div > div { background-color: #00cc96; }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-low { color: #00cc96; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心数据层 (Akshare 缓存优化) ---

@st.cache_data(ttl=600)
def get_realtime_market_data():
    """获取全市场实时行情 (用于选股扫描)"""
    try:
        df = ak.stock_zh_a_spot_em()
        rename_dict = {
            "代码": "symbol", "名称": "name", "最新价": "price", 
            "涨跌幅": "change_pct", "市盈率-动态": "pe", "市净率": "pb",
            "换手率": "turnover", "总市值": "market_cap", "所处行业": "industry",
            "量比": "volume_ratio"
        }
        df = df.rename(columns=rename_dict)
        
        # --- 修复点1：区分文本和数字的初始化 ---
        str_cols = ['symbol', 'name', 'industry']
        for col in rename_dict.values():
            if col not in df.columns:
                # 如果是文本列，填空字符串；如果是数字列，填0
                df[col] = "" if col in str_cols else 0
            
        # 数值转换 (强制清洗非数字字符)
        numeric_cols = ['price', 'change_pct', 'pe', 'pb', 'turnover', 'market_cap', 'volume_ratio']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"行情接口异常: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_stock_history(symbol, days=365):
    """获取个股历史K线 (用于技术分析与回测)"""
    try:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"})
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return pd.DataFrame()

# --- 3. 策略与计算引擎 ---

def calculate_indicators(df):
    """计算 MACD, RSI"""
    if df.empty: return df
    
    # MA
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def run_simple_backtest(df):
    """简易回测：双均线策略"""
    if df.empty: return 0, pd.DataFrame()
    
    df['signal'] = np.where(df['MA5'] > df['MA20'], 1, 0)
    df['pct_change'] = df['close'].pct_change()
    df['strategy_return'] = df['pct_change'] * df['signal'].shift(1)
    
    df['cum_return'] = (1 + df['strategy_return']).cumprod()
    df['benchmark'] = (1 + df['pct_change']).cumprod()
    
    total_return = (df['cum_return'].iloc[-1] - 1) * 100 if not df['cum_return'].isnull().all() else 0
    return total_return, df

def ai_diagnosis(row, hist_df):
    """五维诊断模型"""
    scores = {}
    
    pe = row['pe']
    scores['估值'] = 100 - min(pe, 100) if pe > 0 else 40
    
    if not hist_df.empty:
        curr = hist_df.iloc[-1]
        trend_score = 50
        if curr['close'] > curr['MA20']: trend_score += 20
        if curr['MA5'] > curr['MA20']: trend_score += 30
        scores['趋势'] = trend_score
    else:
        scores['趋势'] = 0
        
    to = row.get('turnover', 0)
    scores['资金'] = min(to * 10, 100)
    
    if not hist_df.empty and 'RSI' in hist_df.columns:
        rsi = hist_df.iloc[-1]['RSI'] if not pd.isna(hist_df.iloc[-1]['RSI']) else 50
        scores['动量'] = 100 - abs(50 - rsi) * 2 
    else:
        scores['动量'] = 50
        
    pct = row['change_pct']
    scores['情绪'] = 50 + pct * 5 
    
    total_score = sum(scores.values()) / 5
    return total_score, scores

# --- 4. 界面交互逻辑 ---

# Sidebar
with st.sidebar:
    st.title("🎛️ 控制面板")
    mode = st.radio("使用模式", ["新手模式 (开箱即用)", "专业模式 (自定义参数)"])
    st.markdown("---")
    if mode == "专业模式 (自定义参数)":
        pe_range = st.slider("PE范围", 0, 200, (0, 60))
        min_mkt_cap = st.number_input("最小市值(亿)", 0, 1000, 50)
        show_backtest = st.checkbox("显示回测详情", True)
    else:
        pe_range = (0, 80)
        min_mkt_cap = 20
        show_backtest = False
        st.info("💡 新手模式：已自动过滤高风险股，隐藏复杂参数。")

# Main Area
st.title("🚀 A股智能量化决策系统")
st.markdown("功能全覆盖：**选股 + 诊断 + 回测 + 导出** | 数据源：**Akshare (无Token)**")

# 1. 行业扫描
st.subheader("1. 热门赛道扫描")
col1, col2, col3, col4, col5 = st.columns(5)
sectors = {"新能源": ["光伏", "锂", "能", "隆基"], 
           "半导体": ["芯", "微", "韦尔", "紫光"], 
           "消费": ["酒", "乳", "免税", "茅台"], 
           "数字经济": ["软件", "云", "算力", "浪潮"],
           "医药": ["药", "医", "恒瑞", "迈瑞"]}

if 'sector' not in st.session_state: st.session_state.sector = "新能源"

for i, (name, kw) in enumerate(sectors.items()):
    with [col1, col2, col3, col4, col5][i]:
        if st.button(f"{name}", use_container_width=True):
            st.session_state.sector = name

# 执行筛选
all_data = get_realtime_market_data()
if not all_data.empty:
    keywords = sectors[st.session_state.sector]
    
    # --- 修复点2：稳健的字符串筛选逻辑 ---
    # 强制将列转为字符串类型后再匹配，防止因数据类型混杂导致的 AttributeError
    mask_name = all_data['name'].astype(str).str.contains('|'.join(keywords), na=False)
    mask_ind = all_data['industry'].astype(str).str.contains(st.session_state.sector, na=False)
    
    df_sector = all_data[mask_name | mask_ind].copy()
    
    # 过滤与排序
    df_final = df_sector[
        (df_sector['pe'] >= pe_range[0]) & 
        (df_sector['pe'] <= pe_range[1]) &
        (df_sector['market_cap'] > min_mkt_cap * 100000000)
    ].sort_values('change_pct', ascending=False).head(10)
    
    st.subheader(f"2. {st.session_state.sector} 精选结果与 AI 诊断")
    
    # 导出
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("📥 导出当前结果", csv, "stock_results.csv", "text/csv")
    
    if df_final.empty:
        st.warning("当前筛选条件下无结果，请尝试切换行业或在专业模式调整参数。")
    
    for idx, row in df_final.iterrows():
        hist_df = get_stock_history(row['symbol'])
        hist_df = calculate_indicators(hist_df)
        score, dimensions = ai_diagnosis(row, hist_df)
        
        risk_level = "高" if score < 40 else ("中" if score < 70 else "低")
        risk_color = "red" if risk_level == "高" else ("orange" if risk_level == "中" else "green")
        
        with st.expander(f"{row['name']} ({row['symbol']}) | 评分: {score:.0f} | 风险: {risk_level}", expanded=False):
            c1, c2, c3 = st.columns([1.5, 1.5, 1])
            
            with c1:
                st.markdown("#### 🔍 核心数据")
                st.write(f"**最新价**: {row['price']} (涨幅 {row['change_pct']}%)")
                st.write(f"**PE(动)**: {row['pe']} | **PB**: {row['pb']}")
                st.write(f"**换手率**: {row['turnover']}% | **量比**: {row['volume_ratio']}")
                
                advice = "观望"
                if score > 75 and row['change_pct'] < 5: advice = "建议关注 (优质且未暴涨)"
                elif score > 60: advice = "持有/观察"
                elif score < 40: advice = "回避/卖出"
                st.info(f"💡 **AI 建议**: {advice}")

            with c2:
                st.markdown("#### 🕸️ 多维诊断")
                radar_data = pd.DataFrame(dict(r=list(dimensions.values()), theta=list(dimensions.keys())))
                fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True)
                fig_radar.update_traces(fill='toself')
                fig_radar.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_radar, use_container_width=True)
                
            with c3:
                if not hist_df.empty:
                    ret, res_df = run_simple_backtest(hist_df)
                    st.markdown("#### 🔙 历史回测 (1年)")
                    st.metric("双均线策略收益", f"{ret:.1f}%", delta=f"{ret - (hist_df['close'].iloc[-1]/hist_df['close'].iloc[0]-1)*100:.1f}% vs 基准")
                    st.line_chart(res_df[['cum_return', 'benchmark']], height=150)
            
            if mode == "专业模式 (自定义参数)" or show_backtest:
                st.markdown("#### 📈 技术走势 (含 MACD/RSI)")
                if not hist_df.empty:
                    fig_k = go.Figure()
                    fig_k.add_trace(go.Candlestick(x=hist_df['date'], open=hist_df['open'], high=hist_df['high'], low=hist_df['low'], close=hist_df['close'], name='K线'))
                    fig_k.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['MA20'], line=dict(color='orange'), name='MA20'))
                    fig_k.update_layout(height=350, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_k, use_container_width=True)

else:
    st.error("数据加载失败，请检查网络。")

if 'first_visit' not in st.session_state:
    st.toast("🔰 新手模式已开启：只显示最核心的选股结果与建议！")
    st.session_state.first_visit = False
