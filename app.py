# app.py
import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
import datetime

# --- 页面配置 ---
st.set_page_config(page_title="A股极简量化(免Token版)", layout="wide", page_icon="⚡")

# --- 核心数据层 (基于 Akshare，无需 Token) ---

@st.cache_data(ttl=300)  # 缓存5分钟，避免频繁爬取被封IP
def get_realtime_market_data():
    """
    获取东方财富 A 股实时行情
    包含：代码, 名称, 最新价, 涨跌幅, 市盈率(动态), 换手率, 总市值
    """
    try:
        # 获取 A 股实时行情
        df = ak.stock_zh_a_spot_em()
        
        # 数据清洗与重命名，方便后续使用
        df = df.rename(columns={
            "代码": "symbol", "名称": "name", "最新价": "price", 
            "涨跌幅": "change_pct", "市盈率-动态": "pe", 
            "换手率": "turnover", "总市值": "market_cap",
            "所处行业": "industry"
        })
        
        # 转换数值类型
        numeric_cols = ['price', 'change_pct', 'pe', 'turnover', 'market_cap']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"数据获取失败 (网络原因或接口调整): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_history(symbol):
    """获取个股历史K线数据 (用于计算买卖点)"""
    try:
        # 东方财富接口需要纯数字代码
        start_date = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y%m%d")
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        
        # qfq = 前复权
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"})
        return df
    except Exception:
        return pd.DataFrame()

# --- 策略逻辑层 (直接生成结论) ---

def analyze_stock(symbol, name, current_price):
    """
    对单只股票进行快速诊断，生成买卖建议
    """
    hist_df = get_stock_history(symbol)
    if hist_df.empty or len(hist_df) < 20:
        return "数据不足", "观望", 0, None

    # 计算均线
    hist_df['MA5'] = hist_df['close'].rolling(5).mean()
    hist_df['MA10'] = hist_df['close'].rolling(10).mean()
    hist_df['MA20'] = hist_df['close'].rolling(20).mean()
    
    curr = hist_df.iloc[-1]
    prev = hist_df.iloc[-2]
    
    # --- 核心判断逻辑 ---
    score = 60 # 基础分
    advice = "观望"
    reason = "趋势不明显"
    
    # 1. 金叉判断 (短线买点)
    if prev['MA5'] <= prev['MA10'] and curr['MA5'] > curr['MA10']:
        score += 20
        advice = "建议买入"
        reason = "5日线金叉10日线，短线启动"
    
    # 2. 多头排列 (趋势向上)
    elif curr['MA5'] > curr['MA10'] > curr['MA20']:
        score += 10
        advice = "持有/加仓"
        reason = "均线多头排列，上涨趋势中"
        
    # 3. 死叉判断 (卖点)
    elif prev['MA5'] >= prev['MA10'] and curr['MA5'] < curr['MA10']:
        score -= 20
        advice = "建议卖出"
        reason = "5日线死叉10日线，短线调整"
        
    # 4. 价格位置
    stop_loss = current_price * 0.95
    target_price = current_price * 1.1
    
    return advice, reason, score, stop_loss, hist_df

# --- UI 交互层 ---

st.title("⚡ A股实时机会扫描 (免Token版)")
st.markdown("数据来源：**Akshare (东方财富实时接口)** | 无需登录，开箱即用")

# 1. 自动全市场扫描
st.header("1. 热门行业一键选股")

col1, col2, col3, col4 = st.columns(4)
sector = None

# 使用 Streamlit 状态保持按下的按钮
if 'active_sector' not in st.session_state:
    st.session_state.active_sector = "新能源" # 默认

with col1:
    if st.button("🔋 新能源/光伏"): st.session_state.active_sector = "新能源"
with col2:
    if st.button("💻 半导体/芯片"): st.session_state.active_sector = "半导体"
with col3:
    if st.button("🍷 消费/白酒"): st.session_state.active_sector = "白酒"
with col4:
    if st.button("🤖 人工智能"): st.session_state.active_sector = "人工智能"

st.info(f"正在扫描 **{st.session_state.active_sector}** 板块... (实时从交易所获取数据)")

# 2. 获取全市场数据并筛选
all_data = get_realtime_market_data()

if not all_data.empty:
    # 简单的名称筛选模拟行业 (Akshare 也有专门行业接口，但这样最快且稳定)
    # 在真实场景中，可以使用 ak.stock_board_industry_cons_em(symbol="板块名称")
    
    keywords = {
        "新能源": ["光伏", "锂", "能", "宁德", "隆基", "通威"],
        "半导体": ["芯", "半导体", "微", "韦尔", "卓胜微"],
        "白酒": ["酒", "茅台", "五粮液"],
        "人工智能": ["智能", "AI", "科大", "三六零", "浪潮"]
    }
    
    filter_words = keywords.get(st.session_state.active_sector, [])
    
    # 模糊匹配筛选行业
    mask = all_data['name'].str.contains('|'.join(filter_words)) | all_data['industry'].str.contains(st.session_state.active_sector)
    sector_df = all_data[mask].copy()
    
    # 二次筛选：PE > 0 (剔除亏损), 涨幅 > -3% (剔除暴跌)
    valid_stocks = sector_df[(sector_df['pe'] > 0) & (sector_df['pe'] < 60) & (sector_df['change_pct'] > -2)].sort_values('change_pct', ascending=False).head(10)
    
    # 3. 逐个分析并展示结论
    st.header("2. 智能决策结论")
    
    for index, row in valid_stocks.iterrows():
        # 调用分析函数
        advice, reason, score, stop_loss, hist_df = analyze_stock(row['symbol'], row['name'], row['price'])
        
        # 颜色定义
        color = "red" if "买" in advice or "持有" in advice else "green"
        
        with st.expander(f"{row['name']} ({row['symbol']}) | 现价: {row['price']} | 涨幅: {row['change_pct']}% | 建议: {advice}"):
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown(f"### 🤖 AI 结论: :{color}[{advice}]")
                st.write(f"**核心理由**: {reason}")
                st.progress(score)
                st.caption(f"综合评分: {score} 分")
                
                st.markdown("---")
                st.markdown(f"**💰 操作点位**:")
                st.write(f"建议止损线: **{stop_loss:.2f}** 元")
                st.write(f"动态市盈率: {row['pe']}")
                
            with c2:
                if hist_df is not None and not hist_df.empty:
                    # 绘制K线图
                    fig = go.Figure(data=[go.Candlestick(x=hist_df['date'],
                                    open=hist_df['open'], high=hist_df['high'],
                                    low=hist_df['low'], close=hist_df['close'])])
                    # 添加均线
                    fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['MA5'], line=dict(color='orange', width=1), name='MA5'))
                    fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['MA20'], line=dict(color='blue', width=1), name='MA20'))
                    fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("暂无历史数据")

else:
    st.error("无法连接到行情服务器，请检查网络连接。")

# 4. 手动查询
st.markdown("---")
st.header("3. 单股快速诊断")
input_code = st.text_input("输入股票代码 (如 600519)", "")
if input_code:
    # 尝试从全市场数据中查找
    stock_info = all_data[all_data['symbol'] == input_code]
    if not stock_info.empty:
        row = stock_info.iloc[0]
        advice, reason, score, stop_loss, hist_df = analyze_stock(row['symbol'], row['name'], row['price'])
        st.info(f"诊断结果：{row['name']} - {advice} (评分 {score})")
        st.write(f"理由: {reason}")
    else:
        st.warning("未找到该代码，请输入6位数字代码。")
