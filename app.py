# app.py
import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
import datetime

# --- 页面配置 ---
st.set_page_config(page_title="A股极简量化(免Token版)", layout="wide", page_icon="⚡")

# --- 核心数据层 (基于 Akshare，无需 Token) ---

@st.cache_data(ttl=300)  # 缓存5分钟
def get_realtime_market_data():
    """
    获取东方财富 A 股实时行情
    """
    try:
        # 获取 A 股实时行情
        df = ak.stock_zh_a_spot_em()
        
        # 调试：打印一下列名，防止改名 (在后台终端可以看到)
        # print("接口返回的列名:", df.columns.tolist())
        
        # 定义重命名映射
        rename_dict = {
            "代码": "symbol", "名称": "name", "最新价": "price", 
            "涨跌幅": "change_pct", "市盈率-动态": "pe", 
            "换手率": "turnover", "总市值": "market_cap",
            "所处行业": "industry" # 尝试重命名行业，如果不存在也没关系
        }
        
        # 只重命名存在的列
        df = df.rename(columns=rename_dict)
        
        # 如果接口没返回 'industry'，我们手动补一个空列，防止后续报错
        if 'industry' not in df.columns:
            df['industry'] = ''
            
        # 转换数值类型 (确保计算时不报错)
        numeric_cols = ['price', 'change_pct', 'pe', 'turnover', 'market_cap']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"数据获取失败 (网络或接口变动): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_history(symbol):
    """获取个股历史K线数据"""
    try:
        # 计算日期
        start_date = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y%m%d")
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        
        # 获取历史行情
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"})
        return df
    except Exception:
        return pd.DataFrame()

# --- 策略逻辑层 ---

def analyze_stock(symbol, name, current_price):
    """生成买卖建议"""
    hist_df = get_stock_history(symbol)
    
    # 数据校验
    if hist_df.empty or len(hist_df) < 20:
        return "数据不足", "上市时间太短或数据获取失败", 0, 0, None

    # 计算均线
    hist_df['MA5'] = hist_df['close'].rolling(5).mean()
    hist_df['MA10'] = hist_df['close'].rolling(10).mean()
    hist_df['MA20'] = hist_df['close'].rolling(20).mean()
    
    curr = hist_df.iloc[-1]
    prev = hist_df.iloc[-2]
    
    # --- 核心判断逻辑 ---
    score = 60 
    advice = "观望"
    reason = "趋势不明显"
    
    # 1. 金叉 (短线买点)
    if prev['MA5'] <= prev['MA10'] and curr['MA5'] > curr['MA10']:
        score += 20
        advice = "建议买入"
        reason = "5日线金叉10日线，短线启动"
    
    # 2. 多头 (持有)
    elif curr['MA5'] > curr['MA10'] > curr['MA20']:
        score += 10
        advice = "持有/加仓"
        reason = "均线多头排列，上涨趋势中"
        
    # 3. 死叉 (卖点)
    elif prev['MA5'] >= prev['MA10'] and curr['MA5'] < curr['MA10']:
        score -= 20
        advice = "建议卖出"
        reason = "5日线死叉10日线，短线调整"
        
    stop_loss = current_price * 0.95
    
    return advice, reason, score, stop_loss, hist_df

# --- UI 交互层 ---

st.title("⚡ A股实时机会扫描 (修复版)")
st.markdown("数据来源：**Akshare** | 状态：**已修复 Key Error**")

# 1. 行业选择
st.header("1. 热门行业一键选股")

col1, col2, col3, col4 = st.columns(4)
if 'active_sector' not in st.session_state:
    st.session_state.active_sector = "新能源" 

with col1:
    if st.button("🔋 新能源/光伏"): st.session_state.active_sector = "新能源"
with col2:
    if st.button("💻 半导体/芯片"): st.session_state.active_sector = "半导体"
with col3:
    if st.button("🍷 消费/白酒"): st.session_state.active_sector = "白酒"
with col4:
    if st.button("🤖 人工智能"): st.session_state.active_sector = "人工智能"

st.info(f"正在扫描 **{st.session_state.active_sector}** 相关个股...")

# 2. 获取数据并筛选
all_data = get_realtime_market_data()

if not all_data.empty:
    # 定义行业关键词映射
    keywords = {
        "新能源": ["光伏", "锂", "能", "宁德", "隆基", "通威", "特变", "阳光"],
        "半导体": ["芯", "半导体", "微", "韦尔", "卓胜微", "北方华创", "紫光"],
        "白酒": ["酒", "茅台", "五粮液", "泸州", "汾酒"],
        "人工智能": ["智能", "AI", "科大", "三六零", "浪潮", "中科", "海康"]
    }
    
    filter_words = keywords.get(st.session_state.active_sector, [])
    
    # --- 修复点：更稳健的筛选逻辑 ---
    # 主要依靠名称 (Name) 进行模糊匹配
    name_mask = all_data['name'].str.contains('|'.join(filter_words), na=False)
    
    # 如果 industry 列有数据，也尝试匹配；否则只匹配名称
    if 'industry' in all_data.columns and not all_data['industry'].eq('').all():
        ind_mask = all_data['industry'].str.contains(st.session_state.active_sector, na=False)
        mask = name_mask | ind_mask
    else:
        mask = name_mask
        
    sector_df = all_data[mask].copy()
    
    if sector_df.empty:
        st.warning(f"未找到与 '{st.session_state.active_sector}' 相关的个股，请尝试其他板块。")
    else:
        # 二次筛选：剔除亏损 (PE>0) 和 暴跌股
        # 注意：先检查PE列是否存在且为数字
        if 'pe' in sector_df.columns:
            valid_stocks = sector_df[
                (sector_df['pe'] > 0) & 
                (sector_df['pe'] < 100) & 
                (sector_df['change_pct'] > -3)
            ].sort_values('change_pct', ascending=False).head(10)
        else:
            valid_stocks = sector_df.head(10) # 降级处理

        # 3. 展示结果
        st.header(f"2. {st.session_state.active_sector} 精选个股 ({len(valid_stocks)}只)")
        
        for index, row in valid_stocks.iterrows():
            with st.spinner(f"正在分析 {row['name']}..."):
                advice, reason, score, stop_loss, hist_df = analyze_stock(row['symbol'], row['name'], row['price'])
            
            color = "red" if "买" in advice or "持有" in advice else "green"
            
            with st.expander(f"{row['name']} ({row['symbol']}) | 现价: {row['price']} | 建议: {advice}"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"### 🤖 结论: :{color}[{advice}]")
                    st.write(f"**理由**: {reason}")
                    st.progress(min(score, 100))
                    st.write(f"止损参考: **{stop_loss:.2f}**")
                with c2:
                    if hist_df is not None and not hist_df.empty:
                        fig = go.Figure(data=[go.Candlestick(x=hist_df['date'],
                                        open=hist_df['open'], high=hist_df['high'],
                                        low=hist_df['low'], close=hist_df['close'])])
                        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['MA5'], line=dict(color='orange', width=1)))
                        fig.update_layout(height=250, margin=dict(t=0,b=0,l=0,r=0), xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("暂无K线数据")

else:
    st.error("数据接口连接超时，请稍后刷新重试。")

# 4. 手动查询
st.markdown("---")
st.header("3. 单股查询")
input_code = st.text_input("输入6位代码 (如 600519)", "")
if input_code and not all_data.empty:
    stock_info = all_data[all_data['symbol'] == input_code]
    if not stock_info.empty:
        row = stock_info.iloc[0]
        advice, reason, score, stop_loss, hist_df = analyze_stock(row['symbol'], row['name'], row['price'])
        st.info(f"结果：{row['name']} - {advice}")
    else:
        st.warning("未找到该代码，请检查输入。")
