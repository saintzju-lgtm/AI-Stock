import time # 记得在文件头部导入 time 库

@st.cache_data(ttl=600)
def get_realtime_market_data():
    """
    获取全市场实时行情 (增加重试机制，修复超时问题)
    """
    max_retries = 3  # 最大重试次数
    
    for attempt in range(max_retries):
        try:
            # 尝试获取数据
            df = ak.stock_zh_a_spot_em()
            
            # --- 数据清洗逻辑 (保持不变) ---
            rename_dict = {
                "代码": "symbol", "名称": "name", "最新价": "price", 
                "涨跌幅": "change_pct", "市盈率-动态": "pe", "市净率": "pb",
                "换手率": "turnover", "总市值": "market_cap", "所处行业": "industry",
                "量比": "volume_ratio"
            }
            df = df.rename(columns=rename_dict)
            
            # 补全缺失列
            str_cols = ['symbol', 'name', 'industry']
            for col in rename_dict.values():
                if col not in df.columns:
                    df[col] = "" if col in str_cols else 0
                
            # 数值转换
            numeric_cols = ['price', 'change_pct', 'pe', 'pb', 'turnover', 'market_cap', 'volume_ratio']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
            return df
            
        except Exception as e:
            # 如果不是最后一次尝试，则等待后重试
            if attempt < max_retries - 1:
                time.sleep(2) # 等待2秒
                continue
            else:
                # 彻底失败，打印错误信息
                st.error(f"行情接口连接超时 (已重试{max_retries}次): {e}")
                return pd.DataFrame()
