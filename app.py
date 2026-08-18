import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timezone, timedelta
import threading
import requests
import pandas_datareader.data as web

# ==========================================
# 0. 页面全局配置与时区定义
# ==========================================
st.set_page_config(layout="wide", page_title="专业量化决策终端")

BEIJING_TZ = timezone(timedelta(hours=8))

# 全局最小请求间隔(秒):所有打向 Yahoo 的请求(不管前台点击还是后台线程)共享同一个节流器,
# 目的是"别去触发限流",而不是"触发了之后想办法绕过去"。
GLOBAL_RATE_INTERVAL = 1.5

# ==========================================
# 🧠 1. 全局解耦内存存储中心
# ==========================================
@st.cache_resource
def get_global_data_store():
    return {
        "stock_cache": {},
        "options_cache": {},     # key: f"{ticker}_{expiration}" -> option result tuple
        "fetch_timestamps": {},
        "active_queue": set(["BTDR", "AAPL", "TSLA", "NVDA"]),
        "lock": threading.Lock(),
        "last_yahoo_call_ts": 0.0,
    }

GLOBAL_STORE = get_global_data_store()


def _throttled_yahoo_call(func, max_retries=1):
    """
    所有对 Yahoo(不管是走 yfinance 库,还是直接 requests 打 Yahoo 的公开接口)的网络请求统一走这里:
      - 全局节流:整个应用共享一个最小请求间隔,避免并发/高频触发限流。
      - 遇到 429/Too Many Requests 就退避重试一次,超过重试次数就老实报错并提示稍后再试。
      - 不做代理轮换、身份伪装、伪造 User-Agent 池等任何"绕过限制"的操作。
    """
    last_err = None
    for attempt in range(max_retries + 1):
        with GLOBAL_STORE["lock"]:
            wait = GLOBAL_RATE_INTERVAL - (time.time() - GLOBAL_STORE["last_yahoo_call_ts"])
            if wait > 0:
                time.sleep(wait)
            GLOBAL_STORE["last_yahoo_call_ts"] = time.time()
        try:
            return func()
        except Exception as e:
            last_err = e
            msg = str(e)
            if ("429" in msg or "Too Many Requests" in msg) and attempt < max_retries:
                time.sleep(8 + attempt * 7)  # 老实退避,不做规避
                continue
            raise
    raise last_err


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
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=5)
        data = res.json()
        us_stocks = [f"{item['ticker']} - {item['title'].title()}" for item in data.values()]
        return custom_library + [s for s in us_stocks if s.split(" - ")[0] not in [c.split(" - ")[0] for c in custom_library]]
    except Exception:
        return custom_library + ["AMZN - Amazon", "META - Meta Platforms", "BABA - Alibaba", "TCEHY - Tencent", "BILI - Bilibili"]

STOCK_LIBRARY = load_us_stock_library()

# ==========================================
# 📐 3. 股本数据(换手率分母)
# ==========================================
@st.cache_data(ttl=86400)
def get_share_stats(ticker_symbol):
    """
    换手率分母修复:
      - 不再用硬编码字典(会随增发/回购/拆股过时)。
      - 优先用"流通股 floatShares",拿不到才退回"总股本 sharesOutstanding"。
      - 明确返回用的是哪种口径,UI上标注出来,避免两种不同含义的数字被当成同一件事看。
      - 每天刷新一次,不绑死在行情缓存的生命周期里。
      - 流通股在逻辑上不可能超过总股本;如果数据源给出了自相矛盾的数字,
        直接用总股本封顶,不用任何拍脑袋的百分比去"修正"(那种修正本身没有数据依据,
        对本来流通股就接近100%总股本的公司反而会引入新的错误)。
    """
    try:
        info = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).info)
    except Exception:
        info = {}

    float_shares = info.get('floatShares')
    shares_out = info.get('sharesOutstanding')

    float_shares = float(float_shares) if float_shares and float_shares > 0 else None
    shares_out = float(shares_out) if shares_out and shares_out > 0 else None

    if float_shares and shares_out and float_shares > shares_out:
        float_shares = shares_out  # 逻辑封顶,不做无依据的百分比修正

    if float_shares:
        return float_shares, "流通股(Float)"
    if shares_out:
        return shares_out, "总股本(Outstanding·近似)"
    return None, "无股本数据"


def get_effective_float(ticker_symbol):
    """支持侧边栏手动修正:如果你对某个标的的股本有更准的数据源,可以自己填,不用改代码。"""
    override_key = f"float_override_{ticker_symbol}"
    override_val = st.session_state.get(override_key)
    if override_val and override_val > 0:
        return float(override_val), "手动修正"
    return get_share_stats(ticker_symbol)


# ==========================================
# 🚀 4. 抗崩溃行情数据抓取引擎(个股 K 线 / 宏观指数)
# ==========================================
def fetch_macro_api(symbol, stooq_symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
        res = _throttled_yahoo_call(lambda: requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3))
        if res.status_code == 200:
            meta = res.json()['chart']['result'][0]['meta']
            price = meta.get('regularMarketPrice', 0.0)
            prev = meta.get('chartPreviousClose', meta.get('previousClose', price))
            if price > 0:
                return price, ((price / prev - 1) if prev else 0.0)
    except Exception:
        pass
    try:
        s_df = web.DataReader(stooq_symbol, 'stooq').head(2)
        if not s_df.empty and len(s_df) >= 2:
            p1, p2 = s_df['Close'].iloc[0], s_df['Close'].iloc[1]
            return p1, (p1 / p2 - 1)
    except Exception:
        pass
    return 0.0, 0.0


def get_live_quote(ticker_symbol):
    """实时(尽力而为)现价与成交量,来自 yfinance 的 fast_info,比日线K线更贴近当前时刻。"""
    try:
        tk = yf.Ticker(ticker_symbol)
        fi = _throttled_yahoo_call(lambda: tk.fast_info)
        price = fi.get('last_price') or fi.get('lastPrice')
        volume = fi.get('last_volume') or fi.get('lastVolume') or fi.get('regular_market_volume')
        return price, volume
    except Exception:
        return None, None


def build_scenario_model(hist_df):
    """
    组合优化版"场景回归预测"模型(替换原来"单变量OLS + 固定±6%"的做法):

    B. 分位数回归(Quantile Regression):直接对 High/Low 相对昨收的涨跌幅分别拟合
       90%/50%/10% 三条分位数线,作为"乐观/中性/悲观"三个场景,天然带概率含义,
       不再是"中性值 × 拍脑袋的固定百分比"。

    A. 波动率自适应:把"近10日实际波动率"也作为回归自变量之一(而不是只用开盘跳空幅度),
       这样区间会随这只股票当前的真实波动状态自动收窄/放宽,不再是所有股票、所有时期用同一套宽度。

    小样本兜底:数据不够(新股/次新股历史太短)或分位数回归本身失败时,自动退回原来的
    简单线性回归 + 固定百分比,保证页面不会因为模型失败而空白或崩溃。
    """
    fit_df = hist_df.dropna().copy()
    if len(fit_df) < 15:
        return {'mode': 'insufficient'}

    fit_df['gap'] = (fit_df['Open'] - fit_df['昨收']) / fit_df['昨收']
    ret = fit_df['Close'].pct_change()
    fit_df['recent_vol'] = ret.rolling(10).std()
    fit_df = fit_df.dropna(subset=['gap', 'recent_vol'])

    if len(fit_df) < 15:
        return {'mode': 'insufficient'}

    y_h = (fit_df['High'] / fit_df['昨收'] - 1).values
    y_l = (fit_df['Low'] / fit_df['昨收'] - 1).values

    if len(fit_df) >= 25:
        try:
            from sklearn.linear_model import QuantileRegressor
            X = fit_df[['gap', 'recent_vol']].values
            quantile_map = {'乐观': 0.9, '中性': 0.5, '悲观': 0.1}
            models = {'h': {}, 'l': {}}
            for scene, q in quantile_map.items():
                # alpha 是L1正则强度,样本量不大时加一点正则避免系数被少数点带偏
                models['h'][scene] = QuantileRegressor(quantile=q, alpha=0.3, solver='highs').fit(X, y_h)
                models['l'][scene] = QuantileRegressor(quantile=q, alpha=0.3, solver='highs').fit(X, y_l)
            return {'mode': 'quantile', 'models': models}
        except Exception:
            pass  # 分位数回归失败(比如scipy版本太旧不支持highs求解器),落到下面的兜底

    # 兜底方案:样本不足或分位数回归失败,退回原来的简单线性回归 + 固定百分比
    X_simple = fit_df[['gap']].values
    fallback = {}
    for tag, y in [('h', y_h), ('l', y_l)]:
        m = LinearRegression().fit(X_simple, y)
        fallback[f's_{tag}'], fallback[f'i_{tag}'] = m.coef_[0], m.intercept_
    return {'mode': 'linear_fallback', 'params': fallback}


def predict_scenarios(scenario_model, hist_df):
    """根据 build_scenario_model 产出的模型,结合最新一天数据,算出三档场景的压力/支撑参考价。"""
    if not scenario_model or scenario_model.get('mode') == 'insufficient':
        return None, 'insufficient'

    last = hist_df.iloc[-1]
    prev_close = last['昨收']
    if pd.isna(prev_close) or prev_close <= 0:
        return None, 'insufficient'

    gap = (last['Open'] - prev_close) / prev_close

    if scenario_model['mode'] == 'quantile':
        ret = hist_df['Close'].pct_change()
        recent_vol = ret.rolling(10).std().iloc[-1]
        if pd.isna(recent_vol):
            recent_vol = ret.std()
        X_pred = np.array([[gap, recent_vol]])
        models = scenario_model['models']

        h_vals = {scene: prev_close * (1 + models['h'][scene].predict(X_pred)[0]) for scene in ['乐观', '中性', '悲观']}
        l_vals = {scene: prev_close * (1 + models['l'][scene].predict(X_pred)[0]) for scene in ['乐观', '中性', '悲观']}

        # 分位数回归在小样本下偶尔会出现"分位数交叉"(比如10%分位算出来比50%分位还高),
        # 这里做一次排序兜底,保证"乐观>中性>悲观"这个顺序始终符合直觉。
        h_sorted = sorted(h_vals.values(), reverse=True)
        l_sorted = sorted(l_vals.values(), reverse=True)
        rows = [(scene, h_sorted[i], l_sorted[i]) for i, scene in enumerate(['乐观', '中性', '悲观'])]
        return rows, 'quantile'

    if scenario_model['mode'] == 'linear_fallback':
        p = scenario_model['params']
        p_h = prev_close * (1 + (p['i_h'] + p['s_h'] * gap))
        p_l = prev_close * (1 + (p['i_l'] + p['s_l'] * gap))
        rows = [
            ('乐观', p_h * 1.06, p_l * 1.06),
            ('中性', p_h, p_l),
            ('悲观', p_h * 0.94, p_l * 0.94),
        ]
        return rows, 'linear_fallback'

    return None, 'insufficient'


def estimate_iv_expected_move(ticker_symbol, current_price):
    """
    C. 期权隐含波动率(IV)交叉参考:
    找最近一个到期日里,离现价最近的平值(ATM)看涨/看跌期权,取其IV,
    按 sqrt(时间) 换算成"1个交易日等效"的预期波动幅度——这是期权市场对未来波动的定价,
    跟上面纯历史统计的分位数回归是两套独立信息源,放在一起可以互相印证,而不是只信一个模型。
    """
    try:
        exp_list = get_expiration_list(ticker_symbol)
        if not exp_list:
            return None
        nearest_exp = exp_list[0]

        opt_key = f"{ticker_symbol}_{nearest_exp}"
        opt_data = GLOBAL_STORE["options_cache"].get(opt_key)
        if not opt_data:
            opt_data = do_fetch_option_details(ticker_symbol, nearest_exp, current_price)

        calls_df, puts_df = opt_data[0], opt_data[1]
        atm_ivs = []
        for df in (calls_df, puts_df):
            if df is not None and not df.empty:
                idx = (df['strike'] - current_price).abs().idxmin()
                iv = df.loc[idx, 'impliedVolatility']
                if iv and iv > 0.01:
                    atm_ivs.append(float(iv))

        if not atm_ivs:
            return None

        atm_iv = float(np.mean(atm_ivs))
        days_to_exp = max((datetime.strptime(nearest_exp, '%Y-%m-%d') - datetime.now()).days, 1)
        move_1day = current_price * atm_iv * np.sqrt(1 / 365)
        return {'atm_iv': atm_iv, 'nearest_exp': nearest_exp, 'days_to_exp': days_to_exp, 'move_1day': move_1day}
    except Exception:
        return None


def calculate_turnover_metrics(volume, capital, timestamp_unix):
    """
    换手率的实时值 + 盘中外推全天预估值。
    修复点:原来的线性外推(全天量 = 已成交量 × 390/已交易分钟)假设"全天成交量匀速发生",
    但实际开盘前15-30分钟成交量通常远高于全天均值(典型的U型分布)——如果开盘5分钟就外推,
    会把"平时10%的量"直接放大成"全天78%"这种明显失真的数字。
    这里加一道保护:开盘不满30分钟时,只给实时值,不做全天外推,并明确告知原因。
    """
    if not capital or capital <= 0 or not volume or volume <= 0:
        return 0.0, 0.0, "数据不足"

    realtime_turnover = (volume / capital) * 100.0

    dt_utc = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc)
    dt_est = dt_utc.astimezone(timezone(timedelta(hours=-4)))  # 美东时间(不处理夏令时切换,仅作粗略盘中判断)

    weekday = dt_est.weekday()
    time_min = dt_est.hour * 60 + dt_est.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
    total_trade_min = 390
    min_reliable_elapsed = 30  # 开盘不满30分钟,线性外推不可靠

    if weekday < 5 and open_min <= time_min <= close_min:
        elapsed = time_min - open_min
        if elapsed < min_reliable_elapsed:
            return realtime_turnover, realtime_turnover, f"盘中(开盘{elapsed}分钟,量太少暂不外推全天)"
        projected_vol = volume * (total_trade_min / elapsed)
        projected_turnover = (projected_vol / capital) * 100.0
        return realtime_turnover, projected_turnover, f"盘中外推(已交易{elapsed}分钟)"
    elif weekday < 5 and time_min < open_min:
        return realtime_turnover, realtime_turnover, "未开盘(盘前)"
    else:
        return realtime_turnover, realtime_turnover, "已收盘/盘后"


def do_fetch_stock_data(ticker_symbol):
    try:
        now_ts = time.time()
        fetch_time_bj = datetime.fromtimestamp(now_ts, tz=timezone.utc).astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')

        hist = pd.DataFrame()
        source = "Yahoo Finance"

        try:
            hist = _throttled_yahoo_call(lambda: yf.Ticker(ticker_symbol).history(period="100d", interval="1d"))
        except Exception:
            hist = pd.DataFrame()

        if hist.empty:
            try:
                stooq_code = f"{ticker_symbol}.US" if "^" not in ticker_symbol and "." not in ticker_symbol else ticker_symbol
                hist = web.DataReader(stooq_code, 'stooq').head(100).sort_index()
                if not hist.empty:
                    source = "Stooq (备用源)"
            except Exception:
                pass

        if hist.empty:
            return None

        btc, _ = fetch_macro_api("BTC-USD", "BTCUSD")
        nasdaq, nasdaq_pct = fetch_macro_api("^IXIC", "^NDQ")
        vix, vix_pct = fetch_macro_api("^VIX", "^VIX")

        current_float, float_label = get_effective_float(ticker_symbol)

        hist.index = pd.to_datetime(hist.index).date
        hist['昨收'] = hist['Close'].shift(1)
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        std20 = hist['Close'].rolling(20).std()
        hist['Upper'], hist['Lower'] = hist['MA20'] + std20 * 2, hist['MA20'] - std20 * 2
        hist['换手率_raw'] = (hist['Volume'] / current_float) if (current_float and current_float > 0) else np.nan

        tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
        rmf = tp * hist['Volume']
        mfr = pd.Series(np.where(tp > tp.shift(1), rmf, 0)).rolling(14).sum() / pd.Series(np.where(tp < tp.shift(1), rmf, 0)).rolling(14).sum()
        hist['MFI'] = 100 - (100 / (1 + mfr.values))

        avg_vol = hist['Volume'].mean()
        dark = hist[hist['Volume'] > avg_vol * 1.2].tail(8).copy()
        dark['Signal'] = dark.apply(lambda x: "机构吸筹" if x['Close'] > x['Open'] else "大宗派发", axis=1)

        scenario_model = build_scenario_model(hist)

        live_price, live_volume = get_live_quote(ticker_symbol)
        volume_for_turnover = live_volume if (live_volume and live_volume > 0) else hist['Volume'].iloc[-1]
        rt_turnover, proj_turnover, trade_status = calculate_turnover_metrics(volume_for_turnover, current_float, now_ts)

        return hist, scenario_model, dark, {
            'btc': btc, 'nasdaq': nasdaq, 'nasdaq_pct': nasdaq_pct,
            'vix': vix, 'vix_pct': vix_pct,
            'float': current_float, 'float_label': float_label,
            'rt_turnover': rt_turnover, 'proj_turnover': proj_turnover, 'trade_status': trade_status,
            'volume': volume_for_turnover, 'source': source,
            'fetch_time': fetch_time_bj,
            'timestamp': now_ts
        }
    except Exception:
        return None


# ==========================================
# 🎯 5. 期权链(改用 yfinance 自带方法,不再手动拼 crumb/时间戳)
# ==========================================
def get_expiration_list(ticker_symbol):
    """
    到期日直接用 yfinance 的 tk.options,库内部已经处理好了和 Yahoo 的认证与日期映射,
    不需要我们自己算时间戳——这正是之前"选的到期日A、实际拿到B"这个bug的根源,现在直接消失。
    """
    tk = yf.Ticker(ticker_symbol)
    dates = _throttled_yahoo_call(lambda: list(tk.options))
    return dates


def do_fetch_option_details(ticker_symbol, expiration_date, fallback_current_price):
    """
    三级智能降级模型(保留原有设计):OI 优先 -> 成交量 -> 行权价分布权重。
    重构点:
      1. 期权原始数据来自 yfinance 的 tk.option_chain(date),不再手写反爬请求。
      2. 所有网络请求走全局节流器,失败(429)老实退避重试一次,不做任何绕过限制的操作。
      3. 现价优先用 fast_info 的实时价,而不是可能滞后的日线收盘价。
      4. IV 缺失/异常的行权价从 Gamma Flip 的数值积分里剔除,不再用固定值顶替(避免虚构数据)。
      5. 明确的异常状态返回,区分"接口失败/被限流"和"这个到期日确实没有报价"。
    """
    calls_df, puts_df = pd.DataFrame(), pd.DataFrame()
    call_wall, put_wall, gamma_flip, pcr_value = np.nan, np.nan, np.nan, np.nan
    atm_iv, move_1d, move_exp = np.nan, np.nan, np.nan
    calc_mode = "持仓量 (OI)"
    status_msg = None

    try:
        tk = yf.Ticker(ticker_symbol)
        opt = _throttled_yahoo_call(lambda: tk.option_chain(expiration_date), max_retries=1)
        calls, puts = opt.calls.copy(), opt.puts.copy()

        if calls.empty and puts.empty:
            status_msg = "该到期日 Yahoo 未返回任何期权报价(可能流动性太低)"

        live_price, _ = get_live_quote(ticker_symbol)
        current_price = live_price if (isinstance(live_price, (int, float)) and live_price and live_price > 0) else fallback_current_price
        if pd.isna(current_price) or current_price <= 0:
            current_price = 10.0

        for df in (calls, puts):
            if not df.empty:
                for col in ['strike', 'lastPrice', 'openInterest', 'volume', 'impliedVolatility']:
                    if col not in df.columns:
                        df[col] = 0.0
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                df['iv_valid'] = df['impliedVolatility'] > 0.01  # 无效IV直接从Gamma计算里剔除,不用固定值顶替
                df['density_weight'] = 1.0 / (1.0 + (df['strike'] - current_price).abs())

        # 期权隐含波动率(IV)Expected Move:只用"有效IV"的合约里离现价最近的那张(ATM),
        # 找不到有效IV就老实返回NaN,绝不拿固定值顶替——不然算出来的"预期波幅"是编的,不是市场真实定价。
        all_valid = pd.concat([
            calls[calls['iv_valid']] if not calls.empty else pd.DataFrame(),
            puts[puts['iv_valid']] if not puts.empty else pd.DataFrame(),
        ], ignore_index=True) if (not calls.empty or not puts.empty) else pd.DataFrame()

        if not all_valid.empty:
            atm_idx = (all_valid['strike'] - current_price).abs().idxmin()
            atm_iv = float(all_valid.loc[atm_idx, 'impliedVolatility'])
            days_to_exp = max((datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days, 1)
            move_1d = current_price * atm_iv * np.sqrt(1 / 365)
            move_exp = current_price * atm_iv * np.sqrt(days_to_exp / 365)

        if not calls.empty:
            calls = calls[(calls['strike'] >= current_price * 0.2) & (calls['strike'] <= current_price * 3.0)].copy()
        if not puts.empty:
            puts = puts[(puts['strike'] >= current_price * 0.2) & (puts['strike'] <= current_price * 3.0)].copy()

        tot_c_oi = calls['openInterest'].sum() if not calls.empty else 0
        tot_p_oi = puts['openInterest'].sum() if not puts.empty else 0
        tot_c_vol = calls['volume'].sum() if not calls.empty else 0
        tot_p_vol = puts['volume'].sum() if not puts.empty else 0

        if (tot_c_oi + tot_p_oi) > 0:
            w_col = 'openInterest'
            calc_mode = "持仓量 (OI)"
            pcr_value = tot_p_oi / tot_c_oi if tot_c_oi > 0 else 1.0
        elif (tot_c_vol + tot_p_vol) > 0:
            w_col = 'volume'
            calc_mode = "成交量 (Volume)"
            pcr_value = tot_p_vol / tot_c_vol if tot_c_vol > 0 else 1.0
        else:
            w_col = 'density_weight'
            calc_mode = "行权价分布"
            pcr_value = 1.0

        if not calls.empty:
            c_above = calls[calls['strike'] >= current_price]
            c_target = c_above if not c_above.empty else calls
            call_wall = float(c_target.loc[c_target[w_col].idxmax(), 'strike'])
        if not puts.empty:
            p_below = puts[puts['strike'] <= current_price]
            p_target = p_below if not p_below.empty else puts
            put_wall = float(p_target.loc[p_target[w_col].idxmax(), 'strike'])

        if pd.isna(call_wall) and not calls.empty:
            call_wall = float(calls['strike'].max())
        if pd.isna(put_wall) and not puts.empty:
            put_wall = float(puts['strike'].min())

        try:
            exp_date_dt = datetime.strptime(expiration_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            T = max((exp_date_dt - datetime.now(timezone.utc)).days, 1) / 365.0
            s_range = np.linspace(current_price * 0.7, current_price * 1.3, 80)
            net_gammas = []

            valid_calls = calls[calls['iv_valid']] if not calls.empty else calls
            valid_puts = puts[puts['iv_valid']] if not puts.empty else puts

            for s_test in s_range:
                tot_g = 0.0
                for df_leg, sign in ((valid_calls, 1.0), (valid_puts, -1.0)):
                    if df_leg.empty:
                        continue
                    for _, row in df_leg.iterrows():
                        k, w, iv = row['strike'], row[w_col], row['impliedVolatility']
                        w_val = max(float(w), 0.1)
                        if k > 0 and iv > 0:
                            d1 = (np.log(s_test / k) + 0.5 * (iv ** 2) * T) / (iv * np.sqrt(T))
                            tot_g += sign * w_val * (np.exp(-0.5 * d1 ** 2) / (s_test * iv * np.sqrt(2 * np.pi * T)))
                net_gammas.append(tot_g)

            net_gammas = np.array(net_gammas)
            zero_crossings = np.where(np.diff(np.sign(net_gammas)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                y1, y2 = net_gammas[idx], net_gammas[idx + 1]
                x1, x2 = s_range[idx], s_range[idx + 1]
                gamma_flip = float(x1 - y1 * (x2 - x1) / (y2 - y1)) if (y2 - y1) != 0 else float(x1)
            else:
                gamma_flip = float((call_wall + put_wall) / 2.0) if (pd.notnull(call_wall) and pd.notnull(put_wall)) else float(current_price)
        except Exception:
            gamma_flip = float(current_price)

        for df_type, target_df in [('calls', calls), ('puts', puts)]:
            if not target_df.empty:
                idx = (target_df['strike'] - current_price).abs().idxmin()
                slice_df = target_df.iloc[max(0, idx - 4): min(len(target_df), idx + 5)]
                if df_type == 'calls':
                    calls_df = slice_df
                else:
                    puts_df = slice_df

    except Exception as e:
        msg = str(e)
        if "429" in msg or "Too Many Requests" in msg:
            status_msg = "触发 Yahoo 限流(429),请等几分钟再试,或降低刷新频率(不建议强行重试)"
        else:
            status_msg = f"期权数据获取失败: {msg}"
        calc_mode = "接口异常"

    return calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_value, calc_mode, status_msg, atm_iv, move_1d, move_exp


# ==========================================
# 🔄 6. 永不崩溃的后台守护线程
# ==========================================
def background_updater_loop():
    while True:
        try:
            with GLOBAL_STORE["lock"]:
                tickers_to_process = list(GLOBAL_STORE["active_queue"])

            for ticker in tickers_to_process:
                try:
                    data = do_fetch_stock_data(ticker)
                    if data:
                        with GLOBAL_STORE["lock"]:
                            GLOBAL_STORE["stock_cache"][ticker] = data
                            GLOBAL_STORE["fetch_timestamps"][ticker] = data[3]['timestamp']

                        hist_df, _, _, _ = data
                        last_price = hist_df['Close'].iloc[-1]

                        try:
                            exp_list = get_expiration_list(ticker)
                        except Exception:
                            exp_list = []

                        for exp_date in exp_list[:2]:  # 只预热最近两个到期日,减少总请求量
                            try:
                                opt_key = f"{ticker}_{exp_date}"
                                opt_data = do_fetch_option_details(ticker, exp_date, last_price)
                                with GLOBAL_STORE["lock"]:
                                    GLOBAL_STORE["options_cache"][opt_key] = opt_data
                            except Exception:
                                pass

                except Exception:
                    pass
                time.sleep(2.0)

        except Exception:
            pass
        time.sleep(180)


@st.cache_resource
def start_background_engine():
    t = threading.Thread(target=background_updater_loop, daemon=True)
    t.start()
    return True

start_background_engine()

# ==========================================
# 🖥️ 7. 纯前端 UI 渲染层
# ==========================================
st.markdown("""<style> .main { background-color: #FFFFFF !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; } </style>""", unsafe_allow_html=True)

if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = "BTDR"

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
        st.session_state.current_ticker = new_tk
        with GLOBAL_STORE["lock"]:
            GLOBAL_STORE["active_queue"].add(new_tk)
        st.rerun()

    st.divider()
    st.caption("💡 换手率分母的手动修正(留空/填0则自动获取)")
    manual_float = st.number_input(
        f"{st.session_state.current_ticker} 流通股手动修正",
        min_value=0, value=0, step=1000000,
        help="如果你对这个标的的流通股数量有更准确的数据源,可以在这里手动填入,留 0 则走自动获取。"
    )
    override_key = f"float_override_{st.session_state.current_ticker}"
    st.session_state[override_key] = manual_float if manual_float > 0 else None

    st.caption(f"📡 期权数据源: Yahoo Finance(经 yfinance) | 全局节流间隔: {GLOBAL_RATE_INTERVAL}s")

ticker = st.session_state.current_ticker
st.title(f"🎯 {ticker} 专业量化决策终端")

stock_data = GLOBAL_STORE["stock_cache"].get(ticker)
last_ts = GLOBAL_STORE["fetch_timestamps"].get(ticker, 0)
now_ts = time.time()
is_expired = (now_ts - last_ts) > 180

if not stock_data or is_expired:
    with st.spinner(f"🔄 正在同步 **{ticker}** 最新市场数据..."):
        fresh_data = do_fetch_stock_data(ticker)
        if fresh_data:
            stock_data = fresh_data
            with GLOBAL_STORE["lock"]:
                GLOBAL_STORE["stock_cache"][ticker] = fresh_data
                GLOBAL_STORE["fetch_timestamps"][ticker] = fresh_data[3]['timestamp']
                GLOBAL_STORE["active_queue"].add(ticker)

if not stock_data:
    st.error("⚠️ 当前股票数据拉取失败,可能是股票代码不匹配、数据源暂时中断,或被限流(稍后再试)。")
else:
    hist_df, reg, dark_df, mkt = stock_data
    last = hist_df.iloc[-1]

    st.caption(f"🟢 数据解耦防护中 | **{ticker}** 数据抓取于 (北京时间): **{mkt['fetch_time']}** | 行情源: **{mkt['source']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bitcoin", f"${mkt['btc']:,.0f}" if mkt['btc'] > 0 else "N/A")
    m2.metric("Nasdaq", f"{mkt['nasdaq']:,.2f}" if mkt['nasdaq'] > 0 else "N/A", f"{mkt['nasdaq_pct']:.2%}" if mkt['nasdaq'] > 0 else "N/A")
    m3.metric("VIX 恐慌指数", f"{mkt['vix']:.2f}" if mkt['vix'] > 0 else "N/A", f"{mkt['vix_pct']:.2%}" if mkt['vix'] > 0 else "N/A", delta_color="inverse")
    m4.metric(f"{ticker} 现价", f"${last['Close']:.2f}", f"{(last['Close']/last['昨收']-1):.2%}" if pd.notnull(last['昨收']) else "N/A")

    st.divider()

    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时指标")
        if mkt['float'] and mkt['float'] > 0:
            st.write(f"实时已换手 ({mkt['float_label']}): **{mkt['rt_turnover']:.2f}%**")
            st.write(f"预估全天换手: **{mkt['proj_turnover']:.2f}%** (`{mkt['trade_status']}`)")
            st.caption(f"分母股本: {mkt['float']:,.0f} 股 | 分子成交量: {mkt['volume']:,.0f} 股")
        else:
            st.write("实时换手: **N/A (无股本数据)**")
        st.write(f"BOLL 高/低: **{last['Upper']:.2f} / {last['Lower']:.2f}**")
        st.write(f"资金 MFI: **{last['MFI']:.2f}**")
    with c2:
        st.subheader("📍 场景回归预测")
        rows, mode = predict_scenarios(reg, hist_df)

        if rows:
            mode_label = {
                'quantile': "分位数回归(10/50/90分位,已按近10日实际波动率自适应)",
                'linear_fallback': "样本不足或分位数回归失败,已退回简单线性回归+固定区间",
            }.get(mode, mode)
            st.caption(f"模型: {mode_label}")

            st.table(pd.DataFrame({
                "场景": [r[0] for r in rows],
                "压力参考": [r[1] for r in rows],
                "支撑参考": [r[2] for r in rows],
            }).style.format(precision=2))

            iv_ref = estimate_iv_expected_move(ticker, last['Close'])
            if iv_ref:
                st.caption(
                    f"🔮 期权隐含参考(交叉校验): 最近到期日 {iv_ref['nearest_exp']}"
                    f"(剩{iv_ref['days_to_exp']}天),ATM隐含波动率 {iv_ref['atm_iv']:.1%},"
                    f"换算1个交易日预期波动约 ±${iv_ref['move_1day']:.2f} "
                    f"(区间 ${last['Close']-iv_ref['move_1day']:.2f} ~ ${last['Close']+iv_ref['move_1day']:.2f})"
                )
            else:
                st.caption("🔮 期权隐含参考: 暂无法获取(可能被限流,或该标的期权流动性不足)")
        else:
            st.info("历史样本不足,暂无法生成场景预测(通常是新股/次新股历史K线太短)")

    st.divider()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(40).copy()
    p_df['label'] = pd.to_datetime(p_df.index).strftime('%m-%d')

    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Upper'], line=dict(color='rgba(0,102,204,0.3)'), name=f"High:{last['Upper']:.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['Lower'], line=dict(color='rgba(0,102,204,0.3)'), fill='tonexty', name=f"Low:{last['Lower']:.2f}"), row=1, col=1)
    fig.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], line=dict(color='#FF9800'), name=f"MA5:{last['MA5']:.2f}"), row=1, col=1)

    colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]

    if p_df['换手率_raw'].notnull().any():
        fig.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_raw'] * 100, marker_color=colors, name="换手%"), row=2, col=1)
    else:
        fig.add_trace(go.Bar(x=p_df['label'], y=p_df['Volume'] / 10000, marker_color=colors, name="成交量(万股)"), row=2, col=1)

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type='category', tickmode='linear', dtick=1, tickangle=-90)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    o_col, d_col = st.columns([1.6, 1])

    with o_col:
        st.subheader("🕯️ 全景期权决策分析")

        try:
            exp_list = get_expiration_list(ticker)
        except Exception as e:
            exp_list = []
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                st.warning("⚠️ 到期日列表获取被限流(429),请等几分钟再试。")
            else:
                st.warning(f"⚠️ 到期日列表获取异常: {e}")

        if exp_list:
            today_str = datetime.now().strftime('%Y-%m-%d')
            default_exp_idx = 0
            for idx, d in enumerate(exp_list):
                if d > today_str:
                    default_exp_idx = idx
                    break

            selected_exp = st.selectbox("📅 选择期权到期日", options=exp_list, index=default_exp_idx)

            opt_key = f"{ticker}_{selected_exp}"
            opt_data = GLOBAL_STORE["options_cache"].get(opt_key)
            if not opt_data or is_expired:
                opt_data = do_fetch_option_details(ticker, selected_exp, last['Close'])
                if opt_data:
                    with GLOBAL_STORE["lock"]:
                        GLOBAL_STORE["options_cache"][opt_key] = opt_data

            if opt_data:
                calls_df, puts_df, call_wall, put_wall, gamma_flip, pcr_val, calc_mode, status_msg, atm_iv, move_1d, move_exp = opt_data

                if status_msg:
                    st.info(f"ℹ️ {status_msg}")

                iv_caption = f" | ATM隐含波动率: **{atm_iv:.2%}**" if pd.notnull(atm_iv) else " | ATM隐含波动率: 暂无有效IV"
                st.caption(f"ℹ️ 权重维度: **{calc_mode}** | 到期日: **{selected_exp}**{iv_caption}")

                q1, q2, q3, q4 = st.columns(4)
                q1.metric("🧱 看涨墙 (Call Wall)", f"${call_wall:.2f}" if pd.notnull(call_wall) else "N/A")
                q2.metric("🧱 看跌墙 (Put Wall)", f"${put_wall:.2f}" if pd.notnull(put_wall) else "N/A")
                q3.metric("🌀 伽马翻转点", f"${gamma_flip:.2f}" if pd.notnull(gamma_flip) else "N/A")
                q4.metric("📊 Put/Call Ratio", f"{pcr_val:.2f}" if pd.notnull(pcr_val) else "N/A")

                if pd.notnull(move_1d) and pd.notnull(move_exp):
                    em1, em2 = st.columns(2)
                    em1.info(f"🎯 期权隐含单日预期波幅: ±${move_1d:.2f} (区间 ${last['Close']-move_1d:.2f} ~ ${last['Close']+move_1d:.2f})")
                    em2.success(f"📅 至到期日({selected_exp})预期波幅: ±${move_exp:.2f} (区间 ${last['Close']-move_exp:.2f} ~ ${last['Close']+move_exp:.2f})")
                else:
                    st.caption("🎯 期权隐含预期波幅: 该到期日暂无有效IV数据,不做估算(不会拿假数字凑数)")

                t1, t2 = st.tabs(["📈 看涨 (Calls)", "📉 看跌 (Puts)"])
                with t1:
                    if not calls_df.empty:
                        st.dataframe(calls_df[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']].style.format(
                            {'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}
                        ), use_container_width=True)
                    else:
                        st.info("该到期日暂无看涨期权数据")
                with t2:
                    if not puts_df.empty:
                        st.dataframe(puts_df[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']].style.format(
                            {'impliedVolatility': '{:.2%}', 'lastPrice': '{:.2f}', 'strike': '{:.2f}', 'openInterest': '{:,.0f}'}
                        ), use_container_width=True)
                    else:
                        st.info("该到期日暂无看跌期权数据")
        else:
            st.info("💡 当前标的暂无期权链交易,或接口暂时不可用/被限流。")

    with d_col:
        st.subheader("🌑 大宗异动打印 (Dark Pool)")
        if not dark_df.empty:
            st.table(dark_df[['Volume', 'Signal']])
        else:
            st.info("近期无显著异动")

    st.subheader("📋 历史明细")
    hist_show = hist_df.tail(15).copy()
    if hist_show['换手率_raw'].notnull().any():
        hist_show['换手'] = (hist_show['换手率_raw'] * 100).map('{:.2f}%'.format)
    else:
        hist_show['换手'] = "N/A"
    st.dataframe(hist_show[['Open', 'High', 'Low', 'Close', '换手', 'MFI', 'MA20', 'MA5']].style.format(precision=2), use_container_width=True)
