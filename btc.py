import ccxt
import pandas as pd
import time

# ====================== 配置项（按需修改）======================
SYMBOL = 'BTC/USDT'  # 交易对
TIMEFRAME = '1m'  # 1分钟时间粒度
LIMIT_PER_REQUEST = 1500  # 单次请求最大条数（Binance 1m上限）
# 获取全量数据（从2017年BTC/USDT上线至今）；若只需最近数据，改since为'2024-01-01T00:00:00Z'
SINCE = '2017-08-17T00:00:00Z'

# ====================== 核心逻辑：获取数据并保存CSV ======================
# 初始化交易所（仅行情获取，无需API密钥）
exchange = ccxt.binance({
    'enableRateLimit': True,  # 开启限速，避免被交易所封禁
    'timeout': 30000
})

# 转换起始时间为毫秒级时间戳
start_timestamp = exchange.parse8601(SINCE)
all_ohlcv = []  # 存储所有K线数据

print("开始获取BTC/USDT 1分钟行情数据...")
while start_timestamp < exchange.milliseconds():
    try:
        # 分批获取K线数据
        ohlcv = exchange.fetch_ohlcv(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            since=start_timestamp,
            limit=LIMIT_PER_REQUEST
        )
        if not ohlcv:  # 无数据则停止
            break

        all_ohlcv.extend(ohlcv)
        # 更新下一批数据的起始时间（最后一条数据的时间 + 1分钟）
        start_timestamp = ohlcv[-1][0] + 60 * 1000
        print(f"已获取 {len(all_ohlcv)} 条数据...")
        time.sleep(0.1)  # 限速保护

    except Exception as e:
        print(f"临时报错：{e}，5秒后重试...")
        time.sleep(5)

# 转换为DataFrame并处理时间
df = pd.DataFrame(
    all_ohlcv,
    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
)
# 毫秒级时间戳转换为可读时间格式
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
# 去重（防止重复请求导致的数据重复）
df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

# 保存到a.csv
df.to_csv('a.csv', index=False, encoding='utf-8')
print(f"\n✅ 数据已全部保存到a.csv！")
print(f"📊 数据总量：{len(df)} 条1分钟K线")
print(f"🕒 时间范围：{df['datetime'].min()} 至 {df['datetime'].max()}")