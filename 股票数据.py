import akshare as ak
import pandas as pd

# 获取所有白酒股的股票代码
wine_stocks = []

try:
    # 获取概念板块成分股信息
    concept_list = ak.stock_board_concept_name_ths()
    # 查找白酒相关概念
    baijiu_concepts = concept_list[concept_list['name'].str.contains('白酒|酒业|酿酒')]

    print("找到的白酒相关概念板块：")
    print(baijiu_concepts[['name', 'code']])

    for _, concept in baijiu_concepts.iterrows():
        concept_code = concept['code']
        # 获取该概念板块的成分股
        concept_stocks = ak.stock_board_concept_cons_ths(symbol=concept_code)
        wine_stocks.extend(concept_stocks['code'].tolist())

    # 去重
    wine_stocks = list(set(wine_stocks))
    print(f"共找到{len(wine_stocks)}只白酒股")

except Exception as e:
    print(f"通过概念板块获取失败: {e}")
    # 如果无法获取概念板块，使用一些知名的白酒股代码作为备选
    wine_stocks = [
        'sh600519',  # 贵州茅台
        'sz000858',  # 五粮液
        'sz000568',  # 泸州老窖
        'sh600809',  # 山西汾酒
        'sz000860',  # 顺鑫农业
        'sh600702',  # 舍得酒业
        'sz000799',  # 酒鬼酒
        'sh603369',  # 今世缘
        'sh603589',  # 口子窖
        'sz002646',  # 青青稞酒
    ]
    print(f"使用备选的{len(wine_stocks)}只白酒股")

# 获取所有白酒股的历史5分钟线数据并合并
all_wine_data = pd.DataFrame()

# 获取股票名称映射
stock_info = ak.stock_info_a_code_name()

for stock_code in wine_stocks:
    try:
        # 获取单个股票的历史5分钟线数据
        print(f"正在获取{stock_code}的5分钟线数据...")

        # 获取5分钟线数据
        stock_5min_df = ak.stock_zh_a_minute(symbol=stock_code, period="5", adjust="")

        if stock_5min_df is not None and not stock_5min_df.empty:
            # 添加股票代码和股票名称列便于区分
            stock_name = stock_info[stock_info['code'] == stock_code.replace('sh', '').replace('sz', '')]['name'].values
            if len(stock_name) > 0:
                stock_name = stock_name[0]
            else:
                stock_name = "未知股票"

            stock_5min_df['stock_code'] = stock_code
            stock_5min_df['stock_name'] = stock_name

            # 合并到总数据框
            all_wine_data = pd.concat([all_wine_data, stock_5min_df], ignore_index=True)

            print(f"已获取{stock_code}({stock_name})的{len(stock_5min_df)}条5分钟线数据")
        else:
            print(f"{stock_code}没有5分钟线数据或数据为空")

    except Exception as e:
        print(f"获取{stock_code}的5分钟线数据时出现错误: {e}")

# 保存到指定路径
if not all_wine_data.empty:
    save_path = r"C:\Users\wangd\Desktop\wine_5min_data.csv"
    all_wine_data.to_csv(save_path, encoding="utf-8")

    print(f"所有白酒股5分钟线历史行情数据已保存到{save_path}文件中")
    print(f"共获取了{len(all_wine_data)}条数据记录")

    if not all_wine_data.empty:
        print(f"数据时间范围: {all_wine_data.index.min()} 到 {all_wine_data.index.max()}")

    print("数据预览：")
    print(all_wine_data.head(10))
    print("数据尾部预览：")
    print(all_wine_data.tail(10))
else:
    print("未能获取到任何5分钟线数据")
    # 创建空文件作为占位符
    empty_df = pd.DataFrame(
        columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'stock_code', 'stock_name'])
    empty_df.to_csv(r"C:\Users\wangd\Desktop\wine_5min_data.csv", encoding="utf-8")
    print(f"已创建空的CSV文件: C:\\Users\\wangd\\Desktop\\wine_5min_data.csv")



