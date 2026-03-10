from typing import List
from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts.globals import ChartType
from .airport import Airport
from .route import Route

def visualize_airline_network(
    airports: List[Airport],
    routes: List[Route],
    title: str = "中国民航航线网络",
    show_labels: bool = True,
    line_width: int = 2,
    line_opacity: float = 0.7
) -> Geo:
    """
    使用 pyecharts 绘制中国地图 + 机场节点 + 航线边
    
    注意：
        - 需提前安装：pip install pyecharts
        - pyecharts 内置中国地图数据（'china'）
    """
    geo = Geo()
    geo.add_schema(maptype="china", itemstyle_opts=opts.ItemStyleOpts(color="#f0f0f0"))

    # 逐个注册机场坐标（pyecharts 要求先注册坐标点）
    for a in airports:
        geo.add_coordinate(a.name, a.lon, a.lat)

    # 计算节点大小（根据 M 值归一化）
    max_M = max([a.M for a in airports]) if airports else 1
    sizes = [max(8, 5 + a.M / max_M * 15) for a in airports]

    # 添加机场节点（散点图）
    geo.add(
        "",
        [[a.name, size] for a, size in zip(airports, sizes)],
        type_=ChartType.SCATTER,
        symbol_size=lambda x: x[1],  # 使用传入的 size 值
        color="red",
        label_opts=opts.LabelOpts(is_show=show_labels, color="black", font_size=10)
    )

    # 添加航线（有向边）
    # 注意：pyecharts 的 Lines 需要 [起点名称, 终点名称] 格式
    line_data = []
    for r in routes:
        # 每条航线的权重可以通过重复添加或调整线宽来体现
        for _ in range(r.weight):
            line_data.append([r.src.name, r.dst.name])

    if line_data:  # 只有存在航线时才添加
        geo.add(
            "",
            line_data,
            type_=ChartType.LINES,
            effect_opts=opts.EffectOpts(
                symbol="arrow", 
                symbol_size=6, 
                color="blue", 
                trail_length=0.2
            ),
            linestyle_opts=opts.LineStyleOpts(
                width=line_width,
                opacity=line_opacity,
                curve=0,
                color="rgba(0, 100, 200, 0.8)"
            ),
            label_opts=opts.LabelOpts(is_show=False)
        )

    # 配置标题与工具提示
    geo.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        legend_opts=opts.LegendOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(
            formatter="{b}"
        )
    )

    return geo