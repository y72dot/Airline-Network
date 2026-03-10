from typing import Optional
from .utils import compute_M

class Airport:
    """机场节点类"""

    def __init__(
        self,
        name: str,
        city: str,
        lat: float,             # 经度
        lon: float,             # 纬度
        gdp: float,             # 亿元
        population: float,      # 万人
        policy: float = 1,      # 政策因素（0~1）
        w: int = 3,             # 航线数
        id_: Optional[int] = None
    ):
        self.name = name
        self.city = city
        self.lat = lat
        self.lon = lon
        self.gdp = gdp
        self.population = population
        self.M = compute_M(gdp, population)

        # 网络拓扑属性
        self.in_degree = 0      # 入度（被多少航线指向）
        self.out_degree = 0     # 出度（发出多少航线）
        self.id = id_           # 可选唯一标识符（用于后续映射）

        # 保存相连的航线（仅存储目标机场ID或对象引用，避免循环依赖）
        self.outgoing_routes = []   # List[Route] 或 List[tuple(id, weight)]
        self.incoming_routes = []

    def __repr__(self):
        return f"Airport('{self.name}', {self.city}, M={self.M:.2f})"

    def reset_degrees(self):
        """重置度数（用于多次构建网络前清理）"""
        self.in_degree = 0
        self.out_degree = 0
        self.outgoing_routes.clear()
        self.incoming_routes.clear()