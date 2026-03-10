from typing import List, Tuple, Optional
from .utils import haversine_distance
from .airport import Airport

class Route:
    """有向航线边类"""
    def __init__(
        self,
        src: Airport,
        dst: Airport,
        distance: float,     # km
        M_product: float,    # M_src * M_dst
        weight: int = 1      # 航线条数（整数，可表示频次）
    ):
        self.src = src
        self.dst = dst
        self.distance = distance
        self.M_product = M_product
        self.weight = weight

    @property
    def flux(self) -> float:
        """
        引力模型预测流量：F ∝ (M1 * M2) / distance^β
        默认 β=2（经典引力模型），可扩展为可调参数
        返回无量纲相对通量，用于排序或阈值比较
        """
        beta = 2.0
        if self.distance == 0:
            return float('inf')
        return self.M_product / (self.distance ** beta)

    def __repr__(self):
        return f"Route({self.src.name}→{self.dst.name}, w={self.weight}, flux={self.flux:.2f})"

def build_routes(
    airports: List[Airport],
    flux_threshold: float = 10.0,
    max_routes_per_pair: int = 1,
    beta: float = 2.0
) -> List[Route]:
    """
    根据引力模型自动构建航线网络：
    
    参数：
        airports: 所有机场列表
        flux_threshold: 最小通量阈值（低于则不建边）
        max_routes_per_pair: 每对城市最多几条航线（模拟高频航线）
        beta: 引力衰减指数（默认2）

    返回：
        List[Route]：所有应建立的有向航线
    """
    routes = []
    n = len(airports)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a_i, a_j = airports[i], airports[j]
            dist = haversine_distance(a_i.lat, a_i.lon, a_j.lat, a_j.lon)
            M_prod = a_i.M * a_j.M

            # 计算通量（引力值）
            if dist == 0:
                flux = float('inf')
            else:
                flux = M_prod / (dist ** beta)

            # 判断是否建边
            if flux >= flux_threshold:
                # 权重可设为 floor(flux / threshold) 或按需分配（此处简化为 1~max）
                weight = min(max_routes_per_pair, int(flux // flux_threshold))
                if weight < 1:
                    weight = 1
                route = Route(a_i, a_j, dist, M_prod, weight)
                routes.append(route)

                # 同步更新机场的出/入度（注意：只在构建时一次性操作）
                a_i.out_degree += weight
                a_j.in_degree += weight
                a_i.outgoing_routes.append(route)
                a_j.incoming_routes.append(route)

    return routes

def blt_route(airports):
    routes = []
    for i in airports:
        for j in airports:
            if i == j:
                continue
            rout = Route(i, j, haversine_distance(i.lat, i.lon, j.lat, j.lon), i.M * j.M)
            routes.append(rout)
    return routes