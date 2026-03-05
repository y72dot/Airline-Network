import math
from typing import Tuple

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算地球表面两点间的大圆距离（单位：公里）
    使用 Haversine 公式
    """
    R = 6371.0  # 地球平均半径，单位 km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def compute_M(gdp: float, population: float) -> float:
    """
    引力模型中的质量参数 M = sqrt(GDP * population)
    GDP 单位：亿元人民币；population 单位：万人（保持量纲一致即可，可归一化）
    """
    if gdp <= 0 or population <= 0:
        raise ValueError("GDP and population must be positive.")
    return math.sqrt(gdp * population)