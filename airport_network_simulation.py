import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# -----------------------------------------------------------------------------
# 1. 参数配置
# -----------------------------------------------------------------------------
TOTAL_NODES = 50          # 最终网络的节点总数
INITIAL_NODES = 5         # 初始网络的节点数
NEW_NODE_EDGES = 2        # 新节点连接的边数 (m)
INITIAL_WEIGHT = 10       # 新边的初始权重
DELTA_W = 1               # 权重调整步长
MC_STEPS = 50             # 每个时间步的蒙特卡洛迭代次数
ANNEAL_START_RATIO = 0.8  # 后20%开启退火
INITIAL_TEMP = 100.0      # 初始温度 (由于能量值变大，温度可能需要调整)

# 地图与城市参数
MAP_WIDTH = 4000          # 地图宽度 (km)
MAP_HEIGHT = 4000         # 地图高度 (km)
MIN_M = 10                # 最小城市质量因子
MAX_M = 100               # 最大城市质量因子

# -----------------------------------------------------------------------------
# 2. 辅助函数：物理与能量计算
# -----------------------------------------------------------------------------

def get_distance(G, u, v):
    """
    计算两节点间的欧几里得距离。
    
    参数:
    - G: networkx.Graph, 包含节点属性的图对象
    - u, v: int, 节点ID
    
    返回:
    - float: 两节点间的直线距离 (单位: km)
    """
    pos_u = G.nodes[u]['pos']
    pos_v = G.nodes[v]['pos']
    return math.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)

def get_k_factor(dist):
    """
    根据距离计算分段距离修正系数 K(d_ij)。
    
    规则:
    1. 短距离 (<600km): 受高铁强烈替代，K=0.33
    2. 中距离 (600-2000km): 航空黄金运营区间，基准值 K=1.0
    3. 长距离 (>2000km): 运营成本高，K=0.4
    
    参数:
    - dist: float, 两节点间距离 (km)
    
    返回:
    - float: 距离修正系数 K
    """
    if dist < 600:
        return 0.33
    elif dist <= 2000:
        return 1.0
    else:
        return 0.4

def calculate_edge_energy(G, u, v, weight=None):
    """
    计算单条航线(边)的能量 H_ij，基于改进的引力模型。
    
    公式:
    H_ij = - (K * D * (Mi * Mj / d_ij)) * ln(1 + w_ij)
    其中:
    - Mi = sqrt(GDP_i * POP_i): 城市i的质量因子
    - K: 距离修正系数
    - D: 政策修正乘数 (默认1.0)
    - w_ij: 航线权重 (周航班量)
    
    参数:
    - G: networkx.Graph, 图对象
    - u, v: int, 节点ID
    - weight: float (可选), 指定计算时的权重。如果不传则读取图中现有权重。
             这在模拟“尝试修改权重但尚未应用”时非常有用。
    
    返回:
    - float: 该边的能量值 (通常为负值，绝对值越大表示连接越强/越稳定)
    """
    if weight is None:
        weight = G[u][v]['weight']
    
    # 如果权重为0，视为无有效连接，能量贡献为0
    # 数学上 log(1+0) = 0，符合物理意义
    if weight <= 0:
        return 0
        
    dist = get_distance(G, u, v)
    if dist == 0: dist = 1 # 避免除以0，虽不太可能
    
    # 计算城市质量因子 M = sqrt(GDP * POP)
    M_i = math.sqrt(G.nodes[u]['GDP'] * G.nodes[u]['POP'])
    M_j = math.sqrt(G.nodes[v]['GDP'] * G.nodes[v]['POP'])
    
    K = get_k_factor(dist)
    D = 1.0 # 政策修正乘数默认为1
    
    # 引力核心部分: (Mi * Mj) / d_ij * K
    gravity_core = K * D * (M_i * M_j / dist)
    
    # 结合航班量: 能量越低越好，故取负号
    # 航班量越大 -> log项越大 -> 能量越低(更负) -> 系统越趋于连接重要城市
    h_ij = -1 * gravity_core * math.log(1 + weight)
    
    return h_ij

def calculate_total_energy(G):
    """
    计算整个网络的总能量 (所有边能量之和)。
    用于监控系统整体演化趋势。
    
    参数:
    - G: networkx.Graph
    
    返回:
    - float: 网络总能量
    """
    total_energy = 0
    for u, v in G.edges():
        total_energy += calculate_edge_energy(G, u, v)
    return total_energy

def get_preferential_targets(G, m):
    """
    实现偏好连接 (Preferential Attachment) 机制。
    新节点连接到旧节点的概率与旧节点的度数成正比。
    P(i) = k_i / sum(k)
    
    参数:
    - G: networkx.Graph, 当前网络
    - m: int, 需要选择的目标节点数量
    
    返回:
    - list: 选中的目标节点ID列表
    """
    nodes = list(G.nodes())
    if not nodes: return []
    
    degrees = [G.degree(n) for n in nodes]
    total_degree = sum(degrees)
    
    if total_degree == 0:
        # 如果网络还没有边，则均匀随机选择
        probs = [1.0 / len(nodes)] * len(nodes)
    else:
        probs = [d / total_degree for d in degrees]
        
    # 无放回随机选择 m 个节点
    targets = np.random.choice(nodes, size=min(m, len(nodes)), replace=False, p=probs)
    return targets

def annealing_schedule(step, total_steps, initial_temp):
    """
    模拟退火温度调度函数。
    
    策略:
    - 前 80% 步数: 保持恒定高温 (探索阶段)
    - 后 20% 步数: 线性降温 (开发/锁定阶段)
    
    参数:
    - step: int, 当前迭代步数
    - total_steps: int, 总迭代步数
    - initial_temp: float, 初始温度
    
    返回:
    - float: 当前温度 T
    """
    anneal_start_step = int(total_steps * ANNEAL_START_RATIO)
    if step < anneal_start_step:
        return initial_temp
    
    # 线性降温计算
    progress = (step - anneal_start_step) / (total_steps - anneal_start_step)
    current_temp = initial_temp * (1 - progress)
    # 保证温度不为0，避免除零错误
    return max(current_temp, 1e-6)

def add_node_with_attributes(G, node_id):
    """
    向图中添加一个新节点，并随机初始化其属性。
    
    属性包括:
    - pos: (x, y) 地理坐标
    - GDP: 城市生产总值 (亿元)，服从帕累托分布
    - POP: 城市常住人口 (万人)，服从帕累托分布
    
    参数:
    - G: networkx.Graph
    - node_id: int, 新节点的ID
    """
    x = random.uniform(0, MAP_WIDTH)
    y = random.uniform(0, MAP_HEIGHT)
    
    # 模拟GDP (亿元) 和人口 (万人)
    # 使用帕累托分布模拟长尾效应 (少数大城市，多数小城市)
    # GDP范围: ~100 - 50000
    # 人口范围: ~50 - 2500
    
    gdp = (np.random.pareto(3.0) + 1) * 100
    gdp = min(gdp, 50000)
    
    pop = (np.random.pareto(3.0) + 1) * 50
    pop = min(pop, 2500)
    
    # -------------------------------------------------------------------------
    # 假设每个节点总权重 (Total Capacity) 与质量因子 M 成正比
    # M = sqrt(GDP * POP)
    # Total Weight = k * M (这里取 k=1.0，即总权重数值上等于 M)
    # -------------------------------------------------------------------------
    M = math.sqrt(gdp * pop)
    total_weight = int(M) # 取整作为总航班量配额
    
    G.add_node(node_id, pos=(x, y), GDP=gdp, POP=pop, M=M, TotalWeight=total_weight)

# -----------------------------------------------------------------------------
# 3. 主模型逻辑
# -----------------------------------------------------------------------------

def run_simulation():
    """
    运行航空网络演化模拟的主函数。
    
    模拟流程:
    1. 初始化: 生成初始种子网络 (全连接)
    2. 演化循环 (逐个加入新机场):
       - 偏好连接: 新机场按度偏好选择m个旧机场连接
       - 蒙特卡洛优化: 针对新机场的连接进行局部权重调整优化
         - 随机选择两条边进行权重交换 (+Δw / -Δw)
         - 计算能量差 ΔE
         - Metropolis准则决定是否接受新状态
         - 后期进行模拟退火降温
    3. 记录: 每个时间步记录系统总能量
    
    返回:
    - G: networkx.Graph, 演化完成后的最终网络
    - energy_history: list, 系统总能量随时间步的变化曲线
    """
    print(f"--- 开始模拟: 目标节点={TOTAL_NODES}, 能量函数=引力模型 ---")
    
    G = nx.Graph()
    
    # 1. 初始化种子网络
    # 创建初始的几个节点，并形成全连接图 (Clique)
    for i in range(INITIAL_NODES):
        add_node_with_attributes(G, i)
        
    # 构建全连接初始边
    for i in range(INITIAL_NODES):
        for j in range(i + 1, INITIAL_NODES):
            G.add_edge(i, j, weight=INITIAL_WEIGHT)
            
    energy_history = []
    current_total_energy = calculate_total_energy(G)
    energy_history.append(current_total_energy)
    
    # -------------------------------------------------------------------------
    # 2. 逐时间步加入新机场
    # -------------------------------------------------------------------------
    for t in range(INITIAL_NODES, TOTAL_NODES):
        new_node = t
        add_node_with_attributes(G, new_node)
        
        # 2.1 偏好连接生成初始边
        # -------------------------------------------------------------------------
        # 新节点的初始权重分配
        # 假设新节点可分配的总权重为 TotalWeight
        # 将其均匀分配给 NEW_NODE_EDGES 条初始边
        # -------------------------------------------------------------------------
        
        current_capacity = G.nodes[new_node]['TotalWeight']
        # 为了保证初始总权重完全被分配，进行简单的整数分配
        avg_weight = max(1, current_capacity // NEW_NODE_EDGES)
        
        targets = get_preferential_targets(G, NEW_NODE_EDGES)
        for target in targets:
            G.add_edge(new_node, target, weight=avg_weight)
            # 更新总能量 (增量更新: 只需加上新边的能量)
            current_total_energy += calculate_edge_energy(G, new_node, target)
            
    # -------------------------------------------------------------------------
    # 2.2 蒙特卡洛迭代优化
    # -------------------------------------------------------------------------
    # 权重交换规则：
    # 1. 选取一个公共机场节点（中心节点）
    #    - 高概率是新添加机场 (例如 80%)
    #    - 低概率是已有机场 (20%)，且概率与已有机场度数线性相关 (优先选枢纽)
    # 2. 从该节点的已有航线中选取 2 条航线（A-中心、B-中心）
    # 3. 执行权重调整：A增加Δω，B减少Δω，严格保证该中心节点的总权重恒定不变
    # -------------------------------------------------------------------------
    
    current_temp = INITIAL_TEMP
    
    for step in range(MC_STEPS):
        # --- 步骤1: 选取公共节点 (Hub Node) ---
        nodes = list(G.nodes())
        if len(nodes) < 3: continue # 节点太少无法形成有效结构
        
        # 80% 概率选新节点，20% 概率选旧节点
        is_select_new = (random.random() < 0.8)
        
        if is_select_new and (new_node in nodes):
            hub_node = new_node
        else:
            # 按度数概率选择已有节点 (包括新节点)
            degrees = [G.degree(n) for n in nodes]
            total_deg = sum(degrees)
            if total_deg == 0:
                probs = [1.0/len(nodes)] * len(nodes)
            else:
                probs = [d/total_deg for d in degrees]
            hub_node = np.random.choice(nodes, p=probs)
            
        # --- 步骤2: 选取两条航线 ---
        # 获取该节点的所有连边
        edges = list(G.edges(hub_node, data=True))
        if len(edges) < 2: continue # 边数少于2无法进行交换操作
            
        # 随机选两条不同的边进行权重博弈
        # (此处为纯随机，未来可扩展为与航线类型相关)
        idx1, idx2 = random.sample(range(len(edges)), 2)
        u1, v1, _ = edges[idx1]
        u2, v2, _ = edges[idx2]
        
        # 确定边的另一端节点 (邻居)
        n1 = u1 if v1 == hub_node else v1
        n2 = u2 if v2 == hub_node else v2
        
        # 随机选择增减方向 (一条边增加权重，另一条减少)
        # 这样 hub_node 的总权重 (Strength) 保持不变
        if random.random() < 0.5:
            plus_node, minus_node = n1, n2
        else:
            plus_node, minus_node = n2, n1
        
        w_plus_old = G[hub_node][plus_node]['weight']
        w_minus_old = G[hub_node][minus_node]['weight']
        
        # 约束检查: 权重不能减为0或负数
        if w_minus_old <= 0: continue 
        
        # --- 局部能量差计算 (增量法) ---
        # 只计算这两条边变化前后的能量差，无需全图重算
        E_edge1_old = calculate_edge_energy(G, hub_node, plus_node, w_plus_old)
        E_edge2_old = calculate_edge_energy(G, hub_node, minus_node, w_minus_old)
        
        w_plus_new = w_plus_old + DELTA_W
        w_minus_new = w_minus_old - DELTA_W
        
        E_edge1_new = calculate_edge_energy(G, hub_node, plus_node, w_plus_new)
        E_edge2_new = calculate_edge_energy(G, hub_node, minus_node, w_minus_new)
        
        delta_E = (E_edge1_new + E_edge2_new) - (E_edge1_old + E_edge2_old)
        
        # --- Metropolis 接受准则 ---
        accept = False
        if delta_E < 0:
            # 能量降低(系统更稳定)，无条件接受
            accept = True
        else:
            # 能量升高，按玻尔兹曼概率接受
            try:
                p = math.exp(-delta_E / current_temp)
                if random.random() < p:
                    accept = True
            except OverflowError:
                accept = False
        
        if accept:
            # 确认更新图中的权重
            G[hub_node][plus_node]['weight'] = w_plus_new
            G[hub_node][minus_node]['weight'] = w_minus_new
            # 更新总能量记录
            current_total_energy += delta_E
        
        # 降温 (模拟退火)
        current_temp = annealing_schedule(step, MC_STEPS, INITIAL_TEMP)
        
        energy_history.append(current_total_energy)
        if t % 5 == 0:
            print(f"进度: {t}/{TOTAL_NODES}, 总能量: {current_total_energy:.2f}")

    return G, energy_history

def plot_results(G, energy_history):
    """
    可视化模拟结果。
    
    绘制两张图:
    1. 总能量演化曲线: 展示系统是否趋于稳定
    2. 最终网络拓扑图: 展示节点空间分布、连边强度和节点重要性
    
    参数:
    - G: networkx.Graph, 最终网络
    - energy_history: list, 能量数据
    """
    plt.figure(figsize=(14, 6))
    
    # 图1: 能量演化
    plt.subplot(1, 2, 1)
    plt.plot(range(len(energy_history)), energy_history, 'g-')
    plt.title('Total System Energy Evolution')
    plt.xlabel('Time Step')
    plt.ylabel('Energy (Gravitational Potential)')
    plt.grid(True, alpha=0.3)
    
    # 图2: 最终网络拓扑 (使用地理坐标)
    plt.subplot(1, 2, 2)
    
    # 获取节点坐标
    pos = nx.get_node_attributes(G, 'pos')
    
    # 边粗细: 与权重(航班量)成正比
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [0.2 + (w / max_w) * 2.5 for w in weights]
    
    # 节点大小: 根据GDP和人口计算的质量因子
    node_sizes = []
    for n in G.nodes():
        m = math.sqrt(G.nodes[n]['GDP'] * G.nodes[n]['POP'])
        node_sizes.append(m * 0.5) # 调整缩放比例
    
    # 节点颜色: 根据度(连接数)区分
    degrees = [G.degree(n) for n in G.nodes()]
    
    nx.draw(G, pos, 
            node_size=node_sizes, 
            width=widths, 
            node_color=degrees,
            cmap=plt.cm.coolwarm,
            edge_color='gray', 
            alpha=0.6,
            with_labels=False)
            
    plt.title(f'Final Network (Geo-Spatial, N={TOTAL_NODES})')
    plt.axis('on') # 显示坐标轴
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    
    plt.tight_layout()
    plt.savefig('network_simulation_gravity.png', dpi=300)
    print("结果图已保存至: network_simulation_gravity.png")

if __name__ == "__main__":
    start_time = time.time()
    G_final, energy_hist = run_simulation()
    plot_results(G_final, energy_hist)
    print(f"运行时间: {time.time() - start_time:.2f} 秒")
