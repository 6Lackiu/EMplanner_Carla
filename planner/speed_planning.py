#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : speed_planning.py
import numpy as np
from planner import utils
"""
本文件解决的是运动规划下的速度规划问题，解决动态障碍物问题
在已知笛卡尔坐标系下的路径之后，进行速度规划的四个步骤：
1.直接以笛卡尔坐标的path为坐标轴，建立frenet坐标系
2.动态障碍物投影到frenet坐标系上，生成ST图
3.速度决策，动态规划
4.速度规划，二次规划
"""


def construct_ST_graph_and_planning(path_s: list, ego_pos: int, ego_speed: int, ego_acc: int,
                                    obs_pos: list, obs_speed: list,
                                    time_interval=8, time_slot=1):
    """
    我们每次规划的s的长度为100米，获取当前的速度，假设当前车辆通过这段距离是匀速运行的，所以可以预估时间区间T
    T = 100/cur_speed，市区的速度最高也就是50km/h, 13.888m/s, 因此如果道路通行不受限（无障碍物，无限速）保持最高速度匀速大概7.2s通过，
    我们取8s的时间间隔
    先假设车辆在往相同的方向行驶，只有超车和减速跟随两种决策，在实际中如其实只有往往是选择超车，因为如果前方车辆速度大于自车时，我们不认为前车是障碍物
    前车速度小于或者等于自车时，我们才会选择换道超车，先做出换道超车的效果

    实现步骤
    1.计算障碍物未来的每个时间所在的位置
    2.计算自车未来每个时间所在的位置
    3.找到这些未知的相交区域
    4.采样并且计算每个点的cost
    5.根据每个点的cost动态规划
    6.回溯找到最优的决策
    :param path_s: 路径规划所得到的轨迹s
    :param ego_pos: 自车的位置
    :param ego_speed: 自车的速度
    :param ego_acc: 自车的加速度
    :param obs_pos:  动态障碍物的当前位置
    :param obs_speed:  动态障碍物的当前速度
    :param time_interval:  S-T图的时间区间
    :param time_slot:  S-T图的时隙
    :return:  返回每个时间的期望位置
    """
    # 计算自车在8秒时间区间的预测位置,考虑规划起点位置
    ego_predict_pos_list = []
    for t in range(time_interval + 1):
        ego_predict_pos_list.append(ego_pos + t * ego_speed)

    # block 1: construct_ST_graph, the S_L graph contains two elements: grid and obs_predict_pos_list
    # 1.calculate the position of obstacle in the given time interval
    # 计算障碍物在8秒时间区间的预测位置，考虑初始起点位置
    obs_predict_pos_list = []  # 列表存储每个障碍物的预测位置
    # 列表的每个位置对应的是一个动态障碍物的预测列表[[障碍物1的预测],[障碍物2的预测]]
    for obs_n in range(len(obs_pos)):
        cur_obs = []
        for t in range(time_interval + 1):
            cur_obs.append(obs_pos[obs_n] + t * obs_speed)
        obs_predict_pos_list.append(cur_obs)

    # 2.non-uniform sampling for grid
    row = 100
    col = int(time_interval / time_slot)
    grid = np.zeros(shape=(row, col))
    for t_n in range(col):
        for s_n in range(1, row):  # Sparse sampling 稀疏采样
            if row <= 20:  # 20* 0.2 = 4
                grid[s_n][t_n] = grid[s_n][t_n] + 0.2
            elif row <= 40:  # 20 * 0.4 = 8
                grid[s_n][t_n] = grid[s_n][t_n] + 0.4
            elif row <= 60:  # 20 * 0.8 = 16
                grid[s_n][t_n] = grid[s_n][t_n] + 0.8
            elif row <= 80:  # 20 * 1.5 = 30
                grid[s_n][t_n] = grid[s_n][t_n] + 1.5
            else:  # 20 * 2.5 = 40
                grid[s_n][t_n] = grid[s_n][t_n] + 2.5

    # block 2: Planning, find an optimal path for ego-vehicle based on S-T graph
    # 声明一个二维数组记录每个采样点的cost,初始化为无穷大
    cost = np.ones(shape=(row, col)) * np.inf
    # 声明另一个二维数组，记录规划起点距离当前位置的最短路径的前一个位置
    pre_node_index = (row >> 1) ** np.ones(shape=(row, col), dtype="int32")

    # 计算起点到第一列的cost
    w_collision_cost = 1e10
    w_smooth_cost = [300, 2000, 10000]
    w_reference_cost = 20
    for i in range(row):
        cost[i][0] = cal_start_cost(obs_s_list, obs_t_list,
                                    begin_s=ego_pos, begin_t=0,
                                    begin_dt=ego_acc, end_s=grid[i][0], end_t=time_slot,
                                    w_cost_collision=w_collision_cost,
                                    w_cost_smooth=w_smooth_cost,
                                    w_cost_ref=w_reference_cost)


def cal_start_cost(obs_s_list, obs_t_list,
                   begin_s, begin_t, begin_dt,
                   end_s, end_t, sample_t,
                   w_cost_collision, w_cost_smooth, w_cost_ref
                   ):
    """  已验证
    计算当前位置的cost
    参数和DP_algorithm基本一致
    :param obs_s_list: 障碍物的s-t信息
    :param obs_t_list:
    :param begin_s: 规划起点的s-t信息
    :param begin_t:
    :param begin_dt:
    :param end_s:
    :param end_t:
    :param sample_t:
    :param w_cost_collision: 障碍物距离代价权重
    :param w_cost_smooth: 平滑代价权重，列表类型【w_dl_cost, w_ddl_cost, w_dddl_cost】考虑到s-l曲线的三阶导数
    :param w_cost_ref: 参考线代价权重
    :return:  规划起点到当前位置的代价cost
    """
    """下面是动态规划的一个示意图，b代表规划起点,中间是参考线，总共有采样五行（参考线上的采的点没有标出来）
    sample_s, sample_l分别代表沿着s方向的采样间隔和沿着l方向的采样间隔
    将采样的点存入矩阵，然后进行动态规划
        . . . . .
        . . . . .
        . . . . .
        . . . . .
    b  ---------reference line
    """
    # 五次多项式起点的边界条件
    start_t = begin_t
    start_dt = begin_dt
    start_ddt = 0
    start_s = begin_s
    # 终点的边界条件
    end_t = end_t
    end_dt = 0
    end_ddt = 0
    end_s = end_s

    # 计算五次多项式的系数
    coeffi = utils.cal_quintic_coefficient(start_t, start_dt, start_ddt,
                                           end_t, end_dt, end_ddt, start_s, end_s)
    # 在五次多项式构成的曲线上采样十个点计算cost
    s = np.zeros(shape=(10, 1))
    # l = np.zeros(shape=(10, 1))  # reserve memory space
    # dl = np.zeros(shape=(10, 1))
    # ddl = np.zeros(shape=(10, 1))
    # dddl = np.zeros(shape=(10, 1))
    # 计算s
    for i in range(10):
        s[i][0] = start_s + i * sample_t / 10  # 先从起点离散采样t, 然后通过五次多项式计算每一点的l, dt, ddt, dddt
    t = coeffi[0] + coeffi[1] * s + coeffi[2] * (s ** 2) + coeffi[3] * (s ** 3) + coeffi[4] * (s ** 4) + \
        coeffi[5] * (s ** 5)
    dt = coeffi[1] + 2 * coeffi[2] * s + 3 * coeffi[3] * (s ** 2) + 4 * coeffi[4] * (s ** 3) + 5 * coeffi[5] * (s ** 4)
    ddt = 2 * coeffi[2] + 6 * coeffi[3] * s + 12 * coeffi[4] * (s ** 2) + 20 * coeffi[5] * (s ** 3)
    dddt = 6 * coeffi[3] + 24 * coeffi[4] * s + 60 * coeffi[5] * (s * 2)
    cost_smooth = w_cost_smooth[0] * (dt.T @ dt) + w_cost_smooth[1] * (ddt.T @ ddt) + w_cost_smooth[2] * (
            dddt.T @ dddt)  # 平滑代价
    cost_ref = w_cost_ref * (t.T @ t)  # 参考线代价
    # 计算障碍物代价
    cost_collision = 0
    for i in range(len(obs_s_list)):  # 把每个障碍物都遍历一遍
        d_lon = obs_s_list[i] - s
        d_lat = obs_t_list[i] - t
        square_d = d_lon ** 2 + d_lat ** 2  # 这里直接在曲线上近似,实际上应该是计算两点之间的直线，在直角坐标系下进行（x1-x2)**2+(y1-y2)**2，
        # 但是考虑量采样点之间的五次多项式一般较平缓，我们就直接近似，简化计算
        # print(square_d)
        cost_collision += cal_obs_cost(w_cost_collision, square_d)
        if cost_collision > cost_collision:
            break

    return cost_smooth + cost_collision + cost_ref


def cal_obs_cost(w_cost_collision, square_d: np.ndarray, danger_dis=3, safe_dis=5):
    """  已验证
    计算障碍物的代价
    暂时设定为四米意外，不会碰撞
    四米到三米代价是1000/square_d
    三米以内w_cost_collision
    :param w_cost_collision: 障碍物碰撞的代价系数
    :param square_d: 障碍物与五次多项式上离散点的距离， np.array类型shape=(10,1)
    :param danger_dis: 障碍物与五次多项式上离散点的距离， np.array类型shape=(10,1)
    :param safe_dis: 障碍物与五次多项式上离散点的距离， np.array类型shape=(10,1)

    :return: 障碍物的代价
    """
    cost = 0
    for s_d in square_d.squeeze():
        if s_d <= danger_dis ** 2:
            # print("collision", "^^^^^^^^^^^^^^^^^^^^")
            cost += w_cost_collision
            break
        elif danger_dis ** 2 < s_d < safe_dis ** 2:
            # print("danger range", "^^^^^^^^^^^^^^^^^^^^")
            cost += 5000 / s_d
    return cost




