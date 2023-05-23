#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : speed_planning_test.py


"""
本文件解决的是运动规划下的速度规划问题，解决动态障碍物问题
在已知笛卡尔坐标系下的路径之后，进行速度规划的四个步骤：
1.直接以笛卡尔坐标的path为坐标轴，建立frenet坐标系
2.动态障碍物投影到frenet坐标系上，生成ST图
3.速度决策，动态规划
4.速度规划，二次规划
"""


import numpy as np
from scipy.interpolate import interp1d
# from planner import planning_utils
import cvxopt
import cvxopt.solvers


def calc_speed_planning_start_condition(plan_start_vx, plan_start_vy, plan_start_ax, plan_start_ay, plan_start_heading):
    """
    该函数计算速度规划的初始条件
    params: 都为cartesian坐标系下的信息
    """
    tor = np.array([np.cos(plan_start_heading), np.sin(plan_start_heading)])
    # 计算向量 v 在切向的投影
    v_t = np.dot(tor.T, np.array([plan_start_vx, plan_start_vy]))
    a_t = np.dot(tor.T, np.array([plan_start_ax, plan_start_ay]))
    plan_start_s_dot = v_t
    plan_start_s_dot2 = a_t

    return plan_start_s_dot, plan_start_s_dot2


def generate_st_graph(dynamic_obs_s_set, dynamic_obs_l_set, dynamic_obs_s_dot_set, dynamic_obs_l_dot_set):
    """
        param:  动态障碍物的s l 和 sl方向上的速度
    """
    n = len(dynamic_obs_s_set)
    obs_st_s_in_set = np.ones(n) * np.nan
    obs_st_s_out_set = np.ones(n) * np.nan
    obs_st_t_in_set = np.ones(n) * np.nan
    obs_st_t_out_set = np.ones(n) * np.nan
    # obs_st_l_in_set = np.ones(n) * np.nan
    # obs_st_l_out_set = np.ones(n) * np.nan

    for i in range(len(dynamic_obs_s_set)):
        if np.isnan(dynamic_obs_s_set[i]):
            break
        if abs(dynamic_obs_l_dot_set[i]) < 0.3:  # 侧向缓慢移动的障碍物
            if abs(dynamic_obs_l_set[i]) > 2:  # 距离横向太远，速度规划直接忽略
                continue
            else:
                # TODO 需要做虚拟障碍物
                # 这里由于算力原因没做逻辑
                # 如何做：感知模块加逻辑，给出障碍物跟踪，判断两帧之间的感知所看到的障碍物是否为同一个
                #         速度规划模块在一开始先给出虚拟障碍物决策，不做处理
                #         下一帧，路径规划拿到虚拟障碍物标记，规划出绕过去的路径/跟车路径
                #                 速度规划计算出速度，绕过去/跟车
                # 本算法欠缺的：障碍物结构体 结构体不仅要包含坐标 速度 还要包含 决策标记(是否为虚拟障碍物，左绕还是右绕，避让还是超车)
                #              感知模块，判断两帧之间的障碍物是否为同一个
                #              算力
                continue

        # t_zero 为动态障碍物的l到0，所需要的时间
        t_zero = - dynamic_obs_l_set[i] / dynamic_obs_l_dot_set[i]  # 时间等于路程除以速度
        # 计算+-2缓冲时间
        t_boundary1 = 2 / dynamic_obs_l_dot_set[i] + t_zero
        t_boundary2 = -2 / dynamic_obs_l_dot_set[i] + t_zero
        if t_boundary1 > t_boundary2:
            t_max = t_boundary1
            t_min = t_boundary2
        else:
            t_max = t_boundary2
            t_min = t_boundary1
        if t_max < 1 or t_min > 8:
            # 对于切入切出太远的，或者碰瓷的，忽略
            # 车辆运动是要受到车辆动力学制约的，如果有碰瓷的，即使规划出了很大的加速度，车辆也执行不了
            # 碰瓷障碍物也需要做虚拟障碍物和路径规划一起解决
            continue
        if t_min < 0 and t_max > 0:
            # 在感知看到的时候，障碍物已经在+-2的内部了
            obs_st_s_in_set[i] = dynamic_obs_s_set[i]
            # 匀速运动
            obs_st_s_out_set[i] = dynamic_obs_s_set[i] + dynamic_obs_s_dot_set[i] * t_max
            obs_st_t_in_set[i] = 0
            obs_st_t_out_set[i] = t_max
        else:
            # 正常障碍物
            obs_st_s_in_set[i] = dynamic_obs_s_set[i] + dynamic_obs_s_dot_set[i] * t_min
            obs_st_s_out_set[i] = dynamic_obs_s_set[i] + dynamic_obs_s_dot_set[i] * t_max
            obs_st_t_in_set[i] = t_min
            obs_st_t_out_set[i] = t_max

    return obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set


def speed_DP(obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set, plan_start_s_dot,
             reference_speed=50, w_cost_ref_speed=4000, w_cost_accel=100, w_cost_obs=10000000):
    """
        速度动态规划部分
        params: dynamic_obs_s_set, dynamic_obs_l_set: 动态障碍物sl信息
                dynamic_obs_s_dot_set, dynamic_obs_l_dot_set: 动态障碍物sl速度信息
                plan_start_vx, vy, ax, ay, heading: 速度规划起点信息（均为直角坐标系下）
                reference_speed_unlimit 推荐速度
                推荐速度代价权重w_cost_ref_speed, 加速度代价权重w_cost_accel, 障碍物代价权重w_cost_obs
    """
    # 时间从0到8开始规划，最多8秒
    # s的范围从0开始到路径规划的path的总长度为止
    # 为了减少算力，采用非均匀采样，s越小的越密，越大的越稀疏
    s_list = np.concatenate((np.arange(0, 5, 0.5), np.arange(5.5, 15, 1), np.arange(16, 30, 1.5), np.arange(32, 55, 2.5)))
    # t采用均匀采样每0.5s采样一个点
    t_list = np.arange(0.5, 8.5, 0.5)
    # 声明st代价矩阵，表示从起点开始到(i,j)点的最小代价为dp_st_cost(i,j)
    # s采样40个 t采样16个 一共是40*16
    dp_st_cost = np.ones((40, 16)) * np.inf
    dp_st_s_dot = np.zeros((40, 16))    # 表示从起点开始到(i,j)点的最优路径的末速度
    # 需要一个矩阵保持最优路径的前一个节点方便回溯
    dp_st_node = np.zeros((40, 16))     # 表示位置为(i,j)的节点中，最优的上一层节点的行号为dp_st_node(i,j)

    # 计算从dp起点到第一列的cost
    for i in range(len(s_list)):
        # 第一列的前一个节点只有起点，起点的s t都是0
        dp_st_cost[i, 0] = CalcDpCost(0, 0, i, 0, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
                                      obs_st_t_out_set, w_cost_ref_speed, reference_speed,
                                      w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, dp_st_s_dot)
        s_end, t_end = CalcSTCoordinate(i, 0, s_list, t_list)
        dp_st_s_dot[i, 0] = s_end / t_end

    # 动态规划主程序
    for i in range(1, len(t_list)):     # i 为列循环
        for j in range(len(s_list)):    # j 为行循环
            cur_row = j
            cur_col = i
            for k in range(len(s_list)):    # 遍历前一列
                pre_row = k
                pre_col = i - 1
                # 计算边的代价 其中起点为pre_row,pre_col 终点为cur_row cur_col
                cost_temp = CalcDpCost(pre_row, pre_col, cur_row, cur_col, obs_st_s_in_set, obs_st_s_out_set,
                                       obs_st_t_in_set, obs_st_t_out_set, w_cost_ref_speed, reference_speed,
                                       w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, dp_st_s_dot)
                if cost_temp + dp_st_cost[pre_row, pre_col] < dp_st_cost[cur_row, cur_col]:
                    dp_st_cost[cur_row, cur_col] = cost_temp + dp_st_cost[pre_row, pre_col]
                    # 计算最优的s_dot
                    s_start, t_start = CalcSTCoordinate(pre_row, pre_col, s_list, t_list)
                    s_end, t_end = CalcSTCoordinate(cur_row, cur_col, s_list, t_list)
                    dp_st_s_dot[cur_row, cur_col] = (s_end - s_start) / (t_end - t_start)
                    # 将最短路径的前一个节点的行号记录下来
                    dp_st_node[cur_row, cur_col] = pre_row

    # 输出初始化
    dp_speed_s = np.ones(len(t_list)) * np.nan
    dp_speed_t = dp_speed_s
    # 找到dp_node_cost 上边界和右边界代价最小的节点
    min_cost = np.inf
    min_row = np.inf
    min_col = np.inf

    for i in range(len(s_list)):    # 遍历右边界
        if dp_st_cost[i, len(t_list) - 1] <= min_cost:
            min_cost = dp_st_cost[i, len(t_list) - 1]
            min_row = i
            min_col = len(t_list) - 1

    for j in range(len(t_list)):    # 遍历上边界
        if dp_st_cost[0, j] <= min_cost:
            min_cost = dp_st_cost[0, j]
            min_row = 0
            min_col = j

    # 这里要注意，虽然我们在t上每0.5s采样一个时间，采了16个点，但是min_col未必等于16
    # 也就是说动态规划的最优解的时间未必是8秒
    # 所以我们动态规划最后的输出s t不一定写满
    # 先把终点的ST输出出来
    dp_speed_s[min_col], dp_speed_t[min_col] = CalcSTCoordinate(min_row,min_col,s_list,t_list)

    # 反向回溯
    while min_col != 0:
        pre_row = dp_st_node[min_row, min_col]
        pre_col = min_col - 1
        dp_speed_s[pre_col], dp_speed_t[pre_col] = CalcSTCoordinate(pre_row,pre_col,s_list,t_list)
        min_row = pre_row
        min_col = pre_col

    return dp_speed_s, dp_speed_t


def CalcDpCost(row_start, col_start, row_end, col_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
               obs_st_t_out_set, w_cost_ref_speed, reference_speed, w_cost_accel, w_cost_obs, plan_start_s_dot,
               s_list, t_list, dp_st_s_dot):
    """
        函数将计算链接两个节点之间边的代价
        params: 边的起点的行列号row_start,col_start 边的终点行列号row_end,col_end
                障碍物st信息obs_st_s_in_set,obs_st_s_out_set,obs_st_t_in_set,obs_st_t_out_set
                推荐速度代价权重w_cost_ref_speed,加速度代价权重w_cost_accel,障碍物代价权重w_cost_obs
                推荐速度reference_speed
                拼接起点的速度plan_start_s_dot
                s_list,t_list 采样距离
                dp_st_s_dot 用于计算加速度
    """

    # 首先计算终点的st坐标
    s_end, t_end = CalcSTCoordinate(row_end, col_end, s_list, t_list)
    # 规定起点的行列号为0
    if row_start == 0:
        # 边的起点为dp的起点
        s_start = 0
        t_start = 0
        s_dot_start = plan_start_s_dot
    else:
        # 边的起点不是dp的起点
        s_start, t_start = CalcSTCoordinate(row_start, col_start, s_list, t_list)
        s_dot_start = dp_st_s_dot[row_start][col_start]
    cur_s_dot = (s_end - s_start) / (t_end - t_start)
    cur_s_dot2 = (cur_s_dot - s_dot_start) / (t_end - t_start)
    # 计算推荐速度代价
    cost_ref_speed = w_cost_ref_speed * (cur_s_dot - reference_speed) ** 2
    # 计算加速度代价，这里注意，加速度不能超过车辆动力学上下限
    if 4 > cur_s_dot2 > -6:
        cost_accel = w_cost_accel * cur_s_dot2 ** 2
    else:
        # 超过车辆动力学限制，代价会增大很多倍
        cost_accel = 100000 * w_cost_accel * cur_s_dot2 ** 2
    cost_obs = CalcObsCost(s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set,
                           obs_st_t_in_set, obs_st_t_out_set, w_cost_obs)
    cost = cost_obs + cost_accel + cost_ref_speed

    return cost


def CalcObsCost(s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set, w_cost_obs):
    """
    该函数将计算边的障碍物代价
        params: 边的起点终点s_start,t_start,s_end,t_end
                障碍物信息 obs_st_s_in_set,obs_st_s_out_set,obs_st_t_in_set,obs_st_t_out_set
                障碍物代价权重w_cost_obs
        return: 边的障碍物代价obs_cost
    """
    obs_cost = 0
    # 采样点的个数
    n = 5
    # 采样时间间隔
    dt = (t_end - t_start)/(n - 1)
    # 边的斜率
    k = (s_end - s_start)/(t_end - t_start)
    for i in range(n):
        # 计算采样点的坐标
        t = t_start + (i - 1) * dt
        s = s_start + k * (i - 1) * dt
        # 遍历所有障碍物
        for j in range(len(obs_st_s_in_set)):
            if np.isnan(obs_st_s_in_set[j]):
                continue
            # 计算点到st线段的最短距离
            vector1 = np.array([obs_st_s_in_set[j], obs_st_t_in_set[j]]) - np.array([s, t])
            vector2 = np.array([obs_st_s_out_set[j], obs_st_t_out_set[j]]) - np.array([s, t])
            vector3 = vector2 - vector1
            min_dis = 0
            dis1 = np.sqrt(vector1.dot(vector1))
            dis2 = np.sqrt(vector2.dot(vector2))
            dis3 = abs(vector1[0]*vector3[1] - vector1[1]*vector3[0])/np.sqrt(vector3.dot(vector3))
            if (vector1.dot(vector3) > 0 and vector2.dot(vector3) > 0) or (vector1.dot(vector3) < 0 and vector2.dot(vector3) < 0):
                min_dis = min(dis1, dis2)
            else:
                min_dis = dis3
            obs_cost = obs_cost + CalcCollisionCost(w_cost_obs, min_dis)

    return obs_cost


def CalcCollisionCost(w_cost_obs, min_dis):
    if abs(min_dis) < 0.5:
        collision_cost = w_cost_obs
    elif 0.5 < abs(min_dis) < 1.5:
        # min_dis = 0.5    collision_cost = w_cost_obs**1
        # min_dis = 1.5    collision_cost = w_cost_obs**0 = 1
        collision_cost = w_cost_obs ** ((0.5 - min_dis) + 1)
    else:
        collision_cost = 0

    return collision_cost


def CalcSTCoordinate(row, col, s_list, t_list):
    """
    该函数将计算矩阵节点的s t坐标
        params: row col节点在矩阵的行号和列号
                s_list t_list 采样间隔表
        return: s_value t_value 节点的s t坐标
    矩阵的(0,0) 代表的是最左上角的元素  此点s最大, t最小
    * 。。。。。。。。
    * 。。。。。。。。
    * 。。。。。。。。      可以发现，随着行数（矩阵的行数，不是坐标轴行数）增大（往下），s越来越小
    * 。。。。。。。。              随着列数增大（往右），t越来越大
    ****************
    """
    # 首先取出矩阵的行数
    m = len(s_list)
    # 0,0 对应的是 s的最大值 t的最小值
    s_value = s_list[m - row - 1]       # 按照老王matlab修改为Python，可能维度有错，注意检查！！
    t_value = t_list[col]
    return s_value, t_value


def generate_convex_space(dp_speed_s, dp_speed_t, path_index2s, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
                          obs_st_t_out_set, trajectory_kappa_init, max_lateral_accel=0.2*9.8):
    """
        该函数将计算出s,s_dot的上下界，开辟凸空间供二次规划使用
    """
    # dp_speed_s t 最多有16个，输出初始化n=16
    n = 16
    s_lb = np.ones(n) * -np.inf
    s_ub = np.ones(n) * np.inf
    s_dot_lb = np.ones(n) * -np.inf
    s_dot_ub = np.ones(n) * np.inf
    path_index2s_end_index = len(path_index2s)
    dp_speed_end_index = len(dp_speed_s)

    # path_index2s也是有缓冲的，找到有效的path_index2s的末尾的位置，赋值给path_index2s_end_index
    for k in range(1, len(path_index2s)):
        if path_index2s[k] == 0 and path_index2s[k - 1] != 0:
            path_index2s_end_index = k - 1
            break
        path_index2s_end_index = k

    # 找到dp_speed_s中有效的位置，赋值给dp_speed_end_index
    for k in range(len(dp_speed_s)):
        if np.isnan(dp_speed_s[k]):
            dp_speed_end_index = k - 1
            break

    # 该循环将施加车辆动力学约束
    for i in range(n):
        if np.isnan(dp_speed_s[i]):
            break
        cur_s = dp_speed_s[i]
        # 通过插值的方式找到cur_s所对应的曲率
        cur_kappa = interp1d(path_index2s[0:path_index2s_end_index], trajectory_kappa_init[0:path_index2s_end_index])(cur_s)
        # a = v^2 * k => v^2 = a/k  注意除0保护
        max_speed = np.sqrt(max_lateral_accel / (abs(cur_kappa) + 1e-10))
        min_speed = 0
        s_dot_lb[i] = min_speed
        s_dot_ub[i] = max_speed

    for i in range(len(obs_st_s_in_set)):
        if np.isnan(obs_st_s_in_set[i]):
            continue
        # 取s t 直线的中点，作为obs_s obs_t 的坐标
        obs_t = (obs_st_t_in_set[i] + obs_st_t_out_set[i]) / 2
        obs_s = (obs_st_s_in_set[i] + obs_st_s_out_set[i]) / 2
        # 计算障碍物的纵向速度
        obs_speed = (obs_st_s_out_set[i] - obs_st_s_in_set[i]) / (obs_st_t_out_set[i] - obs_st_t_in_set[i])
        # 插值找到当t = obs_t时，dp_speed_s 的值
        dp_s = interp1d([0] + list(dp_speed_t[0:dp_speed_end_index]), [0] + list(dp_speed_s[0:dp_speed_end_index]))(obs_t)

        # 找到dp_speed_t中 与obs_st_t_in_set(i)最近的时间，并将此时间的编号赋值给t_lb_index
        time1 = 0
        for j in range(len(dp_speed_t) - 1):
            if dp_speed_t[0] > obs_st_t_in_set[i]:
                # 如果障碍物切入时间比0.5秒还要短，那么t_lb_index = 0
                time1 = j
                break
            elif dp_speed_t[j] <= obs_st_t_in_set[i] < dp_speed_t[j + 1]:
                # 否则遍历dp_speed_t 找到与obs_st_t_in_set(i)最近的点的编号
                time1 = j
                break
        t_lb_index = time1

        # 找到dp_speed_t中 与obs_st_t_out_set(i)最近的时间，并将此时间的编号赋值给t_ub_index
        time2 = 0
        for j in range(len(dp_speed_t) - 1):
            if dp_speed_t[0] > obs_st_t_out_set[i]:
                time2 = j
                break
            elif dp_speed_t[j] <= obs_st_t_out_set[i] < dp_speed_t[j + 1]:
                time2 = j
                break
        t_ub_index = time2

        # 这里稍微做个缓冲，把 t_lb_index 稍微缩小一些，t_ub_index稍微放大一些
        t_lb_index = max(t_lb_index - 2, 3)     # 最低为3 因为碰瓷没法处理
        t_ub_index = min(t_ub_index + 2, dp_speed_end_index)

        if obs_s > dp_s:
            # 决策为减速避让
            for m in range(t_lb_index, t_ub_index + 1):
                # 在t_lb_index:t_ub_index的区间上 s的上界不可以超过障碍物st斜直线
                dp_t = dp_speed_t[m]
                s_ub[m] = min(s_ub[m], obs_st_s_in_set[i] + obs_speed * (dp_t - obs_st_t_in_set[i]))
        else:
            # 决策为加速超车
            for m in range(t_lb_index, t_ub_index + 1):
                # 在t_lb_index:t_ub_index的区间上 s的下界不能小于障碍物st斜直线
                dp_t = dp_speed_t[m]
                s_lb[m] = max(s_lb[m], obs_st_s_in_set[i] + obs_speed * (dp_t - obs_st_t_in_set[i]))

    return s_lb, s_ub, s_dot_lb, s_dot_ub


def speed_QP(plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t, s_lb, s_ub, s_dot_lb, s_dot_ub,
             w_cost_s_dot2=10, w_cost_v_ref=50, w_cost_jerk=500, reference_speed=50):
    """
        速度二次规划
        params: 规划起点plan_start_s_dot,plan_start_s_dot2
                动态规划结果dp_speed_s,dp_speed_t
                凸空间约束 s_lb,s_ub,s_dot_lb,s_dot_ub
                加速度代价权重，推荐速度代价权重，jerk代价权重w_cost_s_dot2,w_cost_v_ref,w_cost_jerk
                参考速度speed_reference
        return: 速度曲线 qp_s_init,qp_s_dot_init,qp_s_dot2_init,relative_time_init
    """
    # 由于dp的结果未必是16，该算法将计算dp_speed_end到底是多少
    dp_speed_end = 16
    for i in range(len(dp_speed_s)):
        if np.isnan(dp_speed_s[i]):
            dp_speed_end = i - 1
            break

    # 由于dp_speed_end实际上是不确定的，但是输出要求必须是确定长度的值，因此输出初始化选择dp_speed_end的最大值 + 规划起点作为输出初始化的规模
    n = 17
    qp_s_init = np.ones(n) * np.nan
    qp_s_dot_init = np.ones(n) * np.nan
    qp_s_dot2_init = np.ones(n) * np.nan
    relative_time_init = np.ones(n) * np.nan

    s_end = dp_speed_s[dp_speed_end]
    # 此时dp_speed_end表示真正有效的dp_speed_t的元素个数，取出dp_speed_t有效的最后一个元素作为规划的时间终点 记为recommend_T
    recommend_T = dp_speed_t[dp_speed_end]
    # qp的规模应该是dp的有效元素的个数 + 规划起点
    qp_size = dp_speed_end + 1

    # 连续性约束矩阵初始化，这里的等式约束应该是 Aeq'*X == beq 有转置
    Aeq = cvxopt.matrix(np.zeros((3 * qp_size, 2 * qp_size - 2)))
    beq = cvxopt.matrix(np.zeros((2 * qp_size - 2, 1)))
    # 不等式约束初始化
    lb = cvxopt.matrix(np.ones(3 * qp_size))
    ub = lb

    # 计算采样间隔时间dt
    dt = recommend_T / dp_speed_end

    A_sub = np.array([[1, 0],
                      [dt, 1],
                      [(1 / 3) * dt ** 2, (1 / 2) * dt],
                      [-1, 0],
                      [0, -1],
                      [(1 / 6) * dt ** 2, dt / 2]])

    for i in range(qp_size - 1):
        Aeq[3 * i:3 * i + 6, 2 * i:2 * i + 2] = A_sub

    # 这里初始化不允许倒车约束，也就是 s(i) - s(i+1) <= 0
    A = np.zeros((qp_size - 1, 3 * qp_size))
    b = np.zeros((qp_size - 1, 1))

    for i in range(qp_size - 1):
        A[i, 3 * i] = 1
        A[i, 3 * i + 3] = -1

    # 由于生成的凸空间约束s_lb s_ub不带起点，所以lb(i) = s_lb(i-1) 以此类推
    # 允许最小加速度为-6 最大加速度为4(基于车辆动力学)
    for i in range(1, qp_size):
        lb[3 * i] = s_lb[i - 1]
        lb[3 * i + 1] = s_dot_lb[i - 1]
        lb[3 * i + 2] = -6
        ub[3 * i] = s_ub[i - 1]
        ub[3 * i + 1] = s_dot_ub[i - 1]
        ub[3 * i + 2] = 4

    # 起点约束
    lb[0] = 0
    lb[1] = plan_start_s_dot
    lb[2] = plan_start_s_dot2
    ub[0] = lb[0]
    ub[1] = lb[1]
    ub[2] = lb[2]

    # 加速度代价 jerk代价 以及推荐速度代价
    A_s_dot2 = np.zeros((3 * qp_size, 3 * qp_size))
    A_jerk = np.zeros((3 * qp_size, qp_size - 1))
    A_ref = np.zeros((3 * qp_size, 3 * qp_size))

    A4_sub = np.array([[0], [0], [1], [0], [0], [-1]])

    for i in range(1, qp_size + 1):
        A_s_dot2[3 * i - 1, 3 * i - 1] = 1
        A_ref[3 * i - 2, 3 * i - 2] = 1

    for i in range(1, qp_size):
        A_jerk[3 * i - 3:3 * i + 3, i - 1:i] = A4_sub

    # 生成H F
    H = w_cost_s_dot2 * (A_s_dot2 @ A_s_dot2.T) + w_cost_jerk * (A_jerk @ A_jerk.T) + w_cost_v_ref * (A_ref @ A_ref.T)
    H = 2 * H
    f = cvxopt.matrix(np.zeros((3 * qp_size, 1)))
    for i in range(1, qp_size + 1):
        f[3 * i - 2] = -2 * w_cost_v_ref * reference_speed

    # 计算二次规划
    sol = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f), cvxopt.matrix(A), cvxopt.matrix(b), Aeq, beq)
    X = np.array(sol['x'])

    for i in range(1, qp_size + 1):
        qp_s_init[i - 1] = X[3 * i - 3]
        qp_s_dot_init[i - 1] = X[3 * i - 2]
        qp_s_dot2_init[i - 1] = X[3 * i - 1]
        relative_time_init[i - 1] = (i - 1) * dt

    return qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init


def increase_points(s_init, s_dot_init, s_dot2_init, relative_time_init):
    """
    该函数将增密s s_dot s_dot2
    为什么需要增密，因为控制的执行频率是规划的10倍，轨迹点如果不够密，必然会导致规划效果不好
    但是若在速度二次规划中点取的太多，会导致二次规划的矩阵规模太大计算太慢
    所以方法是在二次规划中选取少量的点优化完毕后，在用此函数增密

    params: s_init: initial position vector
            s_dot_init: initial velocity vector
            s_dot2_init: initial acceleration vector
            relative_time_init: initial time vector

    return: s: densified position vector
            s_dot: densified velocity vector
            s_dot2: densified acceleration vector
            relative_time: densified time vector
    """

    # Find t_end
    t_end = len(relative_time_init)
    for i in range(len(relative_time_init)):
        if np.isnan(relative_time_init[i]):
            t_end = i - 1
            break
    T = relative_time_init[t_end]

    # Densify
    n = 401
    dt = T / (n - 1)
    s = np.zeros(n)
    s_dot = np.zeros(n)
    s_dot2 = np.zeros(n)
    relative_time = np.zeros(n)

    tmp = 0
    for i in range(n):
        current_t = (i - 1) * dt
        for j in range(t_end - 1):
            if relative_time_init[j] <= current_t < relative_time_init[j + 1]:
                tmp = j
                break
        x = current_t - relative_time_init[tmp]
        s[i] = s_init[tmp] + s_dot_init[tmp] * x + (1 / 3) * s_dot2_init[tmp] * x ** 2 + (1 / 6) * s_dot2_init[tmp + 1] * x ** 2
        s_dot[i] = s_dot_init[tmp] + 0.5 * s_dot2_init[tmp] * x + 0.5 * s_dot2_init[tmp + 1] * x
        s_dot2[i] = s_dot2_init[tmp] + (s_dot2_init[tmp + 1] - s_dot2_init[tmp]) * x / (
                    relative_time_init[tmp + 1] - relative_time_init[tmp])
        relative_time[i] = current_t

    return s, s_dot, s_dot2, relative_time


def path_speed_merge(s, s_dot, s_dot2, relative_time, current_time, path_s, trajectory_x_init, trajectory_y_init,
                     trajectory_heading_init, trajectory_kappa_init):
    """
        该函数用于合并path和speed
    """
    # 由于path 是 60个点，speed 有 401个点，合并后，path和speed有401个点，因此需要做插值
    n = 401
    trajectory_x = np.zeros(n)
    trajectory_y = np.zeros(n)
    trajectory_heading = np.zeros(n)
    trajectory_kappa = np.zeros(n)
    trajectory_speed = np.zeros(n)
    trajectory_accel = np.zeros(n)

    index = 0
    # 由于trajectory_x_init中不是所有的值都有效，(有nan) 所以要找到有效值的范围
    while not np.isnan(trajectory_x_init[index]):
        index += 1
    index -= 1

    trajectory_time = np.zeros(n)

    for i in range(n - 1):
        # interp1 线性插值
        # 若 x = [1,2] y = [3,4]
        # interp1(x,y,1.5) 会得到3.5
        # interp1 的缺点是 在端点或超过端点的，结果为nan
        # interp1(x,y,2) 不会等于4 而是会等于nan
        # interp1(x,y,6) 也会等于nan，所以在端点处要单独处理
        trajectory_x[i] = np.interp(s[i], path_s[:index], trajectory_x_init[:index])
        trajectory_y[i] = np.interp(s[i], path_s[:index], trajectory_y_init[:index])
        trajectory_heading[i] = np.interp(s[i], path_s[:index], trajectory_heading_init[:index])
        trajectory_kappa[i] = np.interp(s[i], path_s[:index], trajectory_kappa_init[:index])
        trajectory_time[i] = relative_time[i] + current_time
        trajectory_speed[i] = s_dot[i]
        trajectory_accel[i] = s_dot2[i]

    # 端点单独处理
    trajectory_x[-1] = trajectory_x_init[-1]
    trajectory_y[-1] = trajectory_y_init[-1]
    trajectory_heading[-1] = trajectory_heading_init[-1]
    trajectory_kappa[-1] = trajectory_kappa_init[-1]
    trajectory_time[-1] = relative_time[-1] + current_time
    trajectory_speed[-1] = s_dot[-1]
    trajectory_accel[-1] = s_dot2[-1]

    return trajectory_x, trajectory_y, trajectory_heading, trajectory_kappa, trajectory_speed, trajectory_accel, trajectory_time
