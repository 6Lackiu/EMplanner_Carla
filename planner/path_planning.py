#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : path_planning.py

"""
当规划的周期和控制的周期不同时，需要进行控制接口设计和轨迹拼接
"""
import math
import numpy as np
import cvxopt
from planner import planning_utils
import time

"""
首先进行必要的坐标转换
"""
def frenet_2_x_y_theta_kappa(plan_start_s, plan_start_l, enriched_s_list: list, enriched_l_list: list,
                             frenet_path_opt: list,
                             s_map: list):
    """  已验证
    将增加采样点后动态规划得到的s-l路径转换为直角坐标系下路径信息x, y, theta, kappa
    param:  plan_start_s: 规划起点的s,l
            plan_start_l:
            enriched_s_list: 增加采样点后的s,l
            enriched_l_list:
            frenet_path_opt: 优化后的参考线[(x, y, theta, kappa), ... ]
            s_map: 参考线对应的s_map
    return: 直角坐标系下的路径列表，list类型[(x, y, theta, kappa), ...]
    """
    target_xy = []
    # 确定规划起点在s_map中的索引,并加入目标路径中
    proj_x, proj_y, proj_theta, proj_kappa, pre_match_index = cal_proj_point(plan_start_s, 0, frenet_path_opt, s_map)
    nor_v = np.array([-math.sin(proj_theta), math.cos(proj_theta)])  # 法向量
    cur_x, cur_y, = np.array([proj_x, proj_y]) + plan_start_l * nor_v
    target_xy.append((cur_x, cur_y))

    for i in range(len(enriched_l_list)):
        cur_s = enriched_s_list[i]
        cur_l = enriched_l_list[i]
        # cal_proj_point里面存在一个问题，在这里解决，就是如果动态规划纵向采样过长，cur_s会超出s_map的范围，即无法映射，这里直接截断
        if cur_s > s_map[-1]:
            break
        proj_x, proj_y, proj_theta, proj_kappa, pre_match_index = cal_proj_point(cur_s, pre_match_index,
                                                                                 frenet_path_opt, s_map)
        nor_v = np.array([-math.sin(proj_theta), math.cos(proj_theta)])  # 法向量
        cur_x, cur_y, = np.array([proj_x, proj_y]) + cur_l * nor_v
        target_xy.append((cur_x, cur_y))
    target_path = planning_utils.smooth_reference_line(target_xy)

    return target_path


def Quadratic_planning(l_min, l_max, plan_start_l, plan_start_dl, plan_start_ddl, dp_sampling_res=2,
                       w_cost_l=1000, w_cost_dl=10000, w_cost_ddl=3000, w_cost_dddl=150, w_cost_centre=250,
                       w_cost_end_l=40, w_cost_end_dl=40, w_cost_end_ddl=40,
                       host_d1=3, host_d2=3, host_w=3):
    """
    二次规划实现更加平滑的避障
    param:  l_min: 二次规划的上下界（点的凸空间）
            l_max:
            plan_start_l: 规划起点的信息
            plan_start_dl:
            plan_start_ddl:
            dp_sampling_res:  这里是动态规划采用的采样分辨率，作为参数来指导二次规划
            w_cost_l: 参考线代价，使规划的路线向参考线靠近
            w_cost_dl: 平滑代价
            w_cost_ddl:
            w_cost_dddl:
            w_cost_centre: 凸空间中央代价，使参考线向凸空间中心靠近
            w_cost_end_l: 终点的状态代价（希望path的终点状态为（0，0，0））
            w_cost_end_dl:
            w_cost_end_ddl:
            host_d1: 质心到车辆前后轴的距离
            host_d2:
            host_w: host的宽度
    return: qp_path_l, qp_path_dl, qp_path_ddl: 二次规划得到的轨迹信息
    """
    n = len(l_min)
    # print("***************n***********", n)
    """等式约束"""
    Aeq = np.zeros(shape=(2 * n - 2, 3 * n))
    beq = np.zeros(shape=(2 * n - 2, 1))
    ds = dp_sampling_res        # 纵向间隔
    Aeq_sub = np.array([[1, ds, ds ** 2 / 3, -1, 0, ds ** 2 / 6],   # ori: ds**2/2
                        [0, 1, ds / 2, 0, -1, ds / 2]])
    for i in range(n - 1):
        Aeq[i * 2: i * 2 + 2, i * 3:i * 3 + 6] = Aeq_sub

    """不等式约束"""
    A = np.zeros(shape=(8 * n, 3 * n))
    b = np.zeros(shape=(8 * n, 1))
    A_sub = np.array([[1, host_d1, 0],
                      [1, host_d1, 0],
                      [1, -host_d2, 0],
                      [1, -host_d2, 0],
                      [-1, -host_d1, 0],
                      [-1, -host_d1, 0],
                      [-1, host_d2, 0],
                      [-1, host_d2, 0]])

    front_index = math.ceil(host_d1 / ds)
    back_index = math.ceil(host_d2 / ds)
    for i in range(n):
        A[8 * i:8 * i + 8, 3 * i:3 * i + 3] = A_sub
        index1 = min(i + front_index, n - 1)
        index2 = max(i - back_index, 0)

        b_sub = np.array([[l_max[index1] - host_w / 2,
                           l_max[index1] + host_w / 2,
                           l_max[index1] - host_w / 2,
                           l_max[index1] + host_w / 2,
                           -l_min[index2] + host_w / 2,
                           -l_min[index2] - host_w / 2,
                           -l_min[index2] + host_w / 2,
                           -l_min[index2] - host_w / 2,
                           ]])
        b[8 * i:8 * (i + 1), 0] = b_sub

    """生成lb, ub对规划起点和终点做约束"""
    lb = np.ones(shape=(3 * n, 1)) * (-100000)  # 这部分我们只是为了约束起点和终点，
    # 本来是没必要约束中间节点的，但是表达式要考虑这些节点，所以对中间的节点就给了很大的约束空间，这个十万是自己随便给的，只要一个大值就可以
    ub = np.ones(shape=(3 * n, 1)) * 100000
    lb[0] = plan_start_l
    lb[1] = plan_start_dl
    lb[2] = plan_start_ddl
    ub[0] = plan_start_l
    ub[1] = plan_start_dl
    ub[2] = plan_start_ddl

    lb[3 * n - 1] = 0
    lb[3 * n - 2] = 0
    lb[3 * n - 3] = 0
    ub[3 * n - 1] = 0
    ub[3 * n - 2] = 0
    ub[3 * n - 3] = 0

    """将起点约束整合到不等式约束中"""
    A_s = np.concatenate((np.identity(3 * n), -np.identity(3 * n)))  # (6n, 3n)
    b_s = np.concatenate((ub, -lb))  # (6n, 1)
    G = np.concatenate((A, A_s))
    h = np.concatenate((b, b_s))
    # print("G, h", G.shape, h.shape)

    """生成H_L, H_DL, H_DDL, H_CENTRE"""
    H_L = np.zeros(shape=(3 * n, 3 * n))
    H_DL = np.zeros(shape=(3 * n, 3 * n))
    H_DDL = np.zeros(shape=(3 * n, 3 * n))
    # H_CENTRE = np.zeros(shape=(3 * n, 3 * n))
    for i in range(n):
        H_L[3 * i, 3 * i] = 1
        H_DL[3 * i + 1, 3 * i + 1] = 1
        H_DDL[3 * i + 2, 3 * i + 2] = 1
    H_CENTRE = H_L

    """生成H_DDDL"""
    H_DDDL = np.zeros(shape=(n - 1, 3 * n))
    H_dddl_sub = np.array([[0, 0, -1, 0, 0, 1]])
    for i in range(n - 1):
        H_DDDL[i, 3 * i:3 * i + 6] = H_dddl_sub
    """生成H_L_END, H_DL_END, H_DDL_END"""
    H_L_END = np.zeros(shape=(3 * n, 3 * n))
    H_DL_END = np.zeros(shape=(3 * n, 3 * n))
    H_DDL_END = np.zeros(shape=(3 * n, 3 * n))
    H_L_END[3 * n - 3, 3 * n - 3] = 1
    H_DL_END[3 * n - 2, 3 * n - 2] = 1
    H_DDL_END[3 * n - 1, 3 * n - 1] = 1
    """生成二次规划的H"""
    H = w_cost_l * (H_L.T @ H_L) + w_cost_dl * (H_DL.T @ H_L) + w_cost_ddl * (H_DDL.T @ H_DDL) + w_cost_dddl * (
            H_DDDL.T @ H_DDDL) \
        + w_cost_centre * (H_CENTRE.T @ H_CENTRE) + w_cost_end_l * (H_L_END.T @ H_L_END) \
        + w_cost_end_dl * (H_DL_END.T @ H_DL_END) + w_cost_end_ddl * (H_DDL_END.T @ H_DDL_END)
    H = 2 * H
    """生成f"""
    f = np.zeros(shape=(3 * n, 1))
    # 参考线还是选择稳定的，如果以动态规划的dp_path_l为参考线会不稳定
    centre_line = (np.array(l_min) + np.array(l_max)) / 2

    for i in range(n):
        f[3 * i] = -2 * centre_line[i]
    f = w_cost_centre * f
    """二次规划"""
    begin_time = time.time()
    cvxopt.solvers.options['show_progress'] = False  # 程序没有问题之后不再输出中间过程
    # 计算时要将输入转化为cvxopt.matrix
    # 该方法返回值是一个字典类型，包含了很多的参数，其中x关键字对应的是优化后的解
    res = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f),
                            G=cvxopt.matrix(G), h=cvxopt.matrix(h),
                            A=cvxopt.matrix(Aeq), b=cvxopt.matrix(beq)
                            )
    print("the time cost of cvxopt.solvers.qp", time.time() - begin_time)
    qp_path_l = res['x'][0::3]      # 从第1个元素开始，每隔3个元素取1个
    qp_path_dl = res['x'][1::3]
    qp_path_ddl = res['x'][2::3]
    return list(qp_path_l), list(qp_path_dl), list(qp_path_ddl)


def cal_lmin_lmax(dp_path_s, dp_path_l, obs_s_list, obs_l_list, obs_length, obs_width):
    """
    计算二次规划的边界
    param:  dp_path_s: 动态规划得到的s,l列表
            dp_path_l:
            obs_s_list: 障碍物的位置s,l
            obs_l_list:
            obs_length: 障碍物的长和宽
            obs_width:
    return: lmin_list, lmax_list, 轨迹上每个s对应的上下界
    """
    lmin = -6 * np.ones(len(dp_path_s))     # 老王：-8
    lmax = 6 * np.ones(len(dp_path_s))      # 老王：8
    offset = 2      # 偏移量 解释如下
    # 先对障碍物进行处理
    for i in range(len(obs_s_list)):
        obs_s_min = obs_s_list[i] - obs_length / 2  # 障碍物尾部和头部的s
        obs_s_max = obs_s_list[i] + obs_length / 2
        obs_s_min_index = np.argmin(np.abs(np.array(dp_path_s) - obs_s_min)) + offset  # 车尾的s在动态规划的路径中的投影点索引 ori:1
        obs_s_max_index = np.argmin(np.abs(np.array(dp_path_s) - obs_s_max)) + offset  # 车头的s索引 ori:1
        """这里采用的计算车辆头部和尾部位置的方法是一种近似，由于动态规划的路径分辨率是sampling_res，所以最大误差是sampling_res/2米
        *****二次规划的结果中出现了一个比较棘手的问题：障碍物在规划的路径中总是位于避障曲线的右侧，如下图（凸起的曲线就是避障曲线）
                              。。。。。。。
                            。           。
        尾部         。。。。。       <obs> 。。。。。。。    头部
                    理想情况障碍物应该位于避障曲线的正中心
                                。。。。。。。
                              。           。
        尾部           。。。。。     <obs>   。。。。。。。  头部
        分析原因得到的结果是l的边界约束中， lmin和lmax约束的障碍物偏左，所以导致实际的曲线相对于障碍物偏左
        解决办法自然是将约束障碍物的边界向右侧调整，由于起点和终点的计算本身存在误差，所以这里也只能大致调整
        我所选择的策略是车尾向右偏移2个路径分辨率,即索引加2;头部加2的目的是适当扩充右边界，防止车辆离开障碍物的时候过于靠近障碍物
        """

        print("obs_index", obs_s_min_index, obs_s_max_index)
        centre_index = np.argmin(np.abs(np.array(dp_path_s) - obs_s_list[i]))  # 车辆质心的s索引，
        # 就是用来判断障碍物在动态规划路线的哪一侧
        path_l = dp_path_l[centre_index]  # 质心所在位置的l，就是在障碍物所在的s处dp规划车辆应该处的位置
        if path_l < obs_l_list[i]:
            """决策为向右绕过障碍物"""
            print("决策为向右绕过障碍物")
            for j in range(obs_s_min_index, obs_s_max_index + 1):
                # l_max[j]为所有决策为向右绕过障碍物的l边界的最小值
                lmax[j] = min(lmax[j], obs_l_list[i] - obs_width / 2)
        else:
            """决策为向左绕过障碍物"""
            print("决策为向左绕过障碍物")
            for j in range(obs_s_min_index, obs_s_max_index + 1):
                # l_min[j]为所有决策为向左绕过障碍物的l边界的最大值
                lmin[j] = max(lmin[j], obs_l_list[i] + obs_width / 2)
    return lmin, lmax


def DP_algorithm(obs_s_list: list, obs_l_list: list,
                 plan_start_s, plan_start_l, plan_start_dl, plan_start_ddl, sampling_res=2,
                 w_collision_cost=1e12, w_smooth_cost=[300, 1000, 5000], w_reference_cost=20,
                 row=12, col=6, sample_s=15, sample_l=1.5):
    """  已验证
    采用动态规划进行路径规划, 声明一下动态规划和五次多项式用到的dl都是对位矢s的导数，跟坐标变换时的定义有点差别
    param:  obs_s_list: 障碍物在参考线上的位矢信息
            obs_l_list: 障碍物在参考线上的l， 即投影点和实际点之间的长度
            plan_start_s: 规划起点的s
            plan_start_l: 规划起点的l
            plan_start_dl: 规划起点的dl/ds
            plan_start_ddl: 规划起点的d(dl/ds)/ds
            sampling_res: 在相邻两个动态规划点之间五次多项式曲线上的间隔（轨迹增密的间隔）
                          过小计算量大，过大无法及时避障
            w_collision_cost: 障碍物距离代价权重
            w_smooth_cost: 平滑代价权重，列表类型[w_dl_cost, w_ddl_cost, w_dddl_cost]考虑到s-l曲线的三阶导数
            w_reference_cost: 参考线代价权重
            row: 动态规划采样点的行数和列数
            col:
            sample_s: 沿着s方向的采样间隔
            sample_l: 沿着l方向的采样间隔
    return: 规划的得到的s-l路径信息，dp_path_s, dp_path_l 都是列表类型,不包括规划起点
    """
    if len(obs_s_list):  # obs_s_list的长度大于零则表明障碍物存在
        # 声明一个二维数组记录每个采样点的cost,初始化为无穷大
        cost = np.ones(shape=(row, col)) * np.inf
        # 声明另一个二维数组，记录规划起点距离当前位置的最短路径的前一个位置
        # ori: pre_node_index = (row >> 1) ** np.ones(shape=(row, col), dtype="int32")     # row >> 1相当于row除2并向下取整
        pre_node_index = np.ones(shape=(row, col), dtype="int32")   # 初始化，所有节点的上个最优节点的行号都是0，也就是起点
        # 计算起点到第一列的cost
        for i in range(row):
            cost[i][0] = cal_start_cost(obs_s_list, obs_l_list,
                                        begin_s=plan_start_s, begin_l=plan_start_l,
                                        begin_dl=plan_start_dl, begin_ddl=plan_start_ddl,
                                        cur_node_row=i, row=row, sample_s=sample_s, sample_l=sample_l,
                                        w_cost_collision=w_collision_cost,
                                        w_cost_smooth=w_smooth_cost,
                                        w_cost_ref=w_reference_cost)
            # 现实的交通规则是多车道的情况下，左侧车道允许的最大速度高于右侧车道
            # 因此在计算cost时，对于左侧车道的采样点，会额外加上一个较大的cost值以降低其优先级，从而更倾向于选择右侧车道。
            # 通过判断当前采样点所在的行数是否小于总行数的一半（即是否在地图的左侧）来实现的。
            if i < (row >> 1):
                cost[i][0] += 10000
        # 计算后面几列的cost
        for j in range(1, col):
            # print("current col: %d" % j)
            for i in range(row):
                # print("current row: %d" % i)
                # 计算当前node的sl
                cur_node_s = plan_start_s + (j + 1) * sample_s
                cur_node_l = ((row + 1) / 2 - 1 - i) * sample_l
                # 在carla的坐标环境下建立的frenet坐标，前进方向的左侧l是负的，所以row/2 ~ row之间的行号代表左转，0~row/2代表右转
                # 遍历前一列的节点
                for k in range(row):
                    pre_node_s = plan_start_s + j * sample_s
                    pre_node_l = ((row + 1) / 2 - 1 - k) * sample_l
                    cost_neighbor = cal_neighbor_cost(obs_s_list, obs_l_list, pre_node_s, pre_node_l,
                                                      cur_node_s=cur_node_s, cur_node_l=cur_node_l, sample_s=sample_s,
                                                      w_cost_collision=w_collision_cost,
                                                      w_cost_smooth=w_smooth_cost,
                                                      w_cost_ref=w_reference_cost)
                    # 起点到上一个节点的最小代价为cost[k][j - 1]
                    pre_min_cost = cost[k][j - 1]
                    # 再加上node[k][j-1]到node[i][j]的代价(k=1,2,3,...)中最小的
                    cost_temp = pre_min_cost + cost_neighbor
                    if i < (row >> 1):  # 现实的交通规则是多车道的情况下，左侧车道允许的最大速度高于右侧车道
                        cost_temp += 10000
                    # print(cost_temp)
                    if cost_temp < cost[i][j]:
                        cost[i][j] = cost_temp
                        pre_node_index[i][j] = k
        # 确定最优路径
        DP_row_index_list = []
        min_index = cost[:, -1].argmin()    # 选择最后一列中最小元素的索引
        # print(min_index)
        if cost[:, -1].min() > w_collision_cost:  # 找不到无碰撞路径，提示运行错误
            print("********************     can't find a feasible path      ********************")
            # raise RuntimeError("********************can't find a feasible path*********************************")
        DP_row_index_list.append(min_index)
        for i in range(col - 1, 0, -1):  # 倒数第二列col-1遍历到第一列,别把索引提取错了，
            # DP_row_index_list中第一列记录的是起点的相对位置，是车辆已经通过的位置，不在路径考虑的范围内
            # print(min_index)
            min_index = pre_node_index[min_index][i]    # 通过后一列的min_index行更新当前列的min_index行
            DP_row_index_list.append(min_index)
        # DP_row_index_list中的第一位是规划起点后的采样的第一列的某个节点，这里没有加入规划起点，规划起点在enrich_DP_s_l中再考虑
        DP_row_index_list.reverse()
    else:  # 没有障碍物的情况，理想路径就是l=0的一系列点
        DP_row_index_list = list(np.ones(col) * ((row + 1) / 2 - 1))  # 没有障碍物的条件下，就走直线
    # 将数组中的索引转化为s-l
    print("DP_row_index_list: ", DP_row_index_list)
    DP_s_list = []
    DP_l_list = []
    for i in range(len(DP_row_index_list)):
        DP_s_list.append(plan_start_s + (i + 1) * sample_s)
        DP_l_list.append(((row + 1) / 2 - 1 - DP_row_index_list[i]) * sample_l)
    # 初步的动态规划密度不够，相邻点之间在五次多项式的基础上间隔sampling_res米进行采样
    enriched_dp_s, enriched_dp_l = enrich_DP_s_l(DP_s_list, DP_l_list,
                                                 plan_start_s, plan_start_l, plan_start_dl, plan_start_ddl,
                                                 resolution=sampling_res)
    return enriched_dp_s, enriched_dp_l


def enrich_DP_s_l(DP_s_list, DP_l_list, plan_start_s, plan_start_l, plan_start_dl, plan_start_ddl, resolution=1):
    """  已验证
    由于采样点有时候比较稀疏，所以需要在相邻五次多项式曲线上进行采样，增加轨迹带点数（轨迹增密）
    param:  DP_s_list: 动态规划得到的s,l
            DP_l_list:
            plan_start_s: 规划起点的s, l, dl, ddl
            plan_start_l:
            plan_start_dl:
            plan_start_ddl:
            resolution:  在五次多项式上的采样间隔
    return: enriched_s_list, enriched_l_list 增密后的s_list l_list
    """
    enriched_s_list = []
    enriched_l_list = []
    start_s = plan_start_s
    start_l = plan_start_l
    start_dl = plan_start_dl
    start_ddl = plan_start_ddl

    end_s = DP_s_list[0]
    end_l = DP_l_list[0]
    end_dl = 0
    end_ddl = 0
    coeffi = planning_utils.cal_quintic_coefficient(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s)

    """ 采样间隔为resolution， 采样的个数就是终点和起点的差值取整，下面用矩阵运算，加快速度"""
    # 考虑规划起点
    s = start_s + np.arange(0, int(end_s - start_s), resolution)  # 采样间隔为resolution米时，采样的个数就是终点和起点的差值取整
    l0 = coeffi[0] + coeffi[1] * s + coeffi[2] * (s ** 2) + coeffi[3] * (s ** 3) + coeffi[4] * (s ** 4) + \
        coeffi[5] * (s ** 5)
    enriched_s_list = enriched_s_list + list(s)
    enriched_l_list = enriched_l_list + list(l0)
    # 规划起点后的其他点
    for i in range(1, len(DP_s_list)):
        start_s = DP_s_list[i - 1]
        start_l = DP_l_list[i - 1]
        start_dl = 0
        start_ddl = 0

        end_s = DP_s_list[i]
        end_l = DP_l_list[i]
        end_dl = 0
        end_ddl = 0
        coeffi = planning_utils.cal_quintic_coefficient(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s)

        s = start_s + np.arange(0, int(end_s - start_s), resolution)  # 采样间隔为一米时， 采样的个数就是终点和起点的差值取整
        li = coeffi[0] + coeffi[1] * s + coeffi[2] * (s ** 2) + coeffi[3] * (s ** 3) + coeffi[4] * (s ** 4) + \
            coeffi[5] * (s ** 5)
        enriched_s_list = enriched_s_list + list(s)
        enriched_l_list = enriched_l_list + list(li)

    enriched_s_list += [end_s]
    enriched_l_list += [end_l]

    return enriched_s_list, enriched_l_list


def cal_start_cost(obs_s_list, obs_l_list,
                   begin_s, begin_l, begin_dl, begin_ddl,
                   cur_node_row, row,
                   sample_s, sample_l,
                   w_cost_collision, w_cost_smooth, w_cost_ref
                   ):
    """  已验证
    计算当前位置的cost
    参数和DP_algorithm基本一致
    param:  obs_s_list: 障碍物的s-l信息
            obs_l_list:
            begin_s: 规划起点的s-l信息
            begin_l:
            begin_dl:
            begin_ddl:
            cur_node_row:当前节点所在行
            row: 动态规划采样点的行数
            sample_s: 沿着s方向的采样间隔和沿着l方向的采样间隔
            sample_l:
            w_cost_collision: 障碍物距离代价权重
            w_cost_smooth: 平滑代价权重，列表类型【w_dl_cost, w_ddl_cost, w_dddl_cost】考虑到s-l曲线的三阶导数
            w_cost_ref: 参考线代价权重
    return: 规划起点到当前位置的代价cost
    """
    """下面是动态规划的一个示意图，b代表规划起点,中间是参考线，总共有采样五行（参考线上的采的点没有标出来）
    sample_s, sample_l分别代表沿着s方向的采样间隔和沿着l方向的采样间隔
    将采样的点存入矩阵，然后进行动态规划
        . . . . .
        . . . . .
     b  ---------reference line
        . . . . .
        . . . . .
    """
    # 五次多项式起点的边界条件
    start_l = begin_l
    start_dl = begin_dl
    start_ddl = begin_ddl
    start_s = begin_s
    # 终点的边界条件
    # (row + 1) / 2 - 1 代表中间的行号,比如row=3,则:
    # . . .
    # . . .  中间的行号 = (row + 1) / 2 - 1 = 1
    # . . .
    end_l = ((row + 1) / 2 - 1 - cur_node_row) * sample_l  # 动态规划是在参考线两边均匀采点，l是有正负之分的，这里矩阵的行列都是整的，所以对应上l要做转换
    end_dl = 0
    end_ddl = 0
    end_s = begin_s + sample_s

    # 计算五次多项式的系数
    coeffi = planning_utils.cal_quintic_coefficient(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s)
    # 在五次多项式构成的曲线上采样十个点计算cost
    s = np.zeros(shape=(10, 1))
    # l = np.zeros(shape=(10, 1))  # reserve memory space
    # dl = np.zeros(shape=(10, 1))
    # ddl = np.zeros(shape=(10, 1))
    # dddl = np.zeros(shape=(10, 1))
    # 计算s
    for i in range(10):
        s[i][0] = start_s + i * sample_s / 10  # 先从起点离散采样s, 然后通过五次多项式计算每一点的l, dl, ddl, dddl
    l = coeffi[0] + coeffi[1] * s + coeffi[2] * (s ** 2) + coeffi[3] * (s ** 3) + coeffi[4] * (s ** 4) + \
        coeffi[5] * (s ** 5)
    dl = coeffi[1] + 2 * coeffi[2] * s + 3 * coeffi[3] * (s ** 2) + 4 * coeffi[4] * (s ** 3) + 5 * coeffi[5] * (s ** 4)
    ddl = 2 * coeffi[2] + 6 * coeffi[3] * s + 12 * coeffi[4] * (s ** 2) + 20 * coeffi[5] * (s ** 3)
    dddl = 6 * coeffi[3] + 24 * coeffi[4] * s + 60 * coeffi[5] * (s * 2)
    cost_smooth = w_cost_smooth[0] * (dl.T @ dl) + w_cost_smooth[1] * (ddl.T @ ddl) + w_cost_smooth[2] * (dddl.T @ dddl)  # 平滑代价
    cost_ref = w_cost_ref * (l.T @ l)  # 参考线代价
    # 计算障碍物代价
    cost_collision = 0
    for i in range(len(obs_s_list)):  # 把每个障碍物都遍历一遍
        d_lon = obs_s_list[i] - s
        d_lat = obs_l_list[i] - l
        square_d = d_lon ** 2 + d_lat ** 2  # 这里直接在曲线上近似,实际上应该是计算两点之间的直线，在直角坐标系下进行（x1-x2)**2+(y1-y2)**2，
        # 但是考虑量采样点之间的五次多项式一般较平缓，我们就直接近似，简化计算
        # print(square_d)
        cost_collision += cal_obs_cost(w_cost_collision, square_d)
        # TODO 判断障碍物的左右
        if cost_collision > cost_collision:
            break

    return cost_smooth + cost_collision + cost_ref


def cal_neighbor_cost(obs_s_list, obs_l_list, pre_node_s, pre_node_l,
                      cur_node_s, cur_node_l, sample_s,
                      w_cost_collision, w_cost_smooth, w_cost_ref):
    """  已验证
    计算当前位置的cost
    参数和DP_algorithm基本一致
    param:  obs_s_list: 障碍物的s-l信息
            obs_l_list:
            pre_node_s: 的s-l信息
            pre_node_l:
            cur_node_s:当前节点s
            cur_node_l:当前节点l
            sample_s: 沿着s方向的采样间隔
            w_cost_collision: 障碍物距离代价权重
            w_cost_smooth: 平滑代价权重，列表类型[w_dl_cost, w_ddl_cost, w_dddl_cost]考虑到s-l曲线的三阶导数
            w_cost_ref: 参考线代价权重
    return: 规划起点到当前位置的代价cost
    """
    """下面是动态规划的一个示意图，b代表规划起点,中间是参考线，总共有采样五行（参考线上的采的点没有标出来）
    sample_s, sample_l分别代表沿着s方向的采样间隔和沿着l方向的采样间隔
    将采样的点存入矩阵，然后进行动态规划
        . . . . .
        . . . . .
     b  ---------reference line
        . . . . .
        . . . . .
    """
    # 五次多项式起点的边界条件
    start_l = pre_node_l
    start_dl = 0
    start_ddl = 0
    start_s = pre_node_s
    # 终点的边界条件
    end_l = cur_node_l  # 动态规划是在参考线两边均匀采点，l是有正负之分的，这里矩阵的行列都是整的，所以对应上l要做转换
    end_dl = 0
    end_ddl = 0
    end_s = cur_node_s

    # 计算五次多项式的系数
    coeffi = planning_utils.cal_quintic_coefficient(start_l, start_dl, start_ddl,
                                                    end_l, end_dl, end_ddl, start_s, end_s)
    # 在五次多项式构成的曲线上采样十个点计算cost
    s = np.zeros(shape=(10, 1))
    # l = np.zeros(shape=(10, 1))  # reserve memory space
    # dl = np.zeros(shape=(10, 1))
    # ddl = np.zeros(shape=(10, 1))
    # dddl = np.zeros(shape=(10, 1))
    # 计算s
    for i in range(10):
        s[i][0] = start_s + i * sample_s / 10
    l = coeffi[0] + coeffi[1] * s + coeffi[2] * (s ** 2) + coeffi[3] * (s ** 3) + coeffi[4] * (s ** 4) + coeffi[5] * (
            s ** 5)
    dl = coeffi[1] + 2 * coeffi[2] * s + 3 * coeffi[3] * (s ** 2) + 4 * coeffi[4] * (s ** 3) + 5 * coeffi[5] * (s ** 4)
    ddl = 2 * coeffi[2] + 6 * coeffi[3] * s + 12 * coeffi[4] * (s ** 2) + 20 * coeffi[5] * (s ** 3)
    dddl = 6 * coeffi[3] + 24 * coeffi[4] * s + 60 * coeffi[5] * (s * 2)
    cost_smooth = w_cost_smooth[0] * (dl.T @ dl) + w_cost_smooth[1] * (ddl.T @ ddl) + w_cost_smooth[2] * (dddl.T @ dddl)  # 平滑代价
    # print(l)
    cost_ref = w_cost_ref * (l.T @ l)  # 参考线代价
    # 计算障碍物代价
    cost_collision = 0
    for i in range(len(obs_s_list)):  # 把每个障碍物都遍历一遍
        d_lon = obs_s_list[i] - s
        d_lat = obs_l_list[i] - l
        square_d = d_lon ** 2 + d_lat ** 2  # 这里直接在曲线上近似,实际上应该是计算两点之间的直线，在直角坐标系下进行（x1-x2)**2+(y1-y2)**2，
        # 但是考虑量采样点之间的五次多项式一般较平缓，我们就直接近似，简化计算,
        # TODO 这里有时会出现问题，就是曲线扭曲时误差较大，导致无法找到无碰撞路径，后面要考虑把这部分优化掉
        cost_collision += cal_obs_cost(w_cost_collision, square_d)
    print(cost_smooth, cost_collision, cost_ref)
    return cost_smooth + cost_collision + cost_ref


def cal_obs_cost(w_cost_collision, square_d: np.ndarray, danger_dis=4, safe_dis=6):
    """  已验证
    计算障碍物的代价
    暂时设定为四米以外，不会碰撞 cost=0
    4米到3米 cost=1000/square_d
    3米以内 cost=w_cost_collision
    param:  w_cost_collision: 障碍物碰撞的代价系数
            square_d: 障碍物与五次多项式上离散点的距离， np.array类型shape=(10,1)
            danger_dis: 障碍物与五次多项式上离散点的距离， np.array类型shape=(10,1)
            safe_dis: 障碍物与五次多项式上离散点的距离， np.array类型shape=(10,1)
    return: 障碍物的代价
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

def cal_proj_point(s, pre_match_index, frenet_path_opt: list, s_map: list):
    """
    确定给定的s在参考线上的投影点的路径信息
    param:  s: 要计算投影点的位矢s
            pre_match_index: 上个投影点的索引
            frenet_path_opt:  优化后的参考线
            s_map: 位矢map
    return: 投影点的路径信息 (x, y, theta, kappa), 起始匹配点的索引
    """
    # 确定s在s_map中的索引
    start_s_match_index = pre_match_index
    while s_map[start_s_match_index + 1] < s:  # 这里存在一点问题，如果动态规划采样点过长，会超出s_map的范围(在frenet_2_x_y_theta_kappa中解决)
        start_s_match_index += 1
        # if start_s_match_index == (len(s_map) - 1):
        #     break
    mp_x, mp_y, mp_theta, mp_kappa = frenet_path_opt[start_s_match_index]  # 取出投影点的路径信息
    ds = s - s_map[start_s_match_index]  # 计算规划起点的投影点和匹配点之间的位矢
    mp_tou_v = np.array([math.cos(mp_theta), math.sin(mp_theta)])
    r_m = np.array([mp_x, mp_y])  # 匹配点位矢
    proj_x, proj_y = r_m + ds * mp_tou_v  # 近似投影点位置矢量
    proj_theta = mp_theta + mp_kappa * ds
    proj_kappa = mp_kappa
    res = (proj_x, proj_y, proj_theta, proj_kappa, start_s_match_index)
    return res
