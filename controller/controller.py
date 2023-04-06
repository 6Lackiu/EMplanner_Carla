#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : controller.py

import cvxopt
import numpy as np
import math
import carla
from planner.utils import cal_heading_kappa
from collections import deque

# print_flag = False
print_flag = True

"""
MPC 控制算法进行横向控制
总共分为5个模块
1 A, B，C计算模块
“
根据整车参数(a, b, Cf, Cr, m, Iz)和V_x,通过公式计算A,B,C; 具体公式间第八讲总结
a，b是前后轮中心距离车质心的距离
CF, Cr是前后轮的侧偏刚度
m是车的质量
Iz是车的转动惯量
V_x是车辆速度在车轴方向的分量
”

2 e_rr 和 k_r计算模块
“
根据当前测量位置车辆的（x, y, fi, V_y, fi'）和规划位置的（x_r, y_r, theta_r, k_r）计算误差e_rr, 和投影点的曲率
由于车辆不具有路况的观察能力，且控制具有一定的滞后性，因此实际中需要加上一个预测模块，预测当前的车辆状态在到达下一个控制周期之前的可能位置
"

3 A_bar, B_bar, C_bar计算模块
“
采用终点欧拉法和向前欧拉法对状态转移方程进行离散化
A_bar = (I-A*dt)e-1 @ (I + A*dt)
B_bar = (I-A*dt)e-1 @ B * dt
C_bar = (I-A*dt)e-1 @ C * k_r * Vx
注释：这里将非线性项theta_r_dao看成常数， theta_r_dao = d_theta_r/dt = (d_theta_r/ds)*(ds/dt) = k_r * s_dao
在无飘移的假设下，s_dao 近似等于 Vx， 所以C_bar表达式如上
”

4 将代价函数转化为二次型，计算M, C, Cc， Q_bar, R_bar, H, E
"
Xk = M @ xk + C @ uk + Cc
H = R_bar + C.T @ Q_bar @ C
E = C.T @ Q_bar @ Cc + C.T @ Q_bar @ M @ xk
J_min = uk.T @ H @ uk + 2* E.T @ uk
这里还需要根据实际情况进行必要的约束， 我选择的是-1 < u < 1
再将约束转化为对应的 G @ uk < 1 的形式
实际的标准二次型为0.5*x'.H.x + f'x
所以令 
H = 2H
f = 2*E
"

5 最终控制模块计算控制量u
“
最后求解二次规划求出控制区间的所有控制量，将第一个控制区间的控制量取出来交给控制器即可
”
"""


class Lateral_MPC_controller(object):
    def __init__(self, ego_vehicle, vehicle_para, pathway_xy_theta_kappa):
        self._vehicle_state = None
        self._vehicle_para = vehicle_para
        self._vehicle = ego_vehicle  # type: carla.Vehicle
        self._vehicle_Vx = 0
        self._target_path = pathway_xy_theta_kappa
        self._N = 6  # 预测区间
        self._P = 2  # 控制区间
        self._n = 4  # 状态变量长度
        self.A = np.zeros(shape=(self._n, self._n), dtype="float64")
        self.B = np.zeros(shape=(self._n, 1), dtype="float64")
        self.C = np.zeros(shape=(self._n, 1), dtype="float64")
        self.A_bar = None  # 离散化的A,B,C矩阵
        self.B_bar = None
        self.C_bar = None
        self.k_r = None  # 曲率
        self.e_rr = None
        self.min_index = 0
        # debug 变量， 用于显示预测点和投影点的位置
        self.x_pre = 0
        self.y_pre = 0
        self.x_pro = 0
        self.y_pro = 0

    def cal_vehicle_info(self):
        """
        通过调用方法获取车辆的状态信息
        :return: None
        """
        vehicle_loc = self._vehicle.get_location()
        x, y = vehicle_loc.x, vehicle_loc.y
        fi = self._vehicle.get_transform().rotation.yaw * (math.pi / 180)  # 车身横摆角，车轴和x轴的夹角
        V = self._vehicle.get_velocity()  # 航向角是车速与x轴夹角
        V_length = math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)
        beta = (math.atan2(V.y, V.x) - fi)  # 质心侧偏角，车速和车轴之间的夹角
        # print("beta", beta, "theta", math.atan2(V.y, V.x), "fi", fi)
        Vy = V_length * math.sin(beta)  # 车速在车身坐标系下的分量
        """这里又一个非常奇怪的错误，当Vx=-0.0022左右时，
        求解出来的A_bar有些值特别大，进而导致C和H的部分之非常大，引起H的秩为1的情况，使得二次规划求解崩溃
        这里特殊处理一下，让Vx的绝对值不能小于0.005，刚好也能避免Vx为零时的错误，后面就不用担心为零的错误了"""
        if V_length * math.cos(beta) < 0:
            Vx = -max(abs(V_length * math.cos(beta)), 0.005)
        else:
            Vx = max(V_length * math.cos(beta), 0.005)
        # print("Vx", Vx, "Vy", Vy)
        fi_dao = self._vehicle.get_angular_velocity().z * (math.pi / 180)
        self._vehicle_state = (x, y, fi, Vy, fi_dao)
        self._vehicle_Vx = Vx

    def cal_A_B_C_fun(self):
        """
        根据整车参数vehicle_para和V_x,通过公式计算A,B; 具体公式间第八讲总结
              vehicle_para: vehicle_para = (a, b, Cf, Cr, m, Iz)
                            a，b是前后轮中心距离车质心的距离
                            CF, Cr是前后轮的侧偏刚度
                            m是车的质量
                            Iz是车的转动惯量

              V_x: V_x是车辆速度在车轴方向的分量
        :return: 矩阵A和B, np.array type
        A的维度4*4
        B的维度4*1
        """
        V_x = self._vehicle_Vx
        V_x = V_x
        (a, b, Cf, Cr, m, Iz) = self._vehicle_para

        self.A[0][1] = 1

        self.A[1][1] = (Cf + Cr) / (m * V_x)
        self.A[1][2] = -(Cf + Cr) / m
        self.A[1][3] = (a * Cf - b * Cr) / (m * V_x)

        self.A[2][3] = 1

        self.A[3][1] = (a * Cf - b * Cr) / (Iz * V_x)
        self.A[3][2] = -(a * Cf - b * Cr) / Iz
        self.A[3][3] = (a * a * Cf + b * b * Cr) / (Iz * V_x)

        self.B[1][0] = -Cf / m
        self.B[3][0] = -a * Cf / Iz

        self.C[1][0] = (a*Cf + b*Cr)/(m*V_x) - V_x
        self.C[3][0] = (a**2*Cf + b**2*Cr)/(Iz*V_x)

    def cal_coefficient_of_discretion_fun(self):
        """
        根据Q,R和A,B计算K, 通过迭代黎卡提方程求解， P = Q + A^PA - A^PB(R+B^PB)'B^PA.其中A^是求转置，A'是求逆
        :param Q: 是误差代价的权重对应的对角矩阵4*4,Q越大算法的性能越好，但是会牺牲算法稳定性导致最终控制量u很大
        :param R: 控制代价的权重对应的对角矩阵1*1， R越大越平稳，变化越小
        :param A: cal_A_B_fun模块的输出4*4
        :param B:
        :return: K, np.array类型
        """
        ts = 0.1  # 连续lqr离散化的时间间隔
        # 连续lqr离散化的时候， 系数矩阵相应发生变化，采用的是双线性变换
        temp = np.linalg.inv(np.eye(4) - (ts * self.A) / 2)
        self.A_bar = temp @ (np.eye(4) + (ts * self.A) / 2)
        self.B_bar = temp @ self.B * ts
        self.C_bar = temp @ self.C * ts * self.k_r * self._vehicle_Vx
        # 这里将theta_r_dao=k_r * s_dao看做常数，无飘移假设下theta_r_dao = k_r * Vx,
        # 这里做过验证，在无飘移的条件下，s_dao=Vx,微小的漂移量二者也很接近
        # print("self.C_bar.shape", self.C_bar.shape)

    def cal_error_k_fun(self, ts=0.01):
        """
        计算预测点和规划点的误差
        :param ts: 控制周期
        :param target_path: 规划路径构成的矩阵x, y是直角坐标系下位置， theta是速度方向与x轴夹角，k是路径在该点的曲率
        [(x1, y1, theta_1, k1),
         (x4, y3, theta_2, k2),
         ...]
        :param cur_state: 车辆当前位置(x, y, fi, V_x, V_y, fi')
        x,y是车辆测量得到的实际位置
        fi是航向角，车轴和x轴的夹角
        V_x, V_y是速度在车轴方向的分量和垂直车轴方向的分量
        fi'是fi的导数
        :return: 当前位置和目标位置在frenet坐标系下的横向误差e_rr，以及投影点的曲率k
        """
        x, y, fi, V_y, fi_dao = self._vehicle_state
        V_x = self._vehicle_Vx
        # 预测模块
        x = x + V_x * ts * math.cos(fi) - V_y * ts * math.sin(fi)
        y = y + V_y * ts * math.cos(fi) + V_x * ts * math.sin(fi)
        fi = fi + fi_dao * ts
        # V_x, V_y, fi_dao认为在相邻的周期内是不变的
        self.x_pre = x
        self.y_pre = y
        # 1.确定匹配点
        path_length = len(self._target_path)
        min_d = 10000

        for i in range(self.min_index, min(self.min_index + 50, path_length)):  # 当控制器是根据全局路径进行控制的时候，
        # 为了缩短匹配点需要记住上一个匹配点位置，前期没有做局部规划，所以可以这样优化
        # for i in range(0, path_length):  # 后面我们的已经做了局部规划的时候，不能再用索引来记录上一个匹配点位置，
            # 局部路径规划本来长度就短，可以不记录也行；如果想优化，我们可以通过记录绝对时间来实现，用字典类型，每个路径点对应一个绝对时间信息
            d = (self._target_path[i][0] - x) ** 2 + (self._target_path[i][1] - y) ** 2
            if d < min_d:
                min_d = d
                self.min_index = i
        min_index = self.min_index
        # print("min_index:", self.min_index)
        # 2.计算车身坐标系下的轴向向量和法向量
        tor_v = np.array([math.cos(self._target_path[min_index][2]), math.sin(self._target_path[min_index][2])])
        n_v = np.array([-math.sin(self._target_path[min_index][2]), math.cos(self._target_path[min_index][2])])

        # 3.计算匹配点指向实际位置的向量
        d_v = np.array([x - self._target_path[min_index][0], y - self._target_path[min_index][1]])

        # 4.计算e_d, e_s
        e_d = np.dot(n_v, d_v)
        e_s = np.dot(tor_v, d_v)

        # 获取投影点坐标
        self.x_pro, self.y_pro = np.array(
            [self._target_path[min_index][0], self._target_path[min_index][1]]) + e_s * tor_v

        # 5.计算theta_r
        # 曲率的定义是K = delta_theta / delta_s 然后对delta_s求极限
        # 平均曲率就是K = delta_theta / delta_s
        # delta_theta 就是切线转角，delta_s是弧长，
        # 我们要假设匹配点和投影点的航向并不相同，但是近似认为两点的曲率是相同的，这样才可以计算delta_theta
        theta_r = self._target_path[min_index][2] + \
                  self._target_path[min_index][3] * e_s  # 认为投影点和匹配点航向不同，相差一个d_theta = k_r*d_s
        # theta_r = self._target_path[min_index][2]  # delta_theta本质上就是一个很小的值，
        # apollo方案，将其近似为零，认为投影点和匹配点的航向角相同，这样是有误差的，个人更偏向于用不为零的近似

        # 6.计算e_d的导数
        e_d_dao = V_y * math.cos(fi - theta_r) + V_x * math.sin(fi - theta_r)

        # 7.计算e_fi
        # e_fi = fi - theta_r
        e_fi = math.sin(fi - theta_r)  # 这里为了防止角度的多值性，用sin(e_fi)近似e_fi， 一般情况下e_fi就是一个小量，所以可以近似

        # 8.计算S的导数
        S_dao = (V_x * math.cos(fi - theta_r) - V_y * math.sin(fi - theta_r)) / (
                    1 - self._target_path[min_index][3] * e_d)

        # 9.计算e_fi的导数
        e_fi_dao = fi_dao - self._target_path[min_index][3] * S_dao

        # 10.计算投影点的曲率，近似等于匹配点的曲率
        self.k_r = self._target_path[min_index][3]
        self.e_rr = (e_d, e_d_dao, e_fi, e_fi_dao)
        if print_flag:
            print("error-e_d-e_fi:", e_d, e_fi)

    def cal_control_para_fun(self, Q, R, F):
        """
        根据A_bac, B_bar, C_bar计算X_k
        :param Q: 是误差代价的权重对应的对角矩阵4*4,Q越大算法的性能越好，但是会牺牲算法稳定性导致最终控制量u很大
        :param F: 终端误差的权重对应的对角矩阵4*4
        :param R: 应该是控制代价的权重对应的对角矩阵1*1，这里我传入的就是一个数值， R越大越平稳，变化越小
        :return: None
        """
        # 计算M，C，Cc
        M = np.zeros(shape=((self._N+1)*self._n, self._n))
        M[0:self._n, :] = np.eye(self._n)
        for i in range(1, self._N + 1):
            M[i*self._n:(i+1)*self._n, :] = self.A_bar @ M[(i-1)*self._n:i*self._n, :]

        C = np.zeros(shape=((self._N + 1) * self._n, self._N * self._P))
        C[self._n:2*self._n, 0:self._P] = self.B_bar  # 这里self.B_bar是4*1维度的， 由于控制区间是self_P，因此这里有个按列复制的过程
        for i in range(2, self._N + 1):
            C[i * self._n:(i + 1) * self._n, (i-1) * self._P:i * self._P] = self.B_bar
            for j in range(i-2, -1, -1):
                C[i*self._n:(i+1)*self._n, j*self._P:(j+1)*self._P] = \
                    self.A_bar @ C[i*self._n:(i+1)*self._n, (j+1)*self._P:(j+2)*self._P]

        Cc = np.zeros(shape=((self._N+1)*self._n, 1))
        for i in range(1, self._N+1):
            Cc[self._n*i:self._n*(i+1), 0:1] = self.A_bar @ Cc[self._n*(i-1):self._n*i, 0:1] + self.C_bar

        # 计算Q_bar, R_bar
        Q_bar = np.zeros(shape=((self._N+1)*self._n, (self._N+1)*self._n))
        for i in range(self._N):
            Q_bar[i*self._n:(i+1)*self._n, i*self._n:(i+1)*self._n] = Q
        Q_bar[self._N*self._n:, self._N*self._n:] = F

        R_bar = np.zeros(shape=(self._N*self._P, self._N*self._P))
        for i in range(self._N):
            R_bar[i*self._P:(i+1)*self._P, i*self._P:(i+1)*self._P] = np.eye(self._P)*R

        # 计算代价函数的系数矩阵 x'.H.x + 2E'x
        # G = M.T @ Q_bar @ M
        # print("V_x", self._vehicle_Vx)
        H = C.T @ Q_bar @ C + R_bar
        E = C.T @ Q_bar @ Cc + C.T @ Q_bar @ M @ (np.array(self.e_rr).reshape(self._n, 1))
        # 解决二次规划需要转化为标准形式0.5*x'.H.x + f'x
        H = 2 * H
        f = 2 * E
        # print("H.shape", H.shape, np.linalg.matrix_rank(np.matrix(H)))
        # print("f.shape", f.shape, np.linalg.matrix_rank(np.matrix(f)))
        # 约束项
        lb = np.ones(shape=(self._N*self._P, 1))*(-1)
        ub = np.ones(shape=(self._N*self._P, 1))
        G = np.concatenate((np.identity(self._N*self._P), -np.identity(self._N*self._P)))  # （4n, 2n）
        h = np.concatenate((ub, -lb))  # (4n, 1)
        # print("G.shape", G.shape, "The rank of G", np.linalg.matrix_rank(np.matrix(G)))
        # print("h.shape", h.shape, "The rank of h", np.linalg.matrix_rank(np.matrix(h)))
        cvxopt.solvers.options['show_progress'] = False  # 程序没有问题之后不再输出中间过程
        # 计算时要将输入转化为cvxopt.matrix
        # 该方法返回值是一个字典类型，包含了很多的参数，其中x关键字对应的是优化后的解
        res = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f),
                                G=cvxopt.matrix(G), h=cvxopt.matrix(h))
        return res['x'][0]

    def _control(self):
        """
        MPC 计算最终控制量
        :param K: LQR输出
        :param e_rr: 误差输出
        :param delta_f: 前馈输出
        :return: u最终控制量
        """
        b = 1
        Q = np.eye(4)
        Q[0][0] = 250
        Q[1][1] = 1
        Q[2][2] = 50
        Q[3][3] = 1
        F = np.eye(4)
        R = b
        self.cal_vehicle_info()
        self.cal_A_B_C_fun()
        self.cal_error_k_fun(ts=0.1)
        self.cal_coefficient_of_discretion_fun()

        current_steering = self.cal_control_para_fun(Q, R, F)
        # print("raw steering:", current_steering)
        return current_steering


"""
LQR 控制算法进行横向控制
总共分为5个模块
1 A,B计算模块
“
根据整车参数(a, b, Cf, Cr, m, Iz)和V_x,通过公式计算A,B; 具体公式间第八讲总结
a，b是前后轮中心距离车质心的距离
CF, Cr是前后轮的侧偏刚度
m是车的质量
Iz是车的转动惯量
V_x是车辆速度在车轴方向的分量
”
2 LQR 模块
“
根据Q,R和A,B计算K, 通过迭代黎卡提方程求解， P = Q + A^PA - A^PB(R+B^PB)'B^PA.其中A^是求转置，A'是求逆
代价函数：J = w1*X*X + w2*u*u,
当X和u都是列向量时，J= X^Qx + u^Ru，就是对所有误差进行求和
Q,R是误差代价和控制代价的权重对应的对角矩阵
”
3 e_rr, k计算模块
“
根据当前测量位置车辆的（x, y, fi, V_y, fi'）和规划位置的（x_r, y_r, theta_r, k_r）计算误差e_rr, 和投影点的曲率
由于车辆不具有路况的观察能力，且控制具有一定的滞后性，因此实际中需要加上一个预测模块，预测当前的车辆状态在到达下一个控制周期之前的可能位置
”
4 前馈控制模块计算delta_f
“
通过V_x, K, k和前馈控制公式计算前馈控制量delta_f
”
5 最终控制模块计算控制量u
“
通过u=-K*e_rr + delta_f计算最终控制量
”
"""


class Lateral_LQR_controller(object):
    def __init__(self, ego_vehicle, vehicle_para, pathway_xy_theta_kappa):
        self._vehicle_para = vehicle_para
        self._vehicle = ego_vehicle  # type: carla.Vehicle
        self._vehicle_state = None
        self._vehicle_Vx = 0
        self._target_path = pathway_xy_theta_kappa
        self.A = np.zeros(shape=(4, 4), dtype="float64")
        self.B = np.zeros(shape=(4, 1), dtype="float64")
        self.K = None
        self.k_r = None
        self.e_rr = None
        self.delta_f = None
        self.min_index = 0
        # debug 变量， 用于显示预测点和投影点的位置
        self.x_pre = 0
        self.y_pre = 0
        self.x_pro = 0
        self.y_pro = 0

    def xy_list_2_target_path(self, pathway_xy_list):
        """
        将由坐标构成的路径转化为（x, y, theta, k）的形式
        :param pathway_xy_list: 【x0, y0, (x1, y1), ...】
        :return: [(x0, y0, theta0, k0), ...]
        """
        theta_list, kappa_list = cal_heading_kappa(pathway_xy_list)
        # self._target_path = smooth_reference_line(xy_list_ori)  # 对生成的原始轨迹进行平滑,这里只是做了个实验
        for i in range(len(theta_list)):
            self._target_path.append((pathway_xy_list[i][0], pathway_xy_list[i][1], theta_list[i], kappa_list[i]))

    def cal_vehicle_info(self):
        """
        通过调用方法获取车辆的状态信息
        :return: None
        """
        vehicle_loc = self._vehicle.get_location()
        x, y = vehicle_loc.x, vehicle_loc.y
        fi = self._vehicle.get_transform().rotation.yaw*(math.pi/180)  # 车身横摆角，车轴和x轴的夹角
        V = self._vehicle.get_velocity()  # 航向角是车速与x轴夹角
        V_length = math.sqrt(V.x*V.x + V.y*V.y + V.z*V.z)
        beta = math.atan2(V.y, V.x) - fi  # 质心侧偏角，车速和车轴之间的夹角
        # print("beta", beta, "fi", fi)
        Vy = V_length*math.sin(beta)  # 车速在车身坐标系下的分量
        Vx = V_length*math.cos(beta)
        # print("Vx", Vx, "Vy", Vy)
        fi_dao = self._vehicle.get_angular_velocity().z*(math.pi/180)
        self._vehicle_state = (x, y, fi, Vy, fi_dao)
        self._vehicle_Vx = Vx

    def cal_A_B_fun(self):
        """
        根据整车参数vehicle_para和V_x,通过公式计算A,B; 具体公式间第八讲总结
              vehicle_para: vehicle_para = (a, b, Cf, Cr, m, Iz)
                            a，b是前后轮中心距离车质心的距离
                            CF, Cr是前后轮的侧偏刚度
                            m是车的质量
                            Iz是车的转动惯量

              V_x: V_x是车辆速度在车轴方向的分量
        :return: 矩阵A和B, np.array type
        A的维度4*4
        B的维度4*1
        """
        V_x = self._vehicle_Vx
        V_x = V_x + 0.0001  # 因为速度有可能为零，加个小量，避免后面的除法运算报错
        (a, b, Cf, Cr, m, Iz) = self._vehicle_para

        self.A[0][1] = 1

        self.A[1][1] = (Cf + Cr) / (m * V_x)
        self.A[1][2] = -(Cf + Cr) / m
        self.A[1][3] = (a * Cf - b * Cr) / (m * V_x)

        self.A[2][3] = 1

        self.A[3][1] = (a * Cf - b * Cr) / (Iz * V_x)
        self.A[3][2] = -(a * Cf - b * Cr) / Iz
        self.A[3][3] = (a * a * Cf + b * b * Cr) / (Iz * V_x)

        self.B[1][0] = -Cf / m
        self.B[3][0] = -a * Cf / Iz

    def LQR_fun(self, Q, R):
        """
        根据Q,R和A,B计算K, 通过迭代黎卡提方程求解， P = Q + A^PA - A^PB(R+B^PB)'B^PA.其中A^是求转置，A'是求逆
        :param Q: 是误差代价的权重对应的对角矩阵4*4,Q越大算法的性能越好，但是会牺牲算法稳定性导致最终控制量u很大
        :param R: 控制代价的权重对应的对角矩阵4*4， R越大越平稳，变化越小
        :param A: cal_A_B_fun模块的输出4*4
        :param B:
        :return: K, np.array类型
        """
        P = Q
        P_pre = Q
        max_itr = 5000
        eps = 0.1
        ts = 0.1  # 连续lqr离散化的时间间隔
        # 连续lqr离散化的时候， 系数矩阵相应发生变化，采用的是双线性变换
        temp = np.linalg.inv(np.eye(4) - (ts*self.A)/2)
        A = temp @ (np.eye(4) + (ts*self.A)/2)
        B = temp @ self.B * ts
        i = 0
        AT = A.T  # 4*4
        BT = B.T  # 1*4
        for i in range(max_itr):
            P = AT @ P @ A - (AT @ P @ B) @ np.linalg.inv(R + BT @ P @ B) @ (BT @ P @ A) + Q  # 要不断迭代
            if abs(P - P_pre).max() < eps:
                break
            P_pre = P
        if print_flag:
            print("黎卡提方程爹迭代次数：", i)  # 输出迭代的次数

        self.K = np.linalg.inv(BT @ P @ B + R) @ (BT @ P @ A)

    def cal_error_k_fun(self, ts=0.01):
        """
        计算预测点和规划点的误差
        :param ts: 控制周期
        :param target_path: 规划路径构成的矩阵x, y是直角坐标系下位置， theta是速度方向与x轴夹角，k是路径在该点的曲率
        [(x1, y1, theta_1, k1),
         (x4, y3, theta_2, k2),
         ...]
        :param cur_state: 车辆当前位置(x, y, fi, V_x, V_y, fi')
        x,y是车辆测量得到的实际位置
        fi是航向角，车轴和x轴的夹角
        V_x, V_y是速度在车轴方向的分量和垂直车轴方向的分量
        fi'是fi的导数
        :return: 当前位置和目标位置在frenet坐标系下的横向误差e_rr，以及投影点的曲率k
        """
        x, y, fi, V_y, fi_dao = self._vehicle_state
        V_x = self._vehicle_Vx
        # 预测模块
        x = x + V_x * ts * math.cos(fi) - V_y * ts * math.sin(fi)
        y = y + V_y * ts * math.cos(fi) + V_x * ts * math.sin(fi)
        fi = fi + fi_dao * ts
        # V_x, V_y, fi_dao认为在相邻的周期内是不变的
        self.x_pre = x
        self.y_pre = y
        # 1.确定匹配点
        path_length = len(self._target_path)
        min_d = 10000

        # for i in range(self.min_index, min(self.min_index + 50, path_length)):  # 当控制器是根据全局路径进行控制的时候，
        # 为了缩短匹配点需要记住上一个匹配点位置，前期没有做局部规划，所以可以这样优化
        for i in range(0, path_length):  # 后面我们的已经做了局部规划的时候，不能再用索引来记录上一个匹配点位置，
            # 局部路径规划本来长度就短，可以不记录也行；如果想优化，我们可以通过记录绝对时间来实现，用字典类型，每个路径点对应一个绝对时间信息
            d = (self._target_path[i][0] - x) ** 2 + (self._target_path[i][1] - y) ** 2
            if d < min_d:
                min_d = d
                self.min_index = i
        min_index = self.min_index
        # print("min_index:", self.min_index)
        # 2.计算车身坐标系下的轴向向量和法向量
        tor_v = np.array([math.cos(self._target_path[min_index][2]), math.sin(self._target_path[min_index][2])])
        n_v = np.array([-math.sin(self._target_path[min_index][2]), math.cos(self._target_path[min_index][2])])

        # 3.计算匹配点指向实际位置的向量
        d_v = np.array([x - self._target_path[min_index][0], y - self._target_path[min_index][1]])

        # 4.计算e_d, e_s
        e_d = np.dot(n_v, d_v)
        e_s = np.dot(tor_v, d_v)

        # 获取投影点坐标
        self.x_pro, self.y_pro = np.array([self._target_path[min_index][0],
                                           self._target_path[min_index][1]]) + e_s * tor_v

        # 5.计算theta_r
        # 曲率的定义是K = delta_theta / delta_s 然后对delta_s求极限
        # 平均曲率就是K = delta_theta / delta_s
        # delta_theta 就是切线转角，delta_s是弧长，
        # 我们要假设匹配点和投影点的航向并不相同，但是近似认为两点的曲率是相同的，这样才可以计算delta_theta
        theta_r = self._target_path[min_index][2] + self._target_path[min_index][3] * e_s  # 认为投影点和匹配点航向不同，相差一个d_theta = k_r*d_s
        # theta_r = self._target_path[min_index][2]  # delta_theta本质上就是一个很小的值，
        # apollo方案，将其近似为零，认为投影点和匹配点的航向角相同，这样是有误差的，个人更偏向于用不为零的近似

        # 6.计算e_d的导数
        e_d_dao = V_y * math.cos(fi - theta_r) + V_x * math.sin(fi - theta_r)

        # 7.计算e_fi
        # e_fi = fi - theta_r
        e_fi = math.sin(fi - theta_r)  # 这里为了防止角度的多值性，用sin(e_fi)近似e_fi， 一般情况下e_fi就是一个小量，所以可以近似

        # 8.计算S的导数
        S_dao = (V_x * math.cos(fi - theta_r) - V_y * math.sin(fi - theta_r)) / (1 - self._target_path[min_index][3] * e_d)

        # 9.计算e_fi的导数
        e_fi_dao = fi_dao - self._target_path[min_index][3] * S_dao

        # 10.计算投影点的曲率，近似等于匹配点的曲率
        self.k_r = self._target_path[min_index][3]
        self.e_rr = (e_d, e_d_dao, e_fi, e_fi_dao)
        if print_flag:
            print("error-e_d-e_fi:", e_d, e_fi)

    def forward_control_fun(self):
        """
        计算前馈控制量delta_f
        :param vehicle_para: vehicle_para = (a, b, Cf, Cr, m, Iz)
        :param K: LQR的输出结果
        :param k_r: 投影点曲率
        :param V_x: 速度在车轴方向的分量
        :return: 前馈空盒子量delta_f
        """
        a, b, Cf, Cr, m, Iz = self._vehicle_para
        # print(self.K.shape)
        K_3 = self.K[0][2]
        V_x = self._vehicle_Vx
        self.delta_f = self.k_r * (a + b - b * K_3 - (b / Cf + a * K_3 / Cr - a / Cr) * (m * V_x * V_x) / (a + b))
        self.delta_f = self.delta_f*np.pi/180  # 由于之前输入的误差形式弧度，所以这里也要将前馈量转化为弧度形式

    def _control(self):
        """
        LQR 计算最终控制量
        :param K: LQR输出
        :param e_rr: 误差输出
        :param delta_f: 前馈输出
        :return: u最终控制量
        """
        b = 1
        Q = np.eye(4)
        Q[0][0] = 200
        Q[1][1] = 1
        Q[2][2] = 50
        Q[3][3] = 1
        R = b
        self.cal_vehicle_info()
        self.cal_A_B_fun()
        self.LQR_fun(Q=Q, R=R)
        self.cal_error_k_fun(ts=0.1)
        self.forward_control_fun()
        # print("**********", self.K)
        # print("**********", self.delta_f)
        # print("**********", self.k_r)
        current_steering = -np.dot(self.K, np.array(self.e_rr)) + self.delta_f
        current_steering = current_steering[0]
        # print("raw steering:", current_steering)
        return current_steering


class Longitudinal_PID_controller(object):
    """
    PID 控制
    包括比例项， 积分项，微分项
    只有比例项会产生稳态误差，（稳态误差就是控制最终稳定在一个值但是和目标值有一定的差距）
    引入积分项可以消除稳态误差，但是会引起超调、震荡问题和积分饱和问题
    采用积分分离来克服系统超调和震荡
    """
    def __init__(self, ego_vehicle,  K_P=1.15, K_I=0, K_D=0, dt=0.01):
        """
        采用PID进行纵向控制
        :param ego_vehicle: 控制的车辆， 类型是carla.Vehicle
        :param K_P: 比例项系数
        :param K_I: 积分项系数
        :param K_D: 微分项系数
        :param dt: 控制间隔
        """
        self._vehicle = ego_vehicle  # type: carla.Vehicle
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.dt = dt
        self.target_speed = None
        self.error_buffer = deque(maxlen=60)  # 设置一个误差缓存区，用于积分项和差分项的计算
        self.error_threshold = 1  # 设定一个阈值，进行积分分离，标量单位是km/h,
        # 由于carla的最大throttle是1，因此误差如果大于1就让采取积分分离

    def PID_fun(self):
        """

        :return:
        """
        cur_velocity = self._vehicle.get_velocity()  # 调用carla API 获得的速度是个三维矢量 单位是m/s
        cur_speed = 3.6*math.sqrt(cur_velocity.x*cur_velocity.x
                                  + cur_velocity.y*cur_velocity.y
                                  + cur_velocity.z*cur_velocity.z)  # 转化为标量，单位是km/h

        error = self.target_speed - cur_speed  # 当前误差
        self.error_buffer.append(error)  # 将新的误差放入缓存区，如果缓存区满了，最左边的溢出，整体数据左移一位，新的数据加在最右边

        if len(self.error_buffer) >= 2:
            # 积分误差，为了解决稳态误差引入的积分项
            integral_error = sum(self.error_buffer) * self.dt
            # 微分误差，为了缓解超调
            differential_error = (self.error_buffer[-1] - self.error_buffer[-2]) / self.dt
        else:
            integral_error = 0.0
            differential_error = 0.0

        # 积分分离，当误差较大时，采取积分分离防止超调
        if print_flag:
            print("absolute speed error:", abs(error))

        if abs(error) > self.error_threshold:
            # 一旦出现误差大于阈值的情况，积分分离让积分项为0，清除误差缓存区，此时只有比例项发挥作用
            integral_error = 0.0
            self.error_buffer.clear()

        return self.K_P * error + self.K_I * integral_error + self.K_D * differential_error

    def PID_control(self, target_speed):

        self.target_speed = target_speed
        return self.PID_fun()


class Vehicle_control(object):
    def __init__(self, ego_vehicle, vehicle_para, pathway, controller_type="MPC_controller"):
        self._vehicle = ego_vehicle
        self._max_throttle = 1
        self._max_brake = 1
        self._max_steer = 1
        self.min_steer = -1
        if controller_type == "MPC_controller":
            self.Lat_control = Lateral_MPC_controller(ego_vehicle, vehicle_para, pathway)
        elif controller_type == "LQR_controller":
            self.Lat_control = Lateral_LQR_controller(ego_vehicle, vehicle_para, pathway)

        self.Lon_control = Longitudinal_PID_controller(ego_vehicle)  # 这里不允许后续程序修改PID的参数，使用设定好的默认值

    def run_step(self, target_speed):

        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False
        control.gear = 1
        current_steering = self.Lat_control._control()
        # 获取横向和纵向控制量
        current_acceleration = self.Lon_control.PID_control(target_speed)

        # 横向控制整定
        if current_steering >= 0:
            steering = min(self._max_steer, current_steering)
        else:
            steering = max(self.min_steer, current_steering)
        # steering = current_steering
        control.steer = steering

        # 纵向控制整定
        if current_acceleration >= 0:
            control.throttle = min(self._max_throttle, current_acceleration)
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = max(self._max_brake, current_acceleration)  # 没有反向加速，加速度为零时对应的是刹车制动

        V = self._vehicle.get_velocity()
        V_len = math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)
        if print_flag:
            print("current speed:", V_len, "m/s", "current steer control:", steering)
        return control


class Lateral_MPC__with_feedforward_controller(object):
    def __init__(self, ego_vehicle, vehicle_para, pathway_xy_theta_kappa):
        self._vehicle_state = None
        self._vehicle_para = vehicle_para
        self._vehicle = ego_vehicle  # type: carla.Vehicle
        self._vehicle_Vx = 0
        self._target_path = pathway_xy_theta_kappa
        self._N = 4  # 预测区间
        self._P = 2  # 控制区间
        self._n = 4  # 状态变量长度
        self.A = np.zeros(shape=(4, 4), dtype="float64")
        self.B = np.zeros(shape=(4, 1), dtype="float64")
        self.C = np.zeros(shape=(4, 1), dtype="float64")
        self.A_bar = None  # 离散化的A,B,C矩阵
        self.B_bar = None
        self.C_bar = None
        self.K = None  # 反馈增益
        self.k_r = None  # 曲率
        self.e_rr = None
        self.delta_f = None  # 前馈
        self.min_index = 0
        # debug 变量， 用于显示预测点和投影点的位置
        self.x_pre = 0
        self.y_pre = 0
        self.x_pro = 0
        self.y_pro = 0

        # 初始化
        self.cal_vehicle_info()
        self.cal_A_B_C_fun()

    def cal_vehicle_info(self):
        """
        acquire the states of ego-vehicle according to recall the internal methods
        通过调用方法获取车辆的状态信息
        :return: None
        """
        vehicle_loc = self._vehicle.get_location()
        x, y = vehicle_loc.x, vehicle_loc.y
        fi = self._vehicle.get_transform().rotation.yaw * (math.pi / 180)  # 车身横摆角，车轴和x轴的夹角
        V = self._vehicle.get_velocity()  # 航向角是车速与x轴夹角
        V_length = math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)
        beta = math.atan2(V.y, V.x) - fi  # 质心侧偏角，车速和车轴之间的夹角
        # print("beta", beta, "fi", fi)
        Vy = V_length * math.sin(beta)  # 车速在车身坐标系下的分量
        Vx = V_length * math.cos(beta)
        # print("Vx", Vx, "Vy", Vy)
        fi_dao = self._vehicle.get_angular_velocity().z * (math.pi / 180)
        self._vehicle_state = (x, y, fi, Vy, fi_dao)
        self._vehicle_Vx = Vx

    def cal_A_B_C_fun(self):
        """
        calculate the coefficient matrix
        根据整车参数vehicle_para和V_x,通过公式计算A,B; 具体公式间第八讲总结
              vehicle_para: vehicle_para = (a, b, Cf, Cr, m, Iz)
                            a，b是前后轮中心距离车质心的距离
                            CF, Cr是前后轮的侧偏刚度
                            m是车的质量
                            Iz是车的转动惯量

              V_x: V_x是车辆速度在车轴方向的分量
        """
        V_x = self._vehicle_Vx
        V_x = V_x + 0.0001  # 因为速度有可能为零，加个小量，避免后面的除法运算报错
        (a, b, Cf, Cr, m, Iz) = self._vehicle_para
        self.A[0][1] = 1

        self.A[1][1] = (Cf + Cr) / (m * V_x)
        self.A[1][2] = -(Cf + Cr) / m
        self.A[1][3] = (a * Cf - b * Cr) / (m * V_x)

        self.A[2][3] = 1

        self.A[3][1] = (a * Cf - b * Cr) / (Iz * V_x)
        self.A[3][2] = -(a * Cf - b * Cr) / Iz
        self.A[3][3] = (a * a * Cf + b * b * Cr) / (Iz * V_x)

        self.B[1][0] = -Cf / m
        self.B[3][0] = -a * Cf / Iz

        self.C[1][0] = (a*Cf + b*Cr)/(m*V_x) - V_x
        self.C[3][0] = (a**2*Cf + b**2*Cr)/(Iz*V_x)

    def cal_discretized_matrix(self):
        """
        calculate the discrete form of matrix A, B, C after the state equation is discretized.
        计算矩阵A、B、C的离散形式
        """
        ts = 0.1  # 连续lqr离散化的时间间隔
        # 连续lqr离散化的时候， 系数矩阵相应发生变化，采用的是双线性变换
        temp = np.linalg.inv(np.eye(4) - (ts * self.A) / 2)
        self.A_bar = temp @ (np.eye(4) + (ts * self.A) / 2)
        self.B_bar = temp @ self.B * ts
        self.C_bar = temp @ self.C * ts * self.k_r * self._vehicle_Vx  # 这里将theta_r_dao看做常数，无飘移假设下theta_r_dao = k_r * Vx
        # print("self.C_bar.shape", self.C_bar)

    def cal_error_k_fun(self, ts=0.01):
        """
        计算预测点和规划点的误差
        :param ts: 控制周期
        :param target_path: 规划路径构成的矩阵x, y是直角坐标系下位置， theta是速度方向与x轴夹角，k是路径在该点的曲率
        [(x1, y1, theta_1, k1),
         (x4, y3, theta_2, k2),
         ...]
        :param cur_state: 车辆当前位置(x, y, fi, V_x, V_y, fi')
        x,y是车辆测量得到的实际位置
        fi是航向角，车轴和x轴的夹角
        V_x, V_y是速度在车轴方向的分量和垂直车轴方向的分量
        fi'是fi的导数
        :return: 当前位置和目标位置在frenet坐标系下的横向误差e_rr，以及投影点的曲率k
        """
        x, y, fi, V_y, fi_dao = self._vehicle_state
        V_x = self._vehicle_Vx
        # 预测模块
        x = x + V_x * ts * math.cos(fi) - V_y * ts * math.sin(fi)
        y = y + V_y * ts * math.cos(fi) + V_x * ts * math.sin(fi)
        fi = fi + fi_dao * ts
        # V_x, V_y, fi_dao认为在相邻的周期内是不变的
        self.x_pre = x
        self.y_pre = y
        # 1.确定匹配点
        path_length = len(self._target_path)
        min_d = 10000

        # for i in range(self.min_index, min(self.min_index + 50, path_length)):  # 当控制器是根据全局路径进行控制的时候，
        # 为了缩短匹配点需要记住上一个匹配点位置，前期没有做局部规划，所以可以这样优化
        for i in range(0, path_length):  # 后面我们的已经做了局部规划的时候，不能再用索引来记录上一个匹配点位置，
            # 局部路径规划本来长度就短，可以不记录也行；如果想优化，我们可以通过记录绝对时间来实现，用字典类型，每个路径点对应一个绝对时间信息
            d = (self._target_path[i][0] - x) ** 2 + (self._target_path[i][1] - y) ** 2
            if d < min_d:
                min_d = d
                self.min_index = i
        min_index = self.min_index
        # print("min_index:", self.min_index)
        # 2.计算车身坐标系下的轴向向量和法向量
        tor_v = np.array([math.cos(self._target_path[min_index][2]), math.sin(self._target_path[min_index][2])])
        n_v = np.array([-math.sin(self._target_path[min_index][2]), math.cos(self._target_path[min_index][2])])

        # 3.计算匹配点指向实际位置的向量
        d_v = np.array([x - self._target_path[min_index][0], y - self._target_path[min_index][1]])

        # 4.计算e_d, e_s
        e_d = np.dot(n_v, d_v)
        e_s = np.dot(tor_v, d_v)

        # 获取投影点坐标
        self.x_pro, self.y_pro = np.array(
            [self._target_path[min_index][0], self._target_path[min_index][1]]) + e_s * tor_v

        # 5.计算theta_r
        # 曲率的定义是K = delta_theta / delta_s 然后对delta_s求极限
        # 平均曲率就是K = delta_theta / delta_s
        # delta_theta 就是切线转角，delta_s是弧长，
        # 我们要假设匹配点和投影点的航向并不相同，但是近似认为两点的曲率是相同的，这样才可以计算delta_theta
        theta_r = self._target_path[min_index][2] + self._target_path[min_index][
            3] * e_s  # 认为投影点和匹配点航向不同，相差一个d_theta = k_r*d_s
        # theta_r = self._target_path[min_index][2]  # delta_theta本质上就是一个很小的值，
        # apollo方案，将其近似为零，认为投影点和匹配点的航向角相同，这样是有误差的，个人更偏向于用不为零的近似

        # 6.计算e_d的导数
        e_d_dao = V_y * math.cos(fi - theta_r) + V_x * math.sin(fi - theta_r)

        # 7.计算e_fi
        e_fi = fi - theta_r
        # e_fi = math.sin(fi - theta_r)  # 这里为了防止角度的多值性，用sin(e_fi)近似e_fi， 一般情况下e_fi就是一个小量，所以可以近似

        # 8.计算S的导数
        S_dao = (V_x * math.cos(fi - theta_r) - V_y * math.sin(fi - theta_r)) / (
                    1 - self._target_path[min_index][3] * e_d)

        # 9.计算e_fi的导数
        e_fi_dao = fi_dao - self._target_path[min_index][3] * S_dao

        # 10.计算投影点的曲率，近似等于匹配点的曲率
        self.k_r = self._target_path[min_index][3]
        self.e_rr = (e_d, e_d_dao, e_fi, e_fi_dao)
        if print_flag:
            print("error-e_d-e_fi:", e_d, e_fi)

    def cal_control_para_fun(self, Q, R, F):
        """
        calculate the control variable(or signal)
        根据A_bac, B_bar, C_bar计算X_k
        :param Q: 是误差代价的权重对应的对角矩阵4*4,Q越大算法的性能越好，但是会牺牲算法稳定性导致最终控制量u很大
        :param F: 终端误差的权重对应的对角矩阵4*4
        :param R: 应该是控制代价的权重对应的对角矩阵1*1，这里我传入的就是一个数值， R越大越平稳，变化越小
        :return: None
        """
        # 计算M，C，Cc
        M = np.zeros(shape=((self._N+1)*self._n, self._n))
        C = np.zeros(shape=((self._N+1)*self._n, self._N*self._P))
        M[0:self._n, :] = np.eye(self._n)
        for i in range(1, self._N + 1):
            M[i*self._n:(i+1)*self._n, :] = self.A_bar @ M[(i-1)*self._n:i*self._n, :]

        C[self._n:2*self._n, 0:self._P] = self.B_bar  # 这里self.B_bar是4*1维度的， 由于控制区间是self_P，因此这里有个按复制的过程
        for i in range(2, self._N + 1):
            C[i * self._n:(i + 1) * self._n, (i-1) * self._P:i * self._P] = self.B_bar
            for j in range(i-2, -1, -1):
                C[i*self._n:(i+1)*self._n, j*self._P:(j+1)*self._P] = \
                    self.A_bar @ C[i*self._n:(i+1)*self._n, (j+1)*self._P:(j+2)*self._P]
        Cc = np.zeros(shape=((self._N+1)*self._n, 1))
        for i in range(1, self._N+1):
            Cc[self._n*i:self._n*(i+1), 0:1] = self.A_bar @ Cc[self._n*(i-1):self._n*i, 0:1] + self.C_bar

        # 计算Q_bar, R_bar
        Q_bar = np.zeros(shape=((self._N+1)*self._n, (self._N+1)*self._n))
        for i in range(self._N):
            Q_bar[i*self._n:(i+1)*self._n, i*self._n:(i+1)*self._n] = Q
        Q_bar[self._N*self._n:, self._N*self._n:] = F
        R_bar = np.zeros(shape=(self._N*self._P, self._N*self._P))
        for i in range(self._P):
            R_bar[i*self._P:(i+1)*self._P, i*self._P:(i+1)*self._P] = np.eye(self._P)*R

        # 计算代价函数的系数矩阵
        # G = M.T @ Q_bar @ M
        E = M.T @ Q_bar @ C
        H = C.T @ Q_bar @ C + R_bar
        # 解决二次规划需要转化为标准形式0.5*x'.H.x + f'x
        H = 2 * H
        f = 2 * E.T @ (np.array(self.e_rr).reshape(self._n, 1)) + 2 * C.T @ Q_bar.T @ Cc
        # print("H.shape", H.shape)
        # print("f.shape", f.shape)
        # 约束项
        lb = np.ones(shape=(self._N*self._P, 1))*(-1)
        ub = np.ones(shape=(self._N*self._P, 1))
        G = np.concatenate((np.identity(self._N*self._P), -np.identity(self._N*self._P)))  # （4n, 2n）
        h = np.concatenate((ub, -lb))  # (4n, 1)
        # print("G.shape", G.shape)
        # print("h.shape", h.shape)
        cvxopt.solvers.options['show_progress'] = False  # 程序没有问题之后不再输出中间过程
        # 计算时要将输入转化为cvxopt.matrix
        # 该方法返回值是一个字典类型，包含了很多的参数，其中x关键字对应的是优化后的解
        res = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f), G=cvxopt.matrix(G), h=cvxopt.matrix(h))
        return res['x'][0]

    def MPC_control(self):
        """
        计算最终控制量
        :param K: LQR输出
        :param e_rr: 误差输出
        :param delta_f: 前馈输出
        :return: u最终控制量
        """
        b = 1
        Q = np.eye(4)
        Q[0][0] = 200
        Q[1][1] = 1
        Q[2][2] = 1
        Q[3][3] = 1
        F = 10 * np.eye(4)
        R = b
        self.cal_vehicle_info()
        self.cal_A_B_C_fun()
        self.cal_error_k_fun(ts=0.1)
        self.cal_discretized_matrix()
        # print("**********", self.K)
        # print("**********", self.delta_f)
        # print("**********", self.k_r)
        current_steering = self.cal_control_para_fun(Q, R, F)
        # current_steering = current_steering[0]
        # print("raw steering:", current_steering)
        return current_steering