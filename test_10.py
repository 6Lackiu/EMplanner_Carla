#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : test_10.py

#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : test_9.py

"""
实现动态障碍物的感知，进行路径规划和速度规划
1.获取自车匹配点的索引
2.根据匹配点的索引在全局路径上采样81个点
3.对采样点进行平滑
4.根据采样得到的局部路径为参考线，将根据对动态障碍物的预测，判断自车会在什么时间，什么地方遭遇障碍物
5.形成虚拟障碍物，采用路径规划避障（S-L + DP + QP）
6.构建S-T图，为新的路径规划速度信息
7.控制器根据规划得到的新的路径和速度信息引导自车躲避动态障碍物
8.后面可能还需要对规划的轨迹进行拼接和平滑
"""

import carla

import multiprocessing
import time
import math
import numpy as np

from planner import planning_utils, path_planning, speed_planning_test
from controller.controller import Vehicle_control
from planner.global_planning import global_path_planner
from sensors.Sensors_detector_lib import Obstacle_detector
from agents.navigation.behavior_agent import BehaviorAgent


# def get_traffic_light_state(current_traffic_light):  # get_state()方法只能返回红灯和黄灯状态，没有绿灯状态（默认把绿灯认为正常）
#     # 此方法把红绿蓝三种状态都标定出来
#     if current_traffic_light is None:
#         current_traffic_light_state = "Green"
#     else:
#         current_traffic_light_state = current_traffic_light.get_state()
#     return current_traffic_light_state


def emergence_brake():
    brake_control = carla.VehicleControl()
    brake_control.steer = 0  # 转向控制
    brake_control.throttle = 0  # 油门控制
    brake_control.brake = 1  # 刹车控制
    return brake_control


def get_actor_from_world(ego_vehicle: carla.Vehicle, dis_limitation=100):
    """已验证
    获取当前车辆前方潜在的车辆障碍物
    首先获取在主车辆一定范围内的其他车辆，再通过速度矢量和位置矢量将在主车辆运动方向后方的车辆过滤掉
    param:  ego_vehicle: 主车辆
            dis_limitation: 探测范围
    return: v_list:(vehicle, dist)
    """
    carla_world = ego_vehicle.get_world()  # type:carla.World
    vehicle_loc = ego_vehicle.get_location()
    static_vehicle_list = []  # 储存范围内的静态车辆
    dynamic_vehicle_list = []  # 储存范围内的动态车辆
    vehicle_list = carla_world.get_actors().filter("vehicle.*")
    for vehicle in vehicle_list:
        dis = math.sqrt((vehicle_loc.x - vehicle.get_location().x) ** 2 +
                        (vehicle_loc.y - vehicle.get_location().y) ** 2 +
                        (vehicle_loc.z - vehicle.get_location().z) ** 2)
        if dis < dis_limitation and ego_vehicle.id != vehicle.id:
            v1 = np.array([vehicle.get_location().x - vehicle_loc.x,
                           vehicle.get_location().y - vehicle_loc.y,
                           vehicle.get_location().z - vehicle_loc.z])  # 其他车辆到ego_vehicle的矢量
            ego_vehicle_velocity = np.array([ego_vehicle.get_velocity().x, ego_vehicle.get_velocity().y,
                                             ego_vehicle.get_velocity().z])  # ego_vehicle的速度矢量
            """# 如果车辆出现在ego_vehicle的运动前方，则有可能是障碍物
            # 还需要控制可能的障碍物距离参考线的横向距离, 我的想法是将障碍物在参考线上投影，计算投影点和车辆的距离，
            # 如果距离大于阈值则认为不影响ego-vehicle的运动，反之认为是障碍物会影响ego-vehicle的运动
            # 现在简化一下，将横向距离暂时设定为ego-vehicle当前航向方向的垂直距离，即ego完全按照参考线行驶"""
            ego_vehicle_theta = ego_vehicle.get_transform().rotation.yaw * (math.pi / 180)
            n_r = np.array([-math.sin(ego_vehicle_theta), math.cos(ego_vehicle_theta), 0])
            if -5 < np.dot(v1, n_r) < 5:  # 障碍物横向距离
                if np.dot(v1, ego_vehicle_velocity) > -10:  # 在ego车后10m及车前的视为障碍物
                    vehicle_speed = math.sqrt(vehicle.get_velocity().x ** 2 +
                                              vehicle.get_velocity().y ** 2 + vehicle.get_velocity().z ** 2)
                    if vehicle_speed > 1:
                        dynamic_vehicle_list.append((vehicle, dis, vehicle_speed))
                    else:
                        static_vehicle_list.append((vehicle, dis))
                # elif np.dot(v1, ego_vehicle_velocity) < 0 and dis < 10:  # 自车后面十米以内的障碍物仍然考虑，超过十米就不再考虑
                #     static_vehicle_list.append((vehicle, -dis))
    static_vehicle_list.sort(key=lambda tup: tup[1])  # 按距离排序
    dynamic_vehicle_list.sort(key=lambda tup: tup[1])  # 按距离排序
    return static_vehicle_list, dynamic_vehicle_list


def motion_planning(conn):
    while 1:
        # 接收主进程发送的用于局部路径规划的数据，如果没有收到数据子进程会阻塞
        # 主进程中发送的数据：conn1.send((static_obs_info, dynamic_obs_info, (vehicle_loc.x, vehicle_loc.y), (pred_x, pred_y),
        #                         (vehicle_v.x, vehicle_v.y), (vehicle_a.x, vehicle_a.y),
        #                         global_frenet_path, match_point_list, time.time()))
        possible_static_obs_, possible_dynamic_obs_, vehicle_loc_, pred_loc_, vehicle_v_, vehicle_a_,\
            global_frenet_path_, match_point_list_, cur_time = conn.recv()
        start_time = time.time()
        # 1.确定预测点在全局路径上的投影点索引
        match_point_list_, _ = planning_utils.find_match_points(xy_list=[pred_loc_],
                                                                frenet_path_node_list=global_frenet_path_,
                                                                is_first_run=False,
                                                                pre_match_index=match_point_list_[0])
        # 2.根据匹配点的索引在全局路径上采样一定数量的点
        local_frenet_path_ = planning_utils.sampling(match_point_list_[0], global_frenet_path_,
                                                     back_length=10, forward_length=50)
        # 由于我们的道路采样精度最少是2（1的情况不考虑，太小的采样精度在实际中不现实），所以确定参考线的时候向后取50个点可以保证最少以百米的未来参考
        # 后面进行动态规划的时候我们搜索的范围就是一百米，所以要保证动态规划的过程中参考线是存在的

        # 3.对采样点进行平滑，作为后续规划的参考线
        local_frenet_path_opt_ = planning_utils.smooth_reference_line(local_frenet_path_)

        # 计算以车辆当前位置为原点的s_map
        s_map = planning_utils.cal_s_map_fun(local_frenet_path_opt_, origin_xy=vehicle_loc_)
        # path_s, path_l = planning_utils.cal_s_l_fun(local_frenet_path_opt_, local_frenet_path_opt_, s_map)
        # 提取障碍物的位置信息
        if len(possible_static_obs_) != 0 and possible_static_obs_[0][-1] <= 30:
            static_obs_xy = []
            for x, y, dis in possible_static_obs_:
                static_obs_xy.append((x, y))

            # 计算障碍物的s,l
            obs_s_list, obs_l_list = planning_utils.cal_s_l_fun(static_obs_xy, local_frenet_path_opt_, s_map)
        else:
            obs_s_list, obs_l_list = [], []

        dynamic_obs_xy = []
        obs_dis_speed_list = []
        if len(possible_dynamic_obs_) != 0:
            for x, y, dis_, speed_ in possible_dynamic_obs_:
                dynamic_obs_xy.append((x, y))
                obs_dis_speed_list.append((dis_, speed_))

        # 计算规划起点的s, l
        begin_s_list, begin_l_list = planning_utils.cal_s_l_fun([pred_loc_], local_frenet_path_opt_, s_map)

        "自车从规划起点预测后面在不同时刻的位置， 同时预测障碍物在不同时刻的位置，确定二者交汇位置和时间，记录这些信息"
        if len(dynamic_obs_xy) != 0:
            Len_vehicle = 2.910  # 自车长度
            Len_obs = 3  # 障碍物车辆长度
            V_obs = obs_dis_speed_list[0][1]  # 障碍物的速度
            Dis = obs_dis_speed_list[0][0]  # 障碍物距离自车的距离
            V_ego = math.sqrt(vehicle_v_[0] ** 2 + vehicle_v_[1] ** 2)
            delta_v = V_ego - V_obs
            # print("V_ego, V_obs", V_ego, V_obs)
            # 相遇开始的时间和相遇结束的时间
            meet_t = (Dis - Len_vehicle / 2 - Len_obs / 2) / delta_v
            delta_t = (Len_vehicle + Len_obs) / delta_v
            leave_t = meet_t + delta_t
            # print("meet_t, leave_t", meet_t, leave_t)
            """
            meet_s 是障碍物在相遇时尾部的s值
            leave_s 是障碍物在与自车分离时头部的s值
            -------------00(meet_s)^^-------------------------
            -------------------------^^(leave_s)00------------
            """
            meet_s = begin_s_list[0] + Dis + V_obs * meet_t - Len_obs / 2
            leave_s = begin_s_list[0] + Dis + V_obs * leave_t + Len_obs / 2
            delta_s = leave_s - meet_s
            obs_pos = meet_s + delta_s / 2
            # print("meet_s and leave_s", begin_s_list[0], Dis, meet_s, leave_s)
            # print("障碍物位置和长度", obs_pos, delta_s)
            # print(obs_s_list)
            if leave_s < 80:
                obs_s_list.append(meet_s - 10)
                obs_s_list.append(obs_pos)
                obs_s_list.append(leave_s)
                obs_l_list.append(0)
                obs_l_list.append(0)
                obs_l_list.append(0)
        """从规划起点进行动态规划"""
        # 计算规划起点的l对s的导数和偏导数
        l_list, _, _, _, l_ds_list, _, l_dds_list = \
            planning_utils.cal_s_l_deri_fun(xy_list=[pred_loc_],
                                            V_xy_list=[vehicle_v_],
                                            a_xy_list=[vehicle_a_],
                                            local_path_xy_opt=local_frenet_path_opt_,
                                            origin_xy=pred_loc_)
        # 从起点开始沿着s进行横向和纵向采样，然后动态规划,相邻点之间依据五次多项式进一步采样，间隔一米
        # print("*motion planning time cost:", time.time() - start_time)
        dp_path_s, dp_path_l = path_planning.DP_algorithm(obs_s_list, obs_l_list,
                                                          plan_start_s=begin_s_list[0],
                                                          plan_start_l=l_list[0],
                                                          plan_start_dl=l_ds_list[0],
                                                          plan_start_ddl=l_dds_list[0])
        # print("**dp planning time cost:", time.time() - start_time)
        # 对动态规划得到的路径进行降采样，减少二次规划的计算量，然后二次规划完成后再插值填充恢复
        dp_path_l = dp_path_l[::2]
        dp_path_s = dp_path_s[::2]
        l_min, l_max = \
            path_planning.cal_lmin_lmax(dp_path_s=dp_path_s, dp_path_l=dp_path_l,
                                        obs_s_list=obs_s_list, obs_l_list=obs_l_list,
                                        obs_length=5, obs_width=5)  # 这一步的延迟很低，忽略不计

        # 二次规划变量过多会导致计算延迟比较高，需要平衡二者之间的关系
        # print("l_min_max_length", len(l_min))
        """二次规划"""
        qp_path_l, qp_path_dl, qp_path_ddl = \
            path_planning.Quadratic_planning(l_min, l_max,
                                             plan_start_l=l_list[0],
                                             plan_start_dl=l_ds_list[0],
                                             plan_start_ddl=l_dds_list[0])
        # print(qp_path_l)
        # print("**qp planning time cost:", time.time() - start_time)
        path_s = [dp_path_s[0]]
        path_l = [qp_path_l[0]]
        for i in range(1, len(qp_path_l)):
            path_s.append((dp_path_s[i] + dp_path_s[i - 1]) / 2)
            path_l.append((qp_path_l[i] + qp_path_l[i - 1]) / 2)
        path_s.append(dp_path_s[-1])
        path_l.append(qp_path_l[-1])

        cartesian_path_opt = \
            path_planning.frenet_2_x_y_theta_kappa(plan_start_s=begin_s_list[0],
                                                   plan_start_l=begin_l_list[0],
                                                   enriched_s_list=path_s,
                                                   enriched_l_list=path_l,
                                                   frenet_path_opt=local_frenet_path_opt_,
                                                   s_map=s_map)


        """
        #########################################
        #######     speed planning      #########
        #########################################
        """
        ## Frenet 转 Cartesian
        traj_x_init = [], traj_y_init = [], traj_heading_init = [], traj_kappa_init = []
        traj_init = []      # (x y heading kappa)
        for x, y, h, k in cartesian_path_opt:
            traj_x_init.append(x)
            traj_y_init.append(y)
            traj_heading_init.append(h)
            traj_kappa_init.append(k)
            traj_init.append((x,y,h,k))

        ## 轨迹转为s
        path_index2s = planning_utils.trajectory_index2s(trajectory_x_init, trajectory_y_init)

        ## 计算速度规划初始条件
        vehicle_heading_ = math.atan2(vehicle_v_[1], vehicle_v_[0])
        plan_start_s_dot, plan_start_s_dot2 =\
            speed_planning_test.calc_speed_planning_start_condition(plan_start_vx=vehicle_v_[0],
                                                                    plan_start_vy=vehicle_v_[1],
                                                                    plan_start_ax=vehicle_a_[0],
                                                                    plan_start_ay=vehicle_a_[1],
                                                                    plan_start_heading=vehicle_heading_)

        ## 生成ST图
        dy_obs_x_set = [], dy_obs_y_set = [], dy_obs_vx_set = [], dy_obs_vy_set = []
        dy_obs_xy_set = [], dy_obs_vxy_set = [], dy_obs_axy_set = []
        for x, y, vx, vy, ax, ay, _, _ in possible_dynamic_obs_:
            dy_obs_xy_set.append((x, y))
            dy_obs_vxy_set.append((vx, vy))
            dy_obs_axy_set.append((ax, ay))
            dy_obs_x_set.append(x)
            dy_obs_y_set.append(y)
            dy_obs_vx_set.append(vx)
            dy_obs_vy_set.append(vy)

        # proj_node_list: [(x_p0, y_p0, heading_p0, kappa_p0), ...]
        match_point_list_speed_, proj_node_list = planning_utils.find_match_points(xy_list=dy_obs_xy_set,
                                                                    frenet_path_node_list=traj_init,
                                                                    is_first_run=False,
                                                                    pre_match_index=match_point_list_[0])
        proj_heading_list = [], proj_kappa_list = []
        for _, _, h, k in proj_node_list:
            proj_heading_list.append(h)
            proj_kappa_list.append(k)
        # 动态obs投影
        dy_obs_s_list, dy_obs_l_list = planning_utils.cal_s_l_fun(obs_xy_list=dy_obs_xy_set,
                                                                  local_path_opt=traj_init,
                                                                  s_map=path_index2s)
        dy_obs_s_dot_list, dy_obs_l_dot_list, dy_obs_dl_list = planning_utils.cal_dy_obs_deri(l_set=dy_obs_l_list,
                                                                                              vx_set=dy_obs_vx_set,
                                                                                              vy_set=dy_obs_vy_set,
                                                                                              proj_heading_set=proj_heading_list,
                                                                                              proj_kappa_set=proj_kappa_list)
        obs_st_sin_list, obs_st_sout_list, obs_st_tin_list, obs_st_tout_list = \
            speed_planning_test.generate_st_graph(dynamic_obs_s_set=dy_obs_s_list,
                                                  dynamic_obs_l_set=dy_obs_l_list,
                                                  dynamic_obs_s_dot_set=dy_obs_s_dot_list,
                                                  dynamic_obs_l_dot_set=dy_obs_l_dot_list)

        ## 速度动态规划
        dp_speed_s, dp_speed_t = speed_planning_test.speed_DP(obs_st_s_in_set=obs_st_sin_list,
                                                              obs_st_s_out_set=obs_st_sout_list,
                                                              obs_st_t_in_set=obs_st_tin_list,
                                                              obs_st_t_out_set=obs_st_tout_list,
                                                              plan_start_s_dot=plan_start_s_dot)

        ## 生成凸空间
        s_lb, s_ub, s_dot_lb, s_dot_ub = speed_planning_test.generate_convex_space(dp_speed_s=dp_speed_s,
                                                                                   dp_speed_t=dp_speed_t,
                                                                                   path_index2s=path_index2s,
                                                                                   obs_st_s_in_set=obs_st_sin_list,
                                                                                   obs_st_s_out_set=obs_st_sout_list,
                                                                                   obs_st_t_in_set=obs_st_tin_list,
                                                                                   obs_st_t_out_set=obs_st_tout_list,
                                                                                   trajectory_kappa_init=traj_kappa_init)

        ## 速度二次规划
        qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init = speed_planning_test.speed_QP(plan_start_s_dot=plan_start_s_dot,
                                                                                                    plan_start_s_dot2=plan_start_s_dot2,
                                                                                                    dp_speed_s=dp_speed_s,
                                                                                                    dp_speed_t=dp_speed_t,
                                                                                                    s_lb=s_lb,s_ub=s_ub,
                                                                                                    s_dot_lb=s_dot_lb,
                                                                                                    s_dot_ub=s_dot_ub)

        ## 轨迹增密
        _s, _s_dot, _s_dot2, _relative_time = speed_planning_test.increase_points(s_init=qp_s_init,
                                                                             s_dot_init=qp_s_dot_init,
                                                                             s_dot2_init=qp_s_dot2_init,
                                                                             relative_time_init=relative_time_init)

        ## path speed合并
        # 由于没有进行路径规划时的轨迹拼接，所以这里的start_time直接简化为cur_time+0.1 (规划周期10Hz)
        plan_start_time = cur_time + 0.1
        traj_x, traj_y, traj_heading, traj_kappa, traj_speed, traj_accel,traj_time = \
            speed_planning_test.path_speed_merge(s=_s,
                                                 s_dot=_s_dot,
                                                 s_dot2=_s_dot2,
                                                 relative_time=_relative_time,
                                                 current_time=plan_start_time,
                                                 path_s=path_index2s,
                                                 trajectory_x_init=traj_x_init,
                                                 trajectory_y_init=traj_y_init,
                                                 trajectory_heading_init=traj_heading_init,
                                                 trajectory_kappa_init=traj_kappa_init)












        # 将重新规划得到的路径信息发送给主进程，让控制器进行轨迹跟踪
        conn.send((cartesian_path_opt, match_point_list_speed_, path_s, path_l))
        # print("***motion planning time cost:", time.time() - start_time)


if __name__ == '__main__':
    conn1, conn2 = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=motion_planning, args=(conn2,))
    p1.start()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10)
    # 对象创建好了之后，在对象中添加需要的环境中的地图
    world = client.load_world('Town05')  # type: carla.World
    amap = world.get_map()  # type: carla.Map
    topo = amap.get_topology()
    global_route_plan = global_path_planner(world_map=amap, sampling_resolution=2)  # 实例化全局规划器
    # topology, graph, id_map, road_id_to_edge = global_route_plan.get_topology_and_graph_info()

    All_spawn_points = amap.get_spawn_points()  # 获取所有carla提供的actor产生位置
    """# 定义一个ego-vehicle"""
    model3_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    model3_bp.set_attribute('color', '255,88,0')
    model3_spawn_point = All_spawn_points[259]
    # print(model3_spawn_point)
    # model3_spawn_point.location = model3_spawn_point.location + carla.Location(x=-100, y=0, z=0)
    model3_actor = world.spawn_actor(model3_bp, model3_spawn_point)  # type: carla.Vehicle
    # 定义轮胎特性
    # wheel_f = carla.WheelPhysicsControl()  # type: carla.WheelPhysicsControl
    # 定义车辆特性
    physics_control = carla.VehiclePhysicsControl()  # type: carla.VehiclePhysicsControl
    physics_control.mass = 1412  # 质量kg
    model3_actor.apply_physics_control(physics_control)
    """为车辆配备一个障碍物的传感器"""
    # obs_detector = Obstacle_detector(ego_vehicle=model3_actor, world=world)  # 实例化传感器
    # obs_detector.create_sensor()  # 在仿真环境中生成传感器

    """设置静止车辆"""
    # 静止车辆1
    obs_vehicle_bp1 = world.get_blueprint_library().find('vehicle.tesla.model3')
    obs_vehicle_bp1.set_attribute('color', '0,0,255')
    obs_spawn_point1 = carla.Transform()
    obs_spawn_point1.location = carla.Location(x=174.01, y=147.61, z=0.3)
    obs_spawn_point1.rotation = carla.Rotation(yaw=30)
    obs_actor = world.spawn_actor(obs_vehicle_bp1, obs_spawn_point1)  # type: carla.Vehicle

    # 静止车辆2
    obs_vehicle_bp2 = world.get_blueprint_library().find('vehicle.audi.tt')
    obs_vehicle_bp2.set_attribute('color', '0,255,0')
    obs_spawn_point2 = carla.Transform()
    obs_spawn_point2.location = carla.Location(x=105.86, y=189.11, z=0.3)
    obs_spawn_point2.rotation = carla.Rotation(yaw=90)
    obs_actor2 = world.spawn_actor(obs_vehicle_bp2, obs_spawn_point2)  # type: carla.Vehicle

    # 静止车辆3
    obs_vehicle_bp3 = world.get_blueprint_library().find('vehicle.audi.tt')
    obs_vehicle_bp3.set_attribute('color', '0,255,0')
    obs_spawn_point3 = carla.Transform()
    obs_spawn_point3.location = carla.Location(x=105.86, y=194.11, z=0.3)
    obs_spawn_point3.rotation = carla.Rotation(yaw=90)
    obs_actor2 = world.spawn_actor(obs_vehicle_bp3, obs_spawn_point3)  # type: carla.Vehicle

    # 运动车辆
    obs_dy_vehicle_bp1 = world.get_blueprint_library().find('vehicle.tesla.model3')
    obs_dy_vehicle_bp1.set_attribute('color', '0,0,255')
    obs_dy_spawn_point1 = carla.Transform()
    obs_dy_spawn_point1.location = carla.Location(x=192.31, y=10, z=0.3)
    obs_dy_spawn_point1.rotation = model3_spawn_point.rotation
    obs_actor1 = world.spawn_actor(obs_dy_vehicle_bp1, obs_dy_spawn_point1)  # type: carla.Vehicle
    agent = BehaviorAgent(obs_actor1, "normal")
    destination = All_spawn_points[48].location
    agent.set_destination(destination)
    agent.set_target_speed(30.0)

    """路径规划"""
    # 1. 规划路径，输出的每个路径点是一个元组形式[(wp, road_option), ...]第一个是元素是carla中的路点，第二个是当前路点规定的一些车辆行为
    pathway = global_route_plan.search_path_way(origin=model3_spawn_point.location,
                                                destination=All_spawn_points[48].location)
    debug = world.debug  # type: carla.DebugHelper

    # 2. 将路径点构成的路径转换为[(x, y, theta, kappa], ...]的形式
    global_frenet_path = planning_utils.waypoint_list_2_target_path(pathway)

    # 3.提取局部路径
    transform = model3_actor.get_transform()
    vehicle_loc = transform.location  # 获取车辆的当前位置
    match_point_list, _ = planning_utils.find_match_points(xy_list=[(vehicle_loc.x, vehicle_loc.y)],
                                                           frenet_path_node_list=global_frenet_path,
                                                           is_first_run=True,  # 寻找车辆起点的匹配点就属于第一次运行，
                                                           pre_match_index=0)  # 没有上一次运行得到的索引，索引自然是全局路径的起点
    local_frenet_path = planning_utils.sampling(match_point_list[0], global_frenet_path)
    local_frenet_path_opt = planning_utils.smooth_reference_line(local_frenet_path)
    # 计算参考线的s, l
    cur_s_map = planning_utils.cal_s_map_fun(local_frenet_path_opt, origin_xy=(vehicle_loc.x, vehicle_loc.y))
    cur_path_s, cur_path_l = planning_utils.cal_s_l_fun(local_frenet_path_opt, local_frenet_path_opt, cur_s_map)

    """整车参数设定"""
    vehicle_para = (1.015, 2.910 - 1.015, 1412, -148970, -82204, 1537)      # 车辆特性(侧偏刚度、转动惯量...)
    controller = "MPC_controller"  # "MPC_controller" or "LQR_controller"
    Controller = Vehicle_control(ego_vehicle=model3_actor, vehicle_para=vehicle_para,
                                 pathway=local_frenet_path_opt,
                                 controller_type=controller)  # 实例化控制器
    DIS = math.sqrt((pathway[0][0].transform.location.x - pathway[1][0].transform.location.x) ** 2
                    + (pathway[0][0].transform.location.y - pathway[1][0].transform.location.y) ** 2)  # 计算轨迹相邻点之间的距离
    # print("The distance between two adjacent points in route:", DIS)
    direction = []
    speed = []
    plt_target_speed = []
    ref_speed = 50  # 假设正常行驶速度是40km/h，市区允许的最大速度是50km/h, 前方的动态障碍物行驶的速度是30 km/h.
    # 一旦选择超车，自车可以将速度提升到50km/h
    # 设定一个观察者视角
    spectator = world.get_spectator()
    count = 1  # 控制规划器和控制器相对频率
    main_process_start_time = time.time()
    plan_count = 100
    # control_count = 10
    pred_ts = 0.2
    while True:
        """获取交通速度标志,考虑道路速度限制"""
        # 获取车辆位置信息（包括坐标和姿态信息）， get the transformation, a combination of location and rotation
        transform = model3_actor.get_transform()
        # 不断更新观测视角的位置， update the position of spectator
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40), carla.Rotation(pitch=-90)))
        vehicle_loc = transform.location  # 获取车辆的当前位置

        obs_actor1.apply_control(agent.run_step())
        """获取局部路径，局部路径规划的频率是控制的1/100"""
        if count % plan_count == 0:  # 这里表示控制器执行100次规划器执行1次
            cur_time = time.time()
            # print("main_process_cost_time", time.time() - main_process_start_time)
            main_process_start_time = cur_time
            # mark = "replan" + str((round(vehicle_loc.x, 2), round(vehicle_loc.y, 2)))
            # world.debug.draw_string(carla.Location(vehicle_loc.x, vehicle_loc.y, 2), mark, draw_shadow=False,
            #                         color=carla.Color(r=0, g=0, b=255), life_time=1000,
            #                         persistent_lines=True)
            # debug.draw_point(carla.Location(vehicle_loc.x, vehicle_loc.y, 2),
            #                  size=0.05, color=carla.Color(0, 0, 0), life_time=0)

            vehicle_loc = model3_actor.get_transform().location
            vehicle_v = model3_actor.get_velocity()
            vehicle_a = model3_actor.get_acceleration()
            # 基于笛卡尔坐标系预测ts秒过后车辆的位置，以预测点作为规划起点
            pred_x, pred_y, pred_fi = planning_utils.predict_block(model3_actor, ts=pred_ts)
            # 基于frenet坐标系预测ts秒过后车辆的位置，以预测点作为规划起点
            # pred_x, pred_y = planning_utils.predict_block_based_on_frenet(vehicle_loc, vehicle_v,
            #                                                               local_frenet_path_opt,
            #                                                               cur_path_s, cur_path_l, ts=0.2)

            # mark = "predict" + str((round(pred_x, 2), round(pred_y, 2)))
            # world.debug.draw_string(carla.Location(pred_x, pred_y, 2), mark, draw_shadow=False,
            #                         color=carla.Color(r=0, g=0, b=255), life_time=1000,
            #                         persistent_lines=True)
            # debug.draw_point(carla.Location(pred_x, pred_y, 2),
            #                  size=0.05, color=carla.Color(255, 255, 255), life_time=0)
            """
            没有找到合适的传感器，暂时用车联网的方法,设定合适的感知范围，获取周围环境中的actor，这里我们人为制造actor作为障碍物
            再到后面可以考虑用多传感器数据融合来做动态和静态障碍物的融合感知
            """
            possible_static_obs, possible_dynamic_obs = get_actor_from_world(model3_actor, dis_limitation=50)
            # 提取障碍物的位置信息
            static_obs_info = []
            for obs_v, dis in possible_static_obs:
                obs_loc = obs_v.get_transform().location
                static_obs_info.append((obs_loc.x, obs_loc.y, dis))
                # print("static_obs_id:", obs_v.type_id, "dis:", dis)
            dynamic_obs_info = []
            for _obs_v, _dis, _speed in possible_dynamic_obs:
                obs_loc = _obs_v.get_transform().location
                obs_vx = obs_v.get_velocity().x
                obs_vy = obs_v.get_velocity().y
                obs_ax = obs_v.get_acceleration().x
                obs_ay = obs_v.get_acceleration().y
                dynamic_obs_info.append((obs_loc.x, obs_loc.y, obs_vx, obs_vy, obs_ax, obs_ay, _dis, _speed))
                # print("dynamic_obs_id:", obs_v.type_id, "dis:", dis)
            # 将当前的道路状况和车辆信息发送给规划器进行规划控制
            conn1.send((static_obs_info, dynamic_obs_info, (vehicle_loc.x, vehicle_loc.y), (pred_x, pred_y),
                        (vehicle_v.x, vehicle_v.y), (vehicle_a.x, vehicle_a.y),
                        global_frenet_path, match_point_list, time.time()))

            if count != plan_count:  # 第一个循环周期，因为有初始阶段规划好的局部路径，第二个周期的规划还未计算完成，一旦执行接收数据，会阻塞主进程
                cur_local_frenet_path_opt, match_point_list, cur_path_s, cur_path_l = conn1.recv()  # 新规划出的轨迹
                """轨迹拼接
                思路比较简单，由于规划是在预测点进行的，对下个周期进行规划，因此当前周期的车辆运动结束点一定在预测点之前，
                找到上个规划周期轨迹中距离预测点最近的点，与新的规划路径进行拼接，保证轨迹的连续性,拼接完之后还需要进一步平滑
                """
                # min_DIS = 10000
                # for i in range(len(local_frenet_path_opt)):
                #     if (pred_x - local_frenet_path_opt[i][0]) ** 2 + (
                #             pred_y - local_frenet_path_opt[i][1]) ** 2 < min_DIS:
                #         min_DIS = (pred_x - local_frenet_path_opt[i][0]) ** 2 + (
                #                 pred_y - local_frenet_path_opt[i][1]) ** 2
                #     else:
                #         local_frenet_path_opt = local_frenet_path_opt[0:i] + cur_local_frenet_path_opt
                #         break
                local_frenet_path_opt = cur_local_frenet_path_opt
                for point in local_frenet_path_opt:
                    # print(waypoint)
                    debug.draw_point(carla.Location(point[0], point[1], 2),
                                     size=0.05, color=carla.Color(255, 0, 0), life_time=0.3)
            # 注意重新实例化控制器的位置，不能放错了
            Controller = Vehicle_control(ego_vehicle=model3_actor, vehicle_para=vehicle_para,
                                         pathway=local_frenet_path_opt,
                                         controller_type=controller)  # 依据当前局部路径实例化控制器

        """控制部分"""
        control = Controller.run_step(target_speed=ref_speed)  # 实例化的时候已经将必要的信息传递给规划器，这里告知目标速度即可
        direction.append(model3_actor.get_transform().rotation.yaw * (math.pi / 180))
        V = model3_actor.get_velocity()  # 利用 carla API to 获取速度矢量， use the API of carla to get the velocity vector
        V_len = 3.6 * math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)  # transfer m/s to km/h
        speed.append(V_len)
        plt_target_speed.append(ref_speed)
        model3_actor.apply_control(control)  # 执行最终控制指令, execute the final control signal

        """debug 部分"""
        # # 将预测点和投影点的位置标出来, mark the predicted point and project point in the simulation world for debug
        debug.draw_point(carla.Location(Controller.Lat_control.x_pre, Controller.Lat_control.y_pre, 2),
                         size=0.05, color=carla.Color(0, 0, 255), life_time=0)  # 预测点为蓝色
        debug.draw_point(carla.Location(Controller.Lat_control.x_pro, Controller.Lat_control.y_pro, 2),
                         size=0.05, color=carla.Color(0, 255, 0), life_time=0)  # 投影点为绿色

        """距离判断，程序终止条件"""
        count += 1
        # 计算当前车辆和终点的距离, calculate the distance between vehicle and destination
        dist = vehicle_loc.distance(pathway[-1][0].transform.location)
        # print("The distance to the destination: ", dist)

        if dist < 2:  # 到达终点后产生制动信号让车辆停止运动
            control = emergence_brake()
            model3_actor.apply_control(control)
            # print("last waypoint reached")
            p1.terminate()
            break

    """可视化速度变化和航向变化"""
    import matplotlib.pyplot as plt

    # plt.figure(1)
    # plt.plot(direction)
    # plt.ylim(bottom=-5, top=5)
    #
    # plt.figure(2)
    # plt.plot(speed)
    # plt.plot(target_speed, color="r")
    # plt.ylim(bottom=0, top=max(plt_target_speed) + 10)
    # plt.show()
