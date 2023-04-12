#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : test_5.py

"""
实现障碍物的"感知"，进行局部路径规划
1.获取匹配点的索引
2.根据匹配点的索引在全局路径上采样back+front个点
3.对采样点进行平滑
4.轨迹拼接, 未做
5.根据采样得到的局部路径为参考线，实现静态障碍物的避障碍（采用动态规划算法）
"""

import carla
import math
import matplotlib.pyplot as plt
import numpy as np

from planner import planning_utils, path_planning
from controller.controller import Vehicle_control
from planner.global_planning import global_path_planner


# from sensors.Sensors_detector_lib import Obstacle_detector


def get_traffic_light_state(current_traffic_light):  # get_state()方法只能返回红灯和黄灯状态，没有绿灯状态（默认把绿灯认为正常）
    # 此方法把红绿蓝三种状态都标定出来
    if current_traffic_light is None:
        current_traffic_light_state = "Green"
    else:
        current_traffic_light_state = current_traffic_light.get_state()
    return current_traffic_light_state


def emergence_brake():
    brake_control = carla.VehicleControl()
    brake_control.steer = 0  # 转向控制
    brake_control.throttle = 0  # 油门控制
    brake_control.brake = 1  # 刹车控制
    return brake_control


def get_actor_from_world(ego_vehicle: carla.Vehicle, carla_world: carla.World, dis_limitation=50):
    """已验证
    获取当前车辆前方潜在的车辆障碍物
    首先获取在主车辆一定范围内的其他车辆，再通过速度矢量和位置矢量将在主车辆运动方向后方的车辆过滤掉
    param:  ego_vehicle: 主车辆
            carla_world: carla环境
            dis_limitation: 探测范围
    return: v_list:(vehicle, dist)
    """
    v_list = []  # 储存范围内的车辆
    ego_vehicle_loc = ego_vehicle.get_location()
    vehicle_list = carla_world.get_actors().filter("vehicle.*")
    for vehicle in vehicle_list:
        dis = math.sqrt((ego_vehicle_loc.x - vehicle.get_location().x) ** 2 +
                        (ego_vehicle_loc.y - vehicle.get_location().y) ** 2 +
                        (ego_vehicle_loc.z - vehicle.get_location().z) ** 2)
        if dis < dis_limitation and ego_vehicle.id != vehicle.id:
            v1 = np.array([vehicle.get_location().x - ego_vehicle_loc.x,
                           vehicle.get_location().y - ego_vehicle_loc.y,
                           vehicle.get_location().z - ego_vehicle_loc.z])  # 其他车辆到ego_vehicle的矢量
            ego_vehicle_velocity = np.array([ego_vehicle.get_velocity().x, ego_vehicle.get_velocity().y,
                                             ego_vehicle.get_velocity().z])  # ego_vehicle的速度矢量
            # 判断车辆是否在ego_vehicle的运动方向的前方
            if np.dot(v1, ego_vehicle_velocity) > 0:  # 如果车辆出现在ego_vehicle的运动前方，则有可能是障碍物
                # 还需要控制可能的障碍物距离参考线的横向距离, 我的想法是将障碍物在参考线上投影，计算投影点和车辆的距离，
                # 如果距离大于阈值则认为不影响ego-vehicle的运动，反之认为是障碍物会影响ego-vehicle的运动
                # 现在简化一下，将横向距离暂时设定为ego-vehicle当前航向方向的垂直距离
                ego_vehicle_theta = ego_vehicle.get_transform().rotation.yaw * (math.pi / 180)
                n_r = np.array([-math.sin(ego_vehicle_theta), math.cos(ego_vehicle_theta), 0])
                if -10 < np.dot(v1, n_r) < 12:
                    v_list.append((vehicle, dis))
    v_list.sort(key=lambda tup: tup[1])  # 按距离排序
    return v_list


client = carla.Client("localhost", 2000)
client.set_timeout(10)
# 对象创建好了之后，在对象中添加需要的环境中的地图
world = client.load_world('Town05')  # type: carla.World
amap = world.get_map()  # type: carla.Map
topo = amap.get_topology()
global_route_plan = global_path_planner(world_map=amap, sampling_resolution=3)  # 实例化全局规划器
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
# # 静止车辆1
obs_vehicle_bp1 = world.get_blueprint_library().find('vehicle.tesla.model3')
obs_vehicle_bp1.set_attribute('color', '0,0,255')
obs_spawn_point1 = carla.Transform()
obs_spawn_point1.location = carla.Location(x=189.31, y=76.61, z=0.3)
obs_spawn_point1.rotation = carla.Rotation(yaw=60)
obs_actor1 = world.spawn_actor(obs_vehicle_bp1, obs_spawn_point1)  # type: carla.Vehicle
#
# # 静止车辆2
# obs_vehicle_bp2 = world.get_blueprint_library().find('vehicle.audi.a2')
# obs_vehicle_bp2.set_attribute('color', '0,255,0')
# obs_spawn_point2 = carla.Transform()
# obs_spawn_point2.location = carla.Location(x=196.31, y=58.61, z=0.3)
# obs_spawn_point2.rotation = model3_spawn_point.rotation
# obs_actor2 = world.spawn_actor(obs_vehicle_bp2, obs_spawn_point2)  # type: carla.Vehicle
#
# # 静止车辆3
# obs_vehicle_bp3 = world.get_blueprint_library().find('vehicle.dodge.charger_police')
# obs_vehicle_bp3.set_attribute('color', '255,0,0')
# obs_spawn_point3 = carla.Transform()
# obs_spawn_point3.location = carla.Location(x=185.01, y=76.61, z=0.3)
# obs_spawn_point3.rotation = model3_spawn_point.rotation
# obs_actor3 = world.spawn_actor(obs_vehicle_bp3, obs_spawn_point3)  # type: carla.Vehicle
#
# # 静止车辆4
# obs_vehicle_bp4 = world.get_blueprint_library().find('vehicle.toyota.prius')
# obs_vehicle_bp4.set_attribute('color', '255,255,0')
# obs_spawn_point4 = carla.Transform()
# obs_spawn_point4.location = carla.Location(x=204.01, y=64.61, z=0.3)
# obs_spawn_point4.rotation = model3_spawn_point.rotation
# obs_actor4 = world.spawn_actor(obs_vehicle_bp4, obs_spawn_point4)  # type:

# carla.Vehicle# 静止车辆5
obs_vehicle_bp5 = world.get_blueprint_library().find('vehicle.toyota.prius')
obs_vehicle_bp5.set_attribute('color', '255,255,0')
obs_spawn_point5 = carla.Transform()
obs_spawn_point5.location = carla.Location(x=174.01, y=147.61, z=0.3)
obs_spawn_point5.rotation = model3_spawn_point.rotation
obs_actor5 = world.spawn_actor(obs_vehicle_bp5, obs_spawn_point5)  # type: carla.Vehicle

"""路径规划"""
# 1. 规划路径，输出的每个路径点是一个元组形式【(wp, road_option), ...】第一个是元素是carla中的路点，第二个是当前路点规定的一些车辆行为
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
"""整车参数设定"""
# vehicle_para = (1.015, 2.910-1.015, 1412, -110000, -110000, 1537)
vehicle_para = (1.015, 2.910 - 1.015, 1412, -148970, -82204, 1537)
# Lat_controller = Lateral_LQR_controller(ego_vehicle=model3_actor, vehicle_para=vehicle_para, pathway=pathway)
Controller = Vehicle_control(ego_vehicle=model3_actor, vehicle_para=vehicle_para,
                             pathway=local_frenet_path_opt, controller_type="MPC_controller")  # 实例化控制器
DIS = math.sqrt((pathway[0][0].transform.location.x - pathway[1][0].transform.location.x) ** 2
                + (pathway[0][0].transform.location.y - pathway[1][0].transform.location.y) ** 2)  # 计算轨迹相邻点之间的距离
# print("The distance between two adjacent points in route:", DIS)
direction = []
speed = []
target_speed = []
max_speed = 50  # 初始速度设为50km/h

# 设定一个观察者视角
spectator = world.get_spectator()
count = 1
planning_count = 500
pred_ts = 0.2

while True:
    """获取交通速度标志,考虑道路速度限制"""
    # wp = amap.get_waypoint(vehicle_loc)  # type: carla.Waypoint
    # Landmark = wp.get_landmarks(distance=5, stop_at_junction=True)
    # for lm in Landmark:
    #     traffic_sign = world.get_traffic_sign(lm)  # type: carla.TrafficSign
    #     t_sign = traffic_sign.type_id.split(".")
    #     if 'speed_limit' in t_sign:
    #         max_speed = float(t_sign[-1])

    # 获取车辆位置信息（包括坐标和姿态信息）， get the transformation, a combination of location and rotation
    transform = model3_actor.get_transform()
    # 不断更新观测视角的位置， update the position of spectator
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40), carla.Rotation(pitch=-90)))
    vehicle_loc = transform.location  # 获取车辆的当前位置
    """获取局部路径，局部路径规划的频率是控制的1/500"""
    if count % planning_count == 0:  # 这里表示控制器执行500次规划器执行1次
        # 1.获取车辆当前位置在参考线上匹配点的索引
        # mark = "replan"
        # world.debug.draw_string(carla.Location(vehicle_loc.x, vehicle_loc.y, 2), mark, draw_shadow=False,
        #                                                     color=carla.Color(r=0, g=0, b=255), life_time=1000,
        #                                                     persistent_lines=True)

        match_point_list, _ = planning_utils.find_match_points(xy_list=[(vehicle_loc.x, vehicle_loc.y)],
                                                               frenet_path_node_list=global_frenet_path,
                                                               is_first_run=False,
                                                               pre_match_index=match_point_list[0])
        # 2.根据匹配点的索引在全局路径上采样一定数量的点
        local_frenet_path = planning_utils.sampling(match_point_list[0], global_frenet_path)
        # 3.对采样点进行平滑
        local_frenet_path_opt = planning_utils.smooth_reference_line(local_frenet_path)
        # 4.轨迹拼接(未完成)
        for point in local_frenet_path_opt:
            # print(waypoint)
            debug.draw_point(carla.Location(point[0], point[1], 2),
                             size=0.05, color=carla.Color(0, 0, 255), life_time=1)  # 蓝色为平滑后的参考线

        """
        没有找到合适的传感器，暂时用车联网的方法,设定合适的感知范围，获取周围环境中的actor，这里我们人为制造actor作为障碍物
        再到后面可以考虑用多传感器数据融合来做动态和静态障碍物的融合感知"""
        possible_obs = get_actor_from_world(model3_actor, world, dis_limitation=50)
        # 提取障碍物的位置信息
        # TODO 后面可以设计一下检测到可能障碍物后将最大速度降到30，离开后再恢复到道路要求的速度
        if len(possible_obs) != 0 and possible_obs[0][1] <= 30 and possible_obs[-1][1] > 10:
            obs_xy = []
            print("**********  Find %d possible obstacles in front  **********" % (len(possible_obs)))
            for obs_v, dis in possible_obs:
                obs_loc = obs_v.get_transform().location
                obs_xy.append((obs_loc.x, obs_loc.y))
                print("obs_id:", obs_v.type_id, "obs_dis:", dis)

            s_map = planning_utils.cal_s_map_fun(local_frenet_path_opt, origin_xy=(vehicle_loc.x, vehicle_loc.y))
            obs_s_list, obs_l_list = planning_utils.cal_s_l_fun(obs_xy, local_frenet_path_opt, s_map)

            pred_x, pred_y, pred_fi = planning_utils.predict_block(model3_actor, ts=pred_ts)
            begin_s_list, begin_l_list = planning_utils.cal_s_l_fun([(pred_x, pred_y)], local_frenet_path_opt, s_map)
            """从规划起点进行动态规划，理论上规划起点应该是当前时刻加上T0, T0是规划周期；由于规划是比较耗时的，所以等规划结束车辆应该运动一段时间了，
            目前没有写速度规划，控制是根据当前位置在参考线上的投影进行追踪的，所以先不考虑规划起点的预测"""
            vehicle_loc = model3_actor.get_transform().location
            vehicle_v = model3_actor.get_velocity()
            vehicle_a = model3_actor.get_acceleration()
            # 计算规划起点的l对s的导数和偏导数
            l_list, _, _, _, l_ds_list, _, l_dds_list = \
                planning_utils.cal_s_l_deri_fun(xy_list=[(vehicle_loc.x, vehicle_loc.y)],
                                                V_xy_list=[(vehicle_v.x, vehicle_v.y)],
                                                a_xy_list=[(vehicle_a.x, vehicle_a.y)],
                                                local_path_xy_opt=local_frenet_path_opt,
                                                origin_xy=(vehicle_loc.x, vehicle_loc.y))
            # 从起点开始沿着s进行横向和纵向采样，然后动态规划,相邻点之间依据五次多项式进一步采样，间隔一米
            dp_path_s, dp_path_l = path_planning.DP_algorithm(obs_s_list, obs_l_list,
                                                              plan_start_s=begin_s_list[0],
                                                              plan_start_l=begin_l_list[0],
                                                              plan_start_dl=l_ds_list[0],
                                                              plan_start_ddl=l_dds_list[0])

            local_frenet_path_opt = path_planning.frenet_2_x_y_theta_kappa(plan_start_s=begin_s_list[0],
                                                                           plan_start_l=begin_l_list[0],
                                                                           enriched_s_list=dp_path_s,
                                                                           enriched_l_list=dp_path_l,
                                                                           frenet_path_opt=local_frenet_path_opt,
                                                                           s_map=s_map)

            for point in local_frenet_path_opt:
                # print(waypoint)
                debug.draw_point(carla.Location(point[0], point[1], 2),
                                 size=0.05, color=carla.Color(255, 0, 0), life_time=0)
        # 注意重新实例化控制器的位置，不能放错了
        Controller = Vehicle_control(ego_vehicle=model3_actor, vehicle_para=vehicle_para,
                                     pathway=local_frenet_path_opt, controller_type="MPC_controller")  # 依据当前局部路径实例化控制器

    """控制部分"""
    control = Controller.run_step(target_speed=max_speed)  # 实例化的时候已经将必要的信息传递给规划器，这里告知目标速度即可
    direction.append(model3_actor.get_transform().rotation.yaw * (math.pi / 180))
    V = model3_actor.get_velocity()  # 利用 carla API to 获取速度矢量， use the API of carla to get the velocity vector
    V_len = 3.6 * math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)  # transfer m/s to km/h
    speed.append(V_len)
    target_speed.append(max_speed)
    model3_actor.apply_control(control)  # 执行最终控制指令, execute the final control signal

    """debug 部分"""
    # # 将预测点和投影点的位置标出来, mark the predicted point and project point in the simulation world for debug
    # debug.draw_point(carla.Location(Controller.Lat_control.x_pre, Controller.Lat_control.y_pre, 2),
    #                  size=0.05, color=carla.Color(0, 255, 255), life_time=0)
    # debug.draw_point(carla.Location(Controller.Lat_control.x_pro, Controller.Lat_control.y_pro, 2),
    #                  size=0.05, color=carla.Color(100, 0, 0), life_time=0)

    """距离判断，程序终止条件"""
    count += 1
    # 计算当前车辆和终点的距离, calculate the distance between vehicle and destination
    dist = vehicle_loc.distance(pathway[-1][0].transform.location)
    # print("The distance to the destination: ", dist)
    if dist < 2:  # 到达终点后产生制动信号让车辆停止运动
        control = emergence_brake()
        model3_actor.apply_control(control)
        # print("last waypoint reached")
        break

"""可视化速度变化和航向变化"""
plt.figure(1)
plt.plot(direction)
plt.ylim(bottom=-5, top=5)

plt.figure(2)
plt.plot(speed)
plt.plot(target_speed, color="r")
plt.ylim(bottom=0, top=max(target_speed) + 10)
plt.show()
