#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : test_7.py

"""
实现障碍物的感知，进行局部路径规划，完善规划轨迹的拼接
1.获取匹配点的索引
2.根据匹配点的索引在全局路径上采样61个点
3.对采样点进行平滑
4.轨迹拼接
5.根据采样得到的局部路径为参考线，实现静态障碍物的避障碍（采用动态规划算法）
6.采用多进程实现控制和规划分离
7.二次规划实现更加平滑的避障
此外还存在一个问题是两个周期之间的规划不连续
我之前采取的策略是遇到障碍物的时候才采用重规划，判断无障碍物的时候就直接采用参考线作为下个周期的规划路径
这里存在一个问题就是如果车辆在上个周期结束时刚好位于参考线上或者偏差很小时且下一个周期无障碍物，这时路径之间是连续的，
但是如果上个周期结束时没有回到参考线，下一个周期直接采用参考线作为局部路径，这时相邻的路径两个相交处就是不连续的
目前想到两个解决办法：1.直接采用轨迹拼接然后进行平滑，如果偏差较大的话，平滑也很难纠正，这种做法相当于打补丁，不采用
2.不再采用参考线作为局部规划路径，任何情况都让规划期间期间器进行动态规划和二次规划进行规划，
这里就需要优化一下动态规划，如果没有障碍物，动态规划就不进行复杂计算，直接生成路径，在本文件code7中我采用的是这种方法


目前存在的问题是规划模块计算量过大，在遇到障碍物时规划器无法在规划周期内完成计算,下一个更新重点优化计算量
"""

import carla

import multiprocessing
import time
import math
import numpy as np

from planner import utils, path_planning
from controller.controller import Vehicle_control
from planner.global_planning import global_path_planner
from sensors.Sensors_detector_lib import Obstacle_detector


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


def get_actor_from_world(ego_vehicle: carla.Vehicle, dis_limitation=50):
    """已验证
    获取当前车辆前方潜在的车辆障碍物
    首先获取在主车辆一定范围内的其他车辆，再通过速度矢量和位置矢量将在主车辆运动方向后方的车辆过滤掉
    :param ego_vehicle: 主车辆
    :param dis_limitation: 探测范围
    :return:
    """
    carla_world = ego_vehicle.get_world()  # type:carla.World
    vehicle_loc = ego_vehicle.get_location()
    v_list = []  # 储存范围内的车辆
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
            # 现在简化一下，将横向距离暂时设定为ego-vehicle当前航向方向的垂直距离"""
            ego_vehicle_theta = ego_vehicle.get_transform().rotation.yaw * (math.pi / 180)
            n_r = np.array([-math.sin(ego_vehicle_theta), math.cos(ego_vehicle_theta), 0])
            if -10 < np.dot(v1, n_r) < 12:
                if np.dot(v1, ego_vehicle_velocity) > 0:
                    v_list.append((vehicle, dis))
                elif np.dot(v1, ego_vehicle_velocity) < 0 and dis < 10:  # 自车后面十米以内的障碍物仍然考虑，超过十米就不再考虑
                    v_list.append((vehicle, -dis))
    v_list.sort(key=lambda tup: tup[1])  # 按距离排序
    return v_list


def motion_planning(conn):
    while 1:
        # 接收主进程发送的用于局部路径规划的数据，如果没有收到数据子进程会阻塞
        possible_obs_, vehicle_loc_, pred_loc_, vehicle_v_, vehicle_a_, global_frenet_path_, match_point_list_ = conn.recv()
        start_time = time.time()
        # 1.确定预测点在全局路径上的投影点索引
        match_point_list_, _ = planner_utils.find_match_points(xy_list=[pred_loc_],
                                                               frenet_path_node_list=global_frenet_path_,
                                                               is_first_run=False,
                                                               pre_match_index=match_point_list_[0])
        # 2.根据匹配点的索引在全局路径上采样一定数量的点
        local_frenet_path_ = planner_utils.sampling(match_point_list_[0], global_frenet_path_,
                                                    back_length=10, forward_length=30)
        # 3.对采样点进行平滑，作为后续规划的参考线
        local_frenet_path_opt_ = planner_utiles.smooth_reference_line(local_frenet_path_)

        # 计算以车辆当前位置为原点的s_map
        s_map = utils.cal_s_map_fun(local_frenet_path_opt_, origin_xy=vehicle_loc_)
        path_s, path_l = planner_utils.cal_s_l_fun(local_frenet_path_opt_, local_frenet_path_opt_, s_map)
        # 提取障碍物的位置信息
        if len(possible_obs_) != 0 and possible_obs_[0][-1] <= 30:
            obs_xy = []
            for x, y, dis in possible_obs_:
                obs_xy.append((x, y))

            # 计算障碍物的s,l
            obs_s_list, obs_l_list = planner_utils.cal_s_l_fun(obs_xy, local_frenet_path_opt_, s_map)
        else:
            obs_s_list, obs_l_list = [], []
        # 计算规划起点的s, l
        begin_s_list, begin_l_list = planner_utils.cal_s_l_fun([pred_loc_], local_frenet_path_opt_, s_map)
        """从规划起点进行动态规划"""
        # 计算规划起点的l对s的导数和偏导数
        l_list, _, _, _, l_ds_list, _, l_dds_list = \
            planner_utils.cal_s_l_deri_fun(xy_list=[pred_loc_],
                                           V_xy_list=[vehicle_v_],
                                           a_xy_list=[vehicle_a_],
                                           local_path_xy_opt=local_frenet_path_opt_,
                                           origin_xy=pred_loc_)
        # 从起点开始沿着s进行横向和纵向采样，然后动态规划,相邻点之间依据五次多项式进一步采样，间隔一米
        print("*motion planning time cost:", time.time() - start_time)
        dp_path_s, dp_path_l = motion_plan_path_planning.DP_algorithm(obs_s_list, obs_l_list,
                                                                      plan_start_s=begin_s_list[0],
                                                                      plan_start_l=l_list[0],
                                                                      plan_start_dl=l_ds_list[0],
                                                                      plan_start_ddl=l_dds_list[0])
        print("**dp planning time cost:", time.time() - start_time)
        # 对动态规划得到的路径进行降采样，减少二次规划的计算量，然后二次规划完成后再插值填充恢复
        l_min, l_max = motion_plan_path_planning.cal_lmin_lmax(dp_path_s=dp_path_s, dp_path_l=dp_path_l,
                                                               obs_s_list=obs_s_list, obs_l_list=obs_l_list,
                                                               obs_length=5, obs_width=4)  # 这一步的延迟很低，忽略不计
        # print("obs s, l", obs_s_list, obs_l_list)
        # print("l_min", l_min)
        # print("l_max", l_max)
        # print("the length of quadratic planning:", len(dp_path_l))
        # 二次规划变量过多会导致计算延迟比较高，需要平衡二者之间的关系
        print("l_min_max_length", len(l_min))
        """二次规划"""
        qp_path_l, qp_path_dl, qp_path_ddl = motion_plan_path_planning.Quadratic_planning(l_min, l_max,
                                                                                          plan_start_l=l_list[0],
                                                                                          plan_start_dl=l_ds_list[0],
                                                                                          plan_start_ddl=l_dds_list[0])
        print("**qp planning time cost:", time.time() - start_time)
        path_s, path_l = dp_path_s, qp_path_l
        current_local_frenet_path_opt = motion_plan_path_planning.frenet_2_x_y_theta_kappa(
            plan_start_s=begin_s_list[0],
            plan_start_l=begin_l_list[0],
            enriched_s_list=path_s,
            enriched_l_list=path_l,
            frenet_path_opt=local_frenet_path_opt_,
            s_map=s_map)
        # 将重新规划得到的路径信息发送给主进程，让控制器进行轨迹跟踪
        conn.send((current_local_frenet_path_opt, match_point_list_, path_s, path_l))
        print("***motion planning time cost:", time.time() - start_time)


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
    print(model3_spawn_point)
    # model3_spawn_point.location = model3_spawn_point.location + carla.Location(x=-100, y=0, z=0)
    model3_actor = world.spawn_actor(model3_bp, model3_spawn_point)  # type: carla.Vehicle
    # 定义轮胎特性
    # wheel_f = carla.WheelPhysicsControl()  # type: carla.WheelPhysicsControl
    # 定义车辆特性
    physics_control = carla.VehiclePhysicsControl()  # type: carla.VehiclePhysicsControl
    physics_control.mass = 1412  # 质量kg
    model3_actor.apply_physics_control(physics_control)
    """为车辆配备一个障碍物的传感器"""
    obs_detector = Obstacle_detector(ego_vehicle=model3_actor, world=world)  # 实例化传感器
    obs_detector.create_sensor()  # 在仿真环境中生成传感器

    """设置静止车辆"""
    # # 静止车辆1
    obs_vehicle_bp1 = world.get_blueprint_library().find('vehicle.tesla.model3')
    obs_vehicle_bp1.set_attribute('color', '0,0,255')
    obs_spawn_point1 = carla.Transform()
    obs_spawn_point1.location = carla.Location(x=189.31, y=76.61, z=0.3)
    obs_spawn_point1.rotation = model3_spawn_point.rotation
    obs_actor1 = world.spawn_actor(obs_vehicle_bp1, obs_spawn_point1)  # type: carla.Vehicle
    #
    # # 静止车辆2
    obs_vehicle_bp2 = world.get_blueprint_library().find('vehicle.audi.a2')
    obs_vehicle_bp2.set_attribute('color', '0,255,0')
    obs_spawn_point2 = carla.Transform()
    obs_spawn_point2.location = carla.Location(x=196.31, y=58.61, z=0.3)
    obs_spawn_point2.rotation = model3_spawn_point.rotation
    obs_actor2 = world.spawn_actor(obs_vehicle_bp2, obs_spawn_point2)  # type: carla.Vehicle

    # 静止车辆3
    obs_vehicle_bp3 = world.get_blueprint_library().find('vehicle.dodge.charger_police')
    obs_vehicle_bp3.set_attribute('color', '255,0,0')
    obs_spawn_point3 = carla.Transform()
    obs_spawn_point3.location = carla.Location(x=185.01, y=76.61, z=0.3)
    obs_spawn_point3.rotation = model3_spawn_point.rotation
    obs_actor3 = world.spawn_actor(obs_vehicle_bp3, obs_spawn_point3)  # type: carla.Vehicle

    # obs_vehicle_bpx = world.get_blueprint_library().find('vehicle.dodge.charger_police')
    # obs_vehicle_bpx.set_attribute('color', '255,0,0')
    # obs_spawn_pointx = carla.Transform()
    # obs_spawn_pointx.location = carla.Location(x=192.01, y=76.61+18, z=0.3)
    # obs_spawn_pointx.rotation = model3_spawn_point.rotation
    # obs_actorx = world.spawn_actor(obs_vehicle_bpx, obs_spawn_pointx)  # type: carla.Vehicle

    # 静止车辆4
    obs_vehicle_bp4 = world.get_blueprint_library().find('vehicle.toyota.prius')
    obs_vehicle_bp4.set_attribute('color', '255,255,0')
    obs_spawn_point4 = carla.Transform()
    obs_spawn_point4.location = carla.Location(x=204.01, y=64.61, z=0.3)
    obs_spawn_point4.rotation = model3_spawn_point.rotation
    obs_actor4 = world.spawn_actor(obs_vehicle_bp4, obs_spawn_point4)  # type: carla.Vehicle

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

    # 2. 将路径点构成的路径转换为【(x, y, theta, kappa], ...】的形式
    global_frenet_path = planner_utils.waypoint_list_2_target_path(pathway)

    # 3.提取局部路径
    transform = model3_actor.get_transform()
    vehicle_loc = transform.location  # 获取车辆的当前位置
    match_point_list, _ = planner_utils.find_match_points(xy_list=[(vehicle_loc.x, vehicle_loc.y)],
                                                          frenet_path_node_list=global_frenet_path,
                                                          is_first_run=True,  # 寻找车辆起点的匹配点就属于第一次运行，
                                                          pre_match_index=0)  # 没有上一次运行得到的索引，索引自然是全局路径的起点
    local_frenet_path = planner_utils.sampling(match_point_list[0], global_frenet_path)
    local_frenet_path_opt = utils.smooth_reference_line(local_frenet_path)
    # 计算参考线的s, l
    cur_s_map = planner_utils.cal_s_map_fun(local_frenet_path_opt, origin_xy=(vehicle_loc.x, vehicle_loc.y))
    cur_path_s, cur_path_l = planner_utils.cal_s_l_fun(local_frenet_path_opt, local_frenet_path_opt, cur_s_map)

    """整车参数设定"""
    vehicle_para = (1.015, 2.910 - 1.015, 1412, -148970, -82204, 1537)
    controller = "LQR_controller"
    # controller = "MPC_controller"
    Controller = Vehicle_control(ego_vehicle=model3_actor, vehicle_para=vehicle_para,
                                 pathway=local_frenet_path_opt,
                                 controller_type=controller)  # 实例化控制器
    DIS = math.sqrt((pathway[0][0].transform.location.x - pathway[1][0].transform.location.x) ** 2
                    + (pathway[0][0].transform.location.y - pathway[1][0].transform.location.y) ** 2)  # 计算轨迹相邻点之间的距离
    print("The distance between two adjacent points in route:", DIS)
    direction = []
    speed = []
    target_speed = []
    max_speed = 50  # 初始速度设为50km/h，
    # 设定一个观察者视角
    spectator = world.get_spectator()
    count = 1  # 控制规划器和控制器相对频率
    main_process_start_time = time.time()
    planning_count = 100
    pred_ts = 0.2
    # planning_count = 250
    # pred_ts = 0.4
    while True:
        """获取交通速度标志,考虑道路速度限制"""
        # 获取车辆位置信息（包括坐标和姿态信息）， get the transformation, a combination of location and rotation
        transform = model3_actor.get_transform()
        # 不断更新观测视角的位置， update the position of spectator
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40), carla.Rotation(pitch=-90)))
        vehicle_loc = transform.location  # 获取车辆的当前位置

        """获取局部路径，局部路径规划的频率是控制的1/10"""
        if count % planning_count == 0:  # 这里表示控制器执行一千规划器执行一次
            print("main_process_cost_time", time.time() - main_process_start_time)
            main_process_start_time = time.time()
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
            pred_x, pred_y, pred_fi = planner_utils.predict_block(model3_actor, ts=pred_ts)
            # 基于frenet坐标系预测ts秒过后车辆的位置，以预测点作为规划起点
            # pred_x, pred_y = planner_utiles.predict_block_based_on_frenet(vehicle_loc, vehicle_v,
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
            possible_obs = get_actor_from_world(model3_actor, dis_limitation=50)
            # 提取障碍物的位置信息
            obs_xy = []
            for obs_v, dis in possible_obs:
                obs_loc = obs_v.get_transform().location
                obs_xy.append((obs_loc.x, obs_loc.y, dis))
                print("**********", obs_v.type_id, dis)
            # 将当前的道路状况和车辆信息发送给规划器进行规划控制
            conn1.send((obs_xy, (vehicle_loc.x, vehicle_loc.y), (pred_x, pred_y),
                        (vehicle_v.x, vehicle_v.y), (vehicle_a.x, vehicle_a.y),
                        global_frenet_path, match_point_list))

            if count != planning_count:  # 第一个循环周期，因为有初始阶段规划好的局部路径，第二个周期的规划还未计算完成，一旦执行接收数据，会阻塞主进程
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
                                         controller_type=controller)  # 实例化控制器

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
        debug.draw_point(carla.Location(Controller.Lat_control.x_pre, Controller.Lat_control.y_pre, 2),
                         size=0.05, color=carla.Color(0, 255, 255), life_time=0)
        debug.draw_point(carla.Location(Controller.Lat_control.x_pro, Controller.Lat_control.y_pro, 2),
                         size=0.05, color=carla.Color(100, 0, 0), life_time=0)

        """距离判断，程序终止条件"""
        count += 1
        # 计算当前车辆和终点的距离, calculate the distance between vehicle and destination
        dist = vehicle_loc.distance(pathway[-1][0].transform.location)
        # print("The distance to the destination: ", dist)

        if dist < 2:  # 到达终点后产生制动信号让车辆停止运动
            control = emergence_brake()
            model3_actor.apply_control(control)
            print("last waypoint reached")
            p1.terminate()
            break

    """可视化速度变化和航向变化"""
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(direction)
    plt.ylim(bottom=-5, top=5)

    plt.figure(2)
    plt.plot(speed)
    plt.plot(target_speed, color="r")
    plt.ylim(bottom=0, top=max(target_speed) + 10)
    plt.show()
