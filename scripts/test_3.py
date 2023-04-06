#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : test_3.py

"""
控制器测试模块
基于规划的路点级别的全局路径，采用LQR进行横向控制,PID进行纵向控制
"""

import carla
import math

from planner import utils
from controller.controller import Vehicle_control
from planner.global_planning import global_path_planner

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
# 定义车辆特性，这里暂时没有使用具体参数，后面引入carsim的时候会用
physics_control = carla.VehiclePhysicsControl()  # type: carla.VehiclePhysicsControl
physics_control.mass = 1412  # 质量kg
model3_actor.apply_physics_control(physics_control)

# # 运动车辆
obs_vehicle_bp1 = world.get_blueprint_library().find('vehicle.tesla.model3')
obs_vehicle_bp1.set_attribute('color', '0,0,255')
obs_spawn_point1 = carla.Transform()
obs_spawn_point1.location = carla.Location(x=189.31, y=76.61, z=0.3)
obs_spawn_point1.rotation = model3_spawn_point.rotation
obs_actor1 = world.spawn_actor(obs_vehicle_bp1, obs_spawn_point1)  # type: carla.Vehicle

# 规划路径
pathway = global_route_plan.search_path_way(origin=model3_spawn_point.location,
                                            destination=All_spawn_points[12].location)
debug = world.debug  # type: carla.DebugHelper
i = 0
for waypoint in pathway:
    # print(waypoint)
    debug.draw_point(waypoint[0].transform.location + carla.Location(0, 0, 2),
                     size=0.05, color=carla.Color(0, 255, 0), life_time=0)
    # mark = str((i,
    #             round(waypoint[0].transform.location.x, 2),
    #             round(waypoint[0].transform.location.y, 2),
    #             round(waypoint[0].transform.location.z, 2)))
    # world.debug.draw_string(waypoint[0].transform.location + carla.Location(0, 0, 2), mark, draw_shadow=False,
    #                         color=carla.Color(r=0, g=0, b=255), life_time=1000,
    #                         persistent_lines=True)
    # i += 1

# vehicle_para = (1.015, 2.910-1.015, 1412, -110000, -110000, 1537)
vehicle_para = (1.015, 2.910-1.015, 1412, -148970, -82204, 1537)
# Lat_controller = Lateral_LQR_controller(ego_vehicle=model3_actor, vehicle_para=vehicle_para, pathway=pathway)
global_frenet_path = utils.waypoint_list_2_target_path(pathway)
Controller = Vehicle_control(ego_vehicle=model3_actor, vehicle_para=vehicle_para, pathway=global_frenet_path,
                             # controller_type="LQR_controller")  # 实例化控制器
                             controller_type="MPC_controller")  # 实例化控制器

DIS = math.sqrt((pathway[0][0].transform.location.x - pathway[1][0].transform.location.x) ** 2
                + (pathway[0][0].transform.location.y - pathway[1][0].transform.location.y) ** 2)  # 计算轨迹相邻点之间的距离
print("The distance between two adjacent points in route:", DIS)
direction = []
speed = []
target_speed = []
max_speed = 60  # 初始速度设为50km/h

while True:
    # 设定一个观察者视角
    spectator = world.get_spectator()
    # 获取车辆位置信息（包括坐标和姿态信息）， get the transformation, a combination of location and rotation
    transform = model3_actor.get_transform()
    # 不断更新观测视角的位置， update the position of spectator
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40), carla.Rotation(pitch=-90)))

    vehicle_loc = transform.location  # 获取车辆的当前位置

    """获取交通速度标志,考虑道路速度限制"""
    # wp = amap.get_waypoint(vehicle_loc)  # type: carla.Waypoint
    # Landmark = wp.get_landmarks(distance=5, stop_at_junction=True)
    # for lm in Landmark:
    #     traffic_sign = world.get_traffic_sign(lm)  # type: carla.TrafficSign
    #     t_sign = traffic_sign.type_id.split(".")
    #     if 'speed_limit' in t_sign:
    #         max_speed = float(t_sign[-1])
    # 说明:车辆的特殊控制信号只在一定条件下产生##########################

    control = Controller.run_step(target_speed=max_speed)  # 实例化的时候已经将必要的信息传递给规划器，这里告知目标速度即可
    direction.append(model3_actor.get_transform().rotation.yaw*(math.pi/180))
    V = model3_actor.get_velocity()  # 利用 carla API to 获取速度矢量， use the API of carla to get the velocity vector
    V_len = 3.6 * math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)  # transfer m/s to km/h
    speed.append(V_len)
    target_speed.append(max_speed)
    # 将预测点和投影点的位置标出来, mark the predicted point and project point in the simulation world for debug
    debug.draw_point(carla.Location(Controller.Lat_control.x_pre, Controller.Lat_control.y_pre, 2),
                     size=0.05, color=carla.Color(0, 255, 255), life_time=0)
    debug.draw_point(carla.Location(Controller.Lat_control.x_pro, Controller.Lat_control.y_pro, 2),
                     size=0.05, color=carla.Color(100, 0, 0), life_time=0)
    model3_actor.apply_control(control)  # 执行最终控制指令, execute the final control signal
    # 计算当前车辆和终点的距离, calculate the distance between vehicle and destination
    dist = vehicle_loc.distance(pathway[-1][0].transform.location)
    print("The distance to the destination: ", dist)
    if dist < 2:  # 到达终点后产生制动信号让车辆停止运动
        control = emergence_brake()
        model3_actor.apply_control(control)
        print("last waypoint reached")
        break


# 可视化速度变化和航向变化
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(direction)
plt.ylim(bottom=-5, top=5)

plt.figure(2)
plt.plot(speed)
plt.plot(target_speed, color="r")
plt.ylim(bottom=0, top=max(target_speed) + 10)
plt.show()



