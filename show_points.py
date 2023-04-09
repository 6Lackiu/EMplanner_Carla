#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : show_points.py

"""
显示 waypoints 和 spawn points
"""

import carla

Client = carla.Client("localhost", port=2000)
Client.set_timeout(20)
print(Client.get_available_maps())

World = Client.load_world("Town05")
# World = Client.get_world()  # type: carla.World
Map = World.get_map()  # type: # carla.Map

All_waypoints = Map.generate_waypoints(5)

All_spawn_points = Map.get_spawn_points()

debug = World.debug  # type: carla.DebugHelper
waypoint = None  # type: carla.Waypoint
print("the number of waypoints:", len(All_waypoints))
# for waypoint in All_waypoints:
#     # print(waypoint)
#     debug.draw_point(waypoint.transform.location + carla.Location(0, 0, 2), size=0.05, color=carla.Color(0, 255, 0), life_time=0)

# debug = World.debug  # type: # carla.DebugHelperw
spawn_point = None  # type: carla.Transform
i = 0
for spawn_point in All_spawn_points:
    # print(spawn_point)
    debug.draw_point(spawn_point.location + carla.Location(0, 0, 5), size=0.1, color=carla.Color(255, 0, 0), life_time=0)
    mark = str(i)
    debug.draw_string(spawn_point.location, mark, draw_shadow=False,
                            color=carla.Color(r=0, g=0, b=255), life_time=1000.0,
                            persistent_lines=True)
    i += 1