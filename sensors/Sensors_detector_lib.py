import carla
import collections
import math

"""
this lib concludes three sensors: Collision, Lane invasion, Obstacle
Retrieve data when the object they are attached to registers a specific event.
Retrieve data when triggered.
"""
print_flag = True


class Collision_sensor(object):
    def __init__(self, parent_actor):
        self.collision_sensor = None
        self.collision_history = []
        self._parent = parent_actor

    def create_sensor(self):
        world = self._parent.world  # type: carla.World
        sensor_bp = world.get_blueprint_library().find("sensor.other.collision")  # type: carla.ActorBlueprint
        self.collision_sensor = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.sensor_callback(event))

    def get_history(self):
        history = collections.defaultdict(int)
        for (frame, intensity) in self.collision_history:
            history[frame] += intensity
        return history

    def sensor_callback(self, event: carla.CollisionEvent):
        actor_type = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        if print_flag:
            print(event.frame, "Collision target", "---->", actor_type, "intensity:", intensity,  "\n")
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) >= 4000:
            self.collision_history.pop(0)
        """后面可能还要做仿真端显示"""


class Lane_invasion_sensor(object):
    def __init__(self, parent_actor):
        self.lane_invasion = None
        self._parent = parent_actor
        self.lane_invasion_info = None

    def create_sensor(self):
        world = self._parent.world  # type: carla.World
        sensor_bp = world.get_blueprint_library().find("sensor.other.lane_invasion")  # type: carla.ActorBlueprint
        self.lane_invasion = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.lane_invasion)
        self.lane_invasion.listen(lambda event: self.sensor_callback(event))

    def sensor_callback(self, event: carla.LaneInvasionEvent):
        self.lane_invasion_info = set(x.type for x in event.crossed_lane_markings)
        if print_flag:
            print(event.frame, "Lane invasion target", "---->", self.lane_invasion_info, "\n")
        """后面可能还要做仿真端显示"""


class Obstacle_detector_sensor(object):
    """
    原理：In order to anticipate obstacles,
    the sensor creates a capsular shape ahead of the parent vehicle and uses it to check for collisions.
    """
    def __init__(self, parent_actor):
        self.obstacle_detector = None
        self._parent = parent_actor
        self.obstacle_info = None

    def create_sensor(self):
        world = self._parent.world  # type: carla.World
        sensor_bp = world.get_blueprint_library().find("sensor.other.obstacle")  # type: carla.ActorBlueprint
        sensor_bp.set_attribute("distance", "6")   # Distance to trace.
        sensor_bp.set_attribute("hit_radius", "0.5")  # Radius of the trace.
        sensor_bp.set_attribute("only_dynamics", "False")  # If true, the trace will only consider dynamic objects.
        sensor_bp.set_attribute("debug_linetrace", "True")  # If true, the trace will be visible.
        # sensor_bp.set_attribute("sensor_tick", "0.01")  # Simulation seconds between sensor captures (ticks).

        self.obstacle_detector = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.obstacle_detector)
        self.obstacle_detector.listen(lambda event: self.sensor_callback(event))

    def sensor_callback(self, event: carla.ObstacleDetectionEvent):
        self.obstacle_info = [event.frame, event.other_actor, event.distance]
        # if print_flag:
        #     print(event.frame, "Obstacle target", "---->", event.other_actor, "distance is", event.distance, "\n")

        """后面可能还要做仿真端显示"""


class Obstacle_detector(object):
    """
    重写一个障碍传感器构建模块，用于测试规划决策代码，这里和上面的主要区别是实例化的时候传入的是车辆和world
    原理：In order to anticipate obstacles, the sensor creates a capsular shape ahead of the parent vehicle and uses it to check for collisions.
         为了预测障碍物，传感器在主车辆前方创建一个胶囊形状，并使用它来检查碰撞情况
    """
    def __init__(self, ego_vehicle, world):
        self.obstacle_detector = None
        self._parent = ego_vehicle
        self.obstacle_info = None
        self._world = world

    def create_sensor(self):
        world = self._world  # type: carla.World
        sensor_bp = world.get_blueprint_library().find("sensor.other.obstacle")  # type: carla.ActorBlueprint
        sensor_bp.set_attribute("distance", "0")   # Distance to trace.
        sensor_bp.set_attribute("hit_radius", "30")  # Radius of the trace.
        sensor_bp.set_attribute("only_dynamics", "True")  # If true, the trace will only consider dynamic objects.
        sensor_bp.set_attribute("debug_linetrace", "False")  # If true, the trace will be visible.
        # sensor_bp.set_attribute("sensor_tick", "0.01")  # Simulation seconds between sensor captures (ticks).

        self.obstacle_detector = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=self._parent)
        self.obstacle_detector.listen(lambda event: self.sensor_callback(event))

    def sensor_callback(self, event: carla.ObstacleDetectionEvent):
        self.obstacle_info = [event.frame, event.other_actor, event.distance]
        # print(event.frame, "Obstacle target", "---->", event.other_actor, "distance is", event.distance, "\n")

        """后面可能还要做仿真端显示"""

