import carla
import collections
import math
import numpy as np

"""
Different functionalities such as navigation, measurement of physical properties and 2D/3D point maps of the scene.
Retrieve data every simulation step.
"""

print_flag = False


class Radar_sensor(object):
    def __init__(self, parent_actor):
        self.radar_sensor = None
        self._parent = parent_actor
        self.radar_data = None

    def create_radar(self):
        """
        create a radar
        """
        world = self._parent.world  # type: carla.World
        radar_bp = world.get_blueprint_library().find("sensor.other.radar")  # type: carla.ActorBlueprint
        radar_bp.set_attribute("horizontal_fov", "45")
        radar_bp.set_attribute("points_per_second", "1000")
        radar_bp.set_attribute("range", "99")
        radar_bp.set_attribute("sensor_tick", "0.05")
        radar_transform = carla.Transform(carla.Location(x=-2.5, z=2.8), carla.Rotation(pitch=5))
        self.radar_sensor = world.spawn_actor(radar_bp, radar_transform, attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.radar_sensor)
        self.radar_sensor.listen(lambda data: self.radar_callback(data))

    def radar_callback(self, data: carla.RadarMeasurement):
        """
        radar call back method
        :param data: the output of radar sensor,a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        """
        # print("radar capture one group points", data.frame)

        # 将数据以流的形式读入转化成ndarray对象，参数一是原始数据缓存区，第二个是要转化的数据类型
        points = np.frombuffer(buffer=data.raw_data, dtype=np.dtype("f4"))
        self.radar_data = np.reshape(points, (len(data), 4))  # 改变数据维度，使得每一行包含一个点的完整信息【速度，高度，方位角，深度（即物体与传感器的距离）】
        # self._parent.sensor_data_queue.put((data.frame, points))  # put方法默认阻塞模式，阻塞时间将无限长


class GNSS_sensor(object):
    def __init__(self, parent_sensor):
        self.gnss_sensor = None
        self._parent = parent_sensor
        self.gnss_loc = [0, 0, 0]

    def create_sensor(self):
        world = self._parent.world  # type: carla.World
        # 作用是在绝对位置的基础上添加噪声，从而模拟测量值
        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")  # type: carla.ActorBlueprint
        gnss_bp.set_attribute("noise_alt_bias",  "0")  # Mean parameter in the noise model for altitude.
        gnss_bp.set_attribute("noise_alt_stddev",  "0")  # Standard deviation parameter in the noise model for altitude.
        gnss_bp.set_attribute("noise_lat_bias",  "0")
        gnss_bp.set_attribute("noise_lat_stddev",  "0")
        gnss_bp.set_attribute("noise_lon_bias",  "0")
        gnss_bp.set_attribute("noise_lon_stddev",  "0")
        # gnss_bp.set_attribute("noise_seed",  "0")  # Initializer for a pseudorandom number generator.
        # gnss_bp.set_attribute("sensor_tick",  "0")  # Simulation seconds between sensor captures (ticks).

        self.gnss_sensor = world.spawn_actor(gnss_bp,
                                             carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.gnss_sensor)
        self.gnss_sensor.listen(lambda event: self.call_back(event))

    def call_back(self, event: carla.GnssMeasurement):
        self.gnss_loc = [event.latitude, event.longitude, event.altitude]
        if print_flag:
            print("frame", event.frame, "Cur_loc by GNSS:", self.gnss_loc, "\n")


class IMU_sensor(object):
    def __init__(self, parent_actor):
        self.imu_sensor = None
        self._parent = parent_actor
        self.acc = None
        self.gyr = None
        self.compass = None

    def create_sensor(self):
        world = self._parent.world  # type: carla.World
        imu_bp = world.get_blueprint_library().find("sensor.other.imu")  # type: carla.ActorBlueprint
        # Standard deviation parameter in the noise model for acceleration.
        imu_bp.set_attribute("noise_accel_stddev_x", "0")
        imu_bp.set_attribute("noise_accel_stddev_y", "0")
        imu_bp.set_attribute("noise_accel_stddev_z", "0")
        # Mean parameter in the noise model for the gyroscope
        imu_bp.set_attribute("noise_gyro_bias_x", "0")
        imu_bp.set_attribute("noise_gyro_bias_y", "0")
        imu_bp.set_attribute("noise_gyro_bias_z", "0")
        # Standard deviation parameter in the noise model for the gyroscope
        imu_bp.set_attribute("noise_gyro_stddev_x", "0")
        imu_bp.set_attribute("noise_gyro_stddev_y", "0")
        imu_bp.set_attribute("noise_gyro_stddev_z", "0")
        imu_bp.set_attribute("noise_seed", "0")  # Initializer for a pseudorandom number generator.
        imu_bp.set_attribute("sensor_tick", "0")  # Simulation seconds between sensor captures (ticks).
        self.imu_sensor = world.spawn_actor(imu_bp,
                                            carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.imu_sensor)
        self.imu_sensor.listen(lambda event: self.call_back(event))

    def call_back(self, event: carla.IMUMeasurement):
        self.acc = event.accelerometer
        self.gyr = event.gyroscope
        self.compass = event.compass
        if print_flag:
            print("Acc:", self.acc, "Gyr:", self.gyr, "Compass:", self.compass, "\n")


