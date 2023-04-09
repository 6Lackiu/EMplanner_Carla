# -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : simulation_2.py

"""
目标：在simulation_1的基础上实现同步功能,并且将不同的功能初步模块化
"""

import collections
import math
import os
import datetime
import time
import weakref

import carla
import random
from queue import Queue
from queue import Empty
import cv2 as cv
# import tensorflow as tf
from agents.navigation.behavior_agent import BehaviorAgent

try:
    import pygame
    from pygame.locals import KEYDOWN, K_ESCAPE, K_SPACE, K_w, K_s, K_a, K_d, K_q
    from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT
except ImportError:
    """
    # we can also use codes to install package automatically
    os.system('pip install pygame==2.0.1')
    import pygame
    from pygame.locals import K_ESCAPE, K_SPACE, K_w, K_s, K_a, K_d, K_q
    """
    # raise the import error is more general ways, because different users may have different environment
    # therefore there package version they want may be differnt
    raise RuntimeError("cannot import pygame, please make sure the pygame package is installed!")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, please make sure the numpy package is installed!")

WIDTH = 1080
HEIGHT = 720
print_flag = True


class Collision_sensor(object):
    def __init__(self, parent_actor):
        self.collision_sensor = None
        self.collision_history = []
        self._parent = parent_actor   # type: World

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
        print(event.frame, "Collision target", "---->", actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) >= 4000:
            self.collision_history.pop(0)


class Radar_sensor(object):
    def __init__(self, parent_actor):
        self.radar_sensor = None
        self._parent = parent_actor  # type: World

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
        points = np.reshape(points, (len(data), 4))  # 改变数据维度，使得每一行包含一个点的完整信息【速度，高度，方位角，深度（即物体与传感器的距离）】
        # self._parent.sensor_data_queue.put((data.frame, points))  # put方法默认阻塞模式，阻塞时间将无限长


class RGB_camera(object):
    def __init__(self, parent_actor):
        self.rgb_camera = None
        self._parent = parent_actor  # type: World
        # the following variables is for detection model
        self.car_detect_model = None  # type: cv.dnn_Net
        self.class_names = None
        self.image = None

    def create_camera(self, camera_type="sensor.camera.rgb", image_size=(WIDTH, HEIGHT)):
        """
        create a camera attached to ego-vehicle
        :param camera_type: the type of desired camera, default is "sensor.camera.rgb"
        :param image_size: the size of image camera captures
        :return: None
        """
        camera_bp = self._parent.blueprint_lib.find(camera_type)  # type: carla.ActorBlueprint
        camera_bp.set_attribute("image_size_x", str(image_size[0]))
        camera_bp.set_attribute("image_size_y", str(image_size[1]))
        camera_bp.set_attribute("fov", "90")  # Horizontal field of view in degrees.
        # camera_bp.set_attribute("sensor_tick", "0.025")  # capture frequency is 40, 经测试该传感器的最大捕获频率是40Hz

        camera_transform = carla.Transform(carla.Location(x=-2.5, z=2.8), carla.Rotation(pitch=-15))
        self.rgb_camera = self._parent.world.spawn_actor(camera_bp, camera_transform,
                                                         attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.rgb_camera)

        self.rgb_camera.listen(lambda data: self.camera_callback(data))

    def camera_callback(self, data: carla.Image):
        """
        call back function of rgb camera
        :param data: the raw data the camera captured
        :return: None
        """
        # print("camera capture one frame image:", data.frame)
        array = np.frombuffer(data.raw_data, dtype='uint8')  # transfer byte type to int type
        array = np.reshape(array, newshape=(data.height, data.width, 4))
        array = array[:, :, :3].copy()  # .copy()方法就很奇怪，有时候不用就会报错
        # array = self.draw_box(array)  # use yolo-v3 to detect and mark the objects,
        # 采用的这个yolo版本(YOLOv3-416),用Python进行处理速度较慢，后面可以考虑用C++直接进行处理
        array = array[:, :, ::-1]  # 摄像机的显示模式是RGB, carla 仿真环境的图像色彩是BGR模式，因此这里需要一个转换
        array = array.swapaxes(0, 1)  # pygame显示的模式是（width, height, channel)；
        # 直接捕获的图片和经过opencv处理后的图片数据都是（height, width, channel)形式，所以需要进行维度转换
        # self._parent.sensor_data_queue.put((data.frame, pygame.surfarray.make_surface(array)))
        # print("Capture one frame image:", array.shape)
        self.image = pygame.surfarray.make_surface(array)

    def load_detection_model(self):
        """load yolov3 model"""  # 加载训练好的车辆识别模型
        config_path = "yolo module/yolov3.cfg"
        weights_path = "yolo module/yolov3.weights"
        self.car_detect_model = cv.dnn.readNetFromDarknet(config_path, weights_path)  # type: cv.dnn_Net
        self.car_detect_model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.car_detect_model.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

        class_namefile = "coco.names"
        with open(class_namefile, "rt") as f:
            self.class_names = f.read().rstrip("\n").split("\n")

    def yolo_detection(self, array, confidence_thre=0.5):
        """
        use trained yolo3 algorithm to detect the target in the captured image
        :param array:
        :param confidence_thre:
        :return:
        """
        blob = cv.dnn.blobFromImage(image=array, scalefactor=1 / 255,
                                    size=(416, 416), mean=[0, 0, 0],
                                    swapRB=1, crop=False)
        self.car_detect_model.setInput(blob)
        layernames = self.car_detect_model.getLayerNames()
        # print(layernames)
        # yolov3有三层输出，分别检测不同尺度的对象
        outputNames = [layernames[i - 1] for i in self.car_detect_model.getUnconnectedOutLayers()]
        outputs = self.car_detect_model.forward(outputNames)
        # yolov3做了三个采样，实现了对更小对象的检测，三次下采样其实就是将像素划分多少个网格
        # yolov2只是将图片做了一次划分，检测不是很准确，从三个维度进行划分，增加了计算量，提高了准确性
        # print(len(outputs), outputs[0].shape)  # 507*85， 13*13*3*85
        # print(len(outputs), outputs[1].shape)  # 2028*85， 26*26*3*85
        # print(len(outputs), outputs[2].shape)  # 8112*85， 52*52*3*85
        hT, wT, cT = array.shape
        bounding_boxes = []
        class_IDs = []
        confidences = []
        for output in outputs:
            boxes_xy = output[:, 0:2] * np.array([[wT, hT]])  # 将中心点的比例尺度转化在图片上具体位置，现在是浮点数，后面画图的时候需要转化为整形
            boxes_wh = output[:, 2:4] * np.array([[wT, hT]])  # 将宽高也进行转化
            boxes_xy = boxes_xy - boxes_wh / 2  # 求出左上角的坐标值，还是浮点的
            Possibility = output[:, 5:]  # 获取分类的概率

            class_ids = np.argmax(Possibility, axis=1).tolist()  # 找到每行（对应一个预测）的最大概率的索引并转化为列表形式

            class_scores = np.max(Possibility, axis=1)  # 找到每个预测的最大概率
            confs = (output[:, 4] * class_scores).tolist()  # 计算置信度，存在物体的概率值*具体物体的概率值
            b_boxes = np.hstack((boxes_xy, boxes_wh)).tolist()  # 将坐标值整合到一起再转化为列表，每个元素是[x,y,w,h]
            bounding_boxes += b_boxes
            confidences += confs
            class_IDs += class_ids
        # 每一个输入都是list形式
        # 输入的bbox元素是中心点和宽高[x,y,w,h]
        indices = cv.dnn.NMSBoxes(bboxes=bounding_boxes, scores=confidences, score_threshold=confidence_thre,
                                  nms_threshold=0.3)
        print(indices)
        return bounding_boxes, confidences, class_IDs, indices

    def draw_box(self, image_array):
        """
        opencv处理数据的时候不关心RGB色彩的排列方式，输入图片的形式是一般的形式，即（height, width, channel)
        :param image_array: input image data
        :return:
        """
        self.load_detection_model()  # load the model
        boxes, scores, classes, indices = self.yolo_detection(image_array)
        for i in indices:
            [x, y, w, h] = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
            text = '{} {:.2f}'.format(self.class_names[classes[i]], scores[i])
            cv.rectangle(image_array, (x, y), (x + w, y + h), (255, 0, 255), 1)
            cv.putText(image_array, text=text, org=(x, y - 10),  # 图片处理的过程中左上角是原点位置
                       fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=4 * (w / WIDTH), color=(255, 0, 0), thickness=1)

        return image_array


class Semantic_seg_camera(object):
    def __init__(self, parent_actor):
        self.Semantic_camera = None
        self._parent = parent_actor  # type: World
        # the following variables is for detection model
        self.image = None

    def create_camera(self, camera_type="sensor.camera.semantic_segmentation", image_size=(WIDTH, HEIGHT)):
        """
        create a camera attached to ego-vehicle
        :param camera_type: the type of desired is "sensor.camera.semantic_segmentation"
        :param image_size: the size of image camera captures
        :return: None
        """
        camera_bp = self._parent.blueprint_lib.find(camera_type)  # type: carla.ActorBlueprint
        camera_bp.set_attribute("image_size_x", str(image_size[0]))
        camera_bp.set_attribute("image_size_y", str(image_size[1]))
        camera_bp.set_attribute("fov", "90")  # Horizontal field of view in degrees.
        camera_bp.set_attribute("sensor_tick", "0.025")  # capture frequency is 40, 经测试该传感器的最大捕获频率是40Hz

        camera_transform = carla.Transform(carla.Location(x=-2.5, z=2.8), carla.Rotation(pitch=-15))
        self.Semantic_camera = self._parent.world.spawn_actor(camera_bp, camera_transform,
                                                              attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.Semantic_camera)

        self.Semantic_camera.listen(lambda data: self.camera_callback(data))

    def camera_callback(self, data: carla.Image):
        """
        call back function of rgb camera
        :param data: the raw data the camera captured
        :return: None
        """
        # print("camera capture one frame image:", data.frame)
        data.convert(carla.ColorConverter.CityScapesPalette)  # 将采集的数据按照规定的标准转化成对应的图片
        array = np.frombuffer(data.raw_data, dtype='uint8')  # transfer byte type to int type
        array = np.reshape(array, newshape=(data.height, data.width, 4))
        array = array[:, :, :3].copy()  # .copy()方法就很奇怪，有时候不用就会报错
        array = array[:, :, ::-1]  # 摄像机的显示模式是RGB, carla 仿真环境的图像色彩是BGR模式，因此这里需要一个转换
        array = array.swapaxes(0, 1)  # pygame显示的模式是（width, height, channel)；
        # 直接捕获的图片和经过opencv处理后的图片数据都是（height, width, channel)形式，所以需要进行维度转换
        # self._parent.sensor_data_queue.put((data.frame, pygame.surfarray.make_surface(array)))
        print("Capture one frame image:", array.shape)
        self.image = pygame.surfarray.make_surface(array)


class World(object):
    """
    design my own agent to learn self driving
    """

    def __init__(self, display):
        self.client = None  # type: carla.Client
        self.world = None  # type: carla.World
        self.map = None  # type: carla.Map
        self.blueprint_lib = None  # type: carla.BlueprintLibrary
        self.ego_vehicle = None  # type: carla.Vehicle
        self.agent = None
        self.spectator = None
        self.Auto_mode = False
        self.keys = None
        self.is_synchronous = None
        self.screen = display
        self.pygame_clock = None  # type: pygame.time.Clock
        self.__info_text = []
        # sensors
        self.rgb_camera = None
        self.semantic_camera = None
        self.radar_sensor = None
        self.collision_sensor = None
        # self.sensor_data_queue = Queue(maxsize=2)
        self.sensor_list = []
        # time_information
        self.sim_timestamp = 0
        self.real_timestamp = None

        # init world
        self.Init_world()

    def print_hi(self, sentence):
        """
        print the greeting sentence
        :return: None
        """
        print(sentence)

    def load_world(self, world_name=None):
        """
        argument:
        world_name: the exist world that we want to use
        :return: None
        """
        # 1.create a client and connect to the server
        self.client = carla.Client("localhost", port=2000)  # type: carla.Client
        self.client.set_timeout(10)

        # 2.load the desired world
        if world_name:
            self.world = self.client.load_world(world_name)  # type: # carla.World
        else:
            self.world = self.client.get_world()  # type: carla.World

        # 3.instantiate the map
        self.map = self.world.get_map()

        # 4. get the blueprint library
        self.blueprint_lib = self.world.get_blueprint_library()

    def set_synchronous_mode(self, is_synchronous=False):
        """
        set synchronous mode for the our simulation
        is_synchronous: True or False, True is set the simulation as synchronous mode
        default mode is asynchronous
        :return: None
        """
        # setting the mode of synchronous and time step
        setting = self.world.get_settings()  # type: carla.WorldSettings

        if is_synchronous:
            setting.synchronous_mode = True  # synchronous is true

        setting.fixed_delta_seconds = 0.05  # the fixed time interval is 0.05, 20 Hz
        # this mean that the simulator will take twenty steps (1/0.5) to recreate one second of the simulated world.

        # 由于物理控制要求比较小的时间间隔才能实现准确的控制，因此当仿真环境中的fix_delta_time相对较大时，这时物理控制可能出现较大的偏差
        # # 因此需要为仿真环境中的物理控制模块提供最小的控制间隔，从而实现精确控制对象的姿态
        # setting.substepping = True  # enable the sub-stepping mode
        # setting.max_substep_delta_time = 0.01  # the max sub-step time is 0.002, 20Hz
        # setting.max_substeps = 5  # the max sub-steps are 10
        # Note（***）: The condition to be fulfilled is: fixed_delta_seconds <= max_substep_delta_time * max_substeps
        # in real simulation fixed_delta_seconds equals substep_delta_time * substeps
        # 这里说明一下，我们设置的最大子步骤间隔时间和最大子步骤数，知识给定了一个上限。意味着实际仿真环境的一个主线时间步骤最多包含10个子步骤，
        # 当前的配置下，子时间步骤的间隔是0.005，也就是仿真环境每过0.05秒刷新一次，控制模块是每隔0.005秒就实现了一次姿态调整
        # 每个子步骤的时间间隔是主线时间步骤的十分之一，配置过程中一定要满足给定的条件***
        self.world.apply_settings(setting)

    def traffic_manager(self, is_synchronous=False, vehicle_number=100, walker_number=100):
        """
        set other traffic participants
        :param vehicle_number: int, the number of vehicles wanted in the traffic
        :param walker_number: int, the number of walkers wanted in the traffic
        :return: None
        """
        tm = self.client.get_trafficmanager()  # type: carla.TrafficManager
        tm.set_global_distance_to_leading_vehicle(3.0)
        tm.set_hybrid_physics_mode(True)  # 将交通管理器中的车辆设定为混合模式，混合模式下车辆运动学控制分为两种，一种是复杂的物理控制（考虑加速，摩擦所有的控制元素）
        # 一种是简单的位置控制，主要模拟车辆的移动特性。
        tm.set_hybrid_physics_radius(20)  # 设定混合控制的作用范围，以hero(可以将任何车辆通过属性修改，将其名字设定为hero)车辆为中心,半径范围内的车辆考虑物理控制
        # 范围外的车辆物理控制失灵，有利于观察目标车辆接触其他车辆时的物理行为
        tm.global_percentage_speed_difference(-50)
        if is_synchronous:  # keep tm's synchronous mode the same with carla world
            tm.set_synchronous_mode(True)
        tm.set_random_device_seed(99)
        tm_port = tm.get_port()

        """
        Spawn Vehicles and set vehicle's behaviour
        """
        vehicle_lib = self.blueprint_lib.filter("vehicle.*.*")
        random.seed(100)
        vehicle_list = []
        spawn_points = self.map.get_spawn_points()
        for _ in range(vehicle_number):  # generate the blueprint of vehicles
            try:
                vehicle_bp = random.choice(vehicle_lib)
                vehicle = self.world.spawn_actor(vehicle_bp, random.choice(spawn_points))
                vehicle_list.append(vehicle)
            except:
                print("vehicle's spawn point collision")

        vehicle = None  # type: carla.Vehicle
        for vehicle in vehicle_list:
            vehicle.set_autopilot(True, tm_port)

        """
        Spawn Walkers and set walker's behaviour
        """
        # 1. set some necessary parameters
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
        # 2. we spawn the walker object
        # walker has no autopilot mode, we can use AI-controller to control their behaviour
        walker_batch = []
        walker_speed = []
        walker_lib = self.blueprint_lib.filter("walker.pedestrian.*")
        for _ in range(walker_number):
            try:
                # choose a spawn point for each walker and generate actor
                walker_spawn_point = carla.Transform()
                location = self.world.get_random_location_from_navigation()
                walker_spawn_point.location = location
                walker_bp = random.choice(walker_lib)  # type: carla.ActorBlueprint
                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    if (random.random() > percentagePedestriansRunning):
                        # walking
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                    else:
                        # running
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                walker = carla.command.SpawnActor(walker_bp, walker_spawn_point)
                walker_batch.append(walker)
            except:
                print("walker's spawn points collision")
        if is_synchronous:
            results = self.client.apply_batch_sync(walker_batch, True)  # set synchronous mode
        else:
            results = self.client.apply_batch_sync(walker_batch, False)
        # 3. we spawn the walker controller
        walker_speed2 = []
        # 由于每个walker都要有一个控制器进行控制，AI_walker控制器也是一个actor,也会有一个id
        # walkers_list 的每一个元素都是一个字典类型，对应的是{"id": walker id, "con": AI_walker id}
        walkers_list = []
        all_id = []
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        AI_walker_batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            AI_walker_batch.append(carla.command.SpawnActor(walker_controller_bp,
                                                            carla.Transform(),
                                                            walkers_list[i]["id"]))
        if is_synchronous:
            results = self.client.apply_batch_sync(AI_walker_batch, True)
        else:
            results = self.client.apply_batch_sync(AI_walker_batch, False)

        for i in range(len(results)):
            if results[i].error:  #
                print(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id  # 将控制器的的信息加入到对应的walker对象的字典里
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)
        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print('spawned %d vehicles and %d walkers.' % (len(vehicle_list), len(walkers_list)))

    def generate_vehicle(self, vehicle_type="vehicle.tesla.model3"):
        """
        argument: generate a ego vehicle in the world
        :param vehicle_type: the desired vehicle you want to use
        :return: None
        """
        ego_vehicle_bp = self.blueprint_lib.find(vehicle_type)  # type: carla.ActorBlueprint
        ego_vehicle_bp.set_attribute("color", "255,0,0")  # set its color
        ego_vehicle_bp.set_attribute("role_name", "hero")  # appoint a name to the vehicle
        if print_flag:
            print("show the attributes of ego-vehicle:")
            for attr in ego_vehicle_bp:
                print('  - {}'.format(attr))
        spwan_point = random.choice(self.map.get_spawn_points())  # choose a spawn point for this vehicle
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, spwan_point)  # spawn this actor

    def update_spectator_transform(self, v_trans: carla.Transform):  # 更新主视角位置
        """
        update the position of spectator for tracking the ego-vehicle
        :param v_trans: ego-vehicle's latest position transform
        :return: None
        """
        transform = carla.Transform(v_trans.location + carla.Location(z=40),
                                    carla.Rotation(pitch=-70, yaw=v_trans.rotation.yaw, roll=v_trans.rotation.roll))
        self.spectator.set_transform(transform)

    def autopilot_mode(self):
        """
        set auto pilot mode
        instance a agent to provide auto pilot function for ego-vehicle
        :return: None
        """
        self.agent = BehaviorAgent(self.ego_vehicle, "aggressive")
        destination = random.choice(self.map.get_spawn_points()).location
        self.agent.set_destination(destination)
        self.agent.set_target_speed(80.0)

    def manual_model(self):
        """
        generate the manual control signal
        :return: control-> the control
        """
        control = self.ego_vehicle.get_control()
        if self.keys[K_w] or self.keys[K_UP]:
            print("press W")
            control.throttle = min(1, control.throttle + 0.05)  # range (0, 1)
            control.reverse = False  # forward and backward is controlled by reverse parameters
        elif self.keys[K_s] or self.keys[K_DOWN]:
            control.throttle = min(1, control.throttle + 0.05)
            control.reverse = True
        else:
            print("************************************")
            control.throttle = 0

        if self.keys[K_a] or self.keys[K_LEFT]:
            control.steer = max(-1, min(control.steer - 0.05, 0))  # range (-1, 1), control the the direction of vehicle
        elif self.keys[K_d] or self.keys[K_RIGHT]:
            control.steer = min(1, max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        # use K_SPACE to control the brake of vehicle # range (0, 1)
        control.brake = min(1, control.brake + 0.1) if self.keys[K_SPACE] else 0
        # set manual gear shift as False
        control.manual_gear_shift = False
        return control

    def keyboard_control(self):
        """
        capture the external inputs from pygame to control the ego-vehicle in carla world
        :return:True or False, this
        """
        for event in pygame.event.get():
            print("event.type", event.type)
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    print("press escape")
                    return True
                print("pygame.K_q", K_q)
                if event.key == pygame.K_q:
                    print("press q key")
                    self.Auto_mode = not self.Auto_mode  # reverse the control mode
                    print(self.Auto_mode)
                    if self.Auto_mode:
                        self.autopilot_mode()
        print("current driving mode:", self.Auto_mode)
        if self.Auto_mode:
            if self.agent.done():
                # if the current task is done, we can choose another random destination again for ego-vehicle
                self.agent.set_destination(random.choice(self.map.get_spawn_points()).location)
                print("The target has been reached, searching for another target")
            self.ego_vehicle.apply_control(self.agent.run_step())
        else:
            self.keys = pygame.key.get_pressed()  # get the state of all keyboard buttons
            self.ego_vehicle.apply_control(self.manual_model())
        return False

    def show_infomation(self):
        """
        show the basic information in pygame
        """
        V_location = self.ego_vehicle.get_location()  # type: carla.Location
        V_speed = self.ego_vehicle.get_velocity()
        V_acc = self.ego_vehicle.get_acceleration()
        settings = self.world.get_settings()
        server_fps = 1 / settings.fixed_delta_seconds

        control_info = self.ego_vehicle.get_control()  # type: carla.VehicleControl
        # get vehicles in the world
        vehicles = self.world.get_actors().filter('vehicle.*')
        self.__info_text = [
            # 每个字段字符长度32，保证字符对齐
            "Server:       % 16.0f FPS" % server_fps,
            "Client:       % 16.0f FPS" % self.pygame_clock.get_fps(),
            "Sim_time:     % 20s" % datetime.timedelta(seconds=int(self.sim_timestamp)),
            "Real_time:    % 20s" % datetime.timedelta(seconds=int(self.real_timestamp)),
            " ",
            "Map:          % 20s" % self.map.name.split("/")[-1],
            "Vehicle:      % 20s" % self.ego_vehicle.type_id,
            " ",
            "Vehicle State>>",
            "Location_xyz:     (%4.1f, %4.1f, %4.1f)" % (V_location.x, V_location.y, V_location.z),
            "Acceleration_xyz: (%3.1f, %3.1f, 0)m/s^2" % (V_acc.x, V_acc.y),
            "Velocity_xyz:     (%3.1f, %3.1f, 0)m/s" % (V_speed.x, V_speed.y),
            "Velocity:              % 5.1f Km/h" % (3.6 * math.sqrt(V_speed.x ** 2 + V_speed.y ** 2)),
            "",
            "Control State>>",
            "Throttle:     % 20.2f" % control_info.throttle,
            "Steer:        % 20.2f" % control_info.steer,
            "Brake:        % 20.2f" % control_info.brake,
            "Reverse:      % 20s" % control_info.reverse,
            "Gear:         % 20d" % control_info.gear
        ]

        # add the nearby vehicle's info
        if len(vehicles) >= 2:
            self.__info_text += [" "]
            self.__info_text += ["Nearby Vehicles>>"]
            distance = lambda loc: math.sqrt((V_location.x - loc.x) ** 2 +
                                             (V_location.y - loc.y) ** 2 +
                                             (V_location.z - loc.z) ** 2)
            velocity = lambda v_xyz: 3.6 * math.sqrt(v_xyz.x ** 2 + v_xyz.y ** 2 + v_xyz.z ** 2)
            vehicles = [(distance(vehicle.get_location()), vehicle.type_id, velocity(vehicle.get_velocity()))
                        for vehicle in vehicles if vehicle.id != self.ego_vehicle.id]
            vehicles = sorted(vehicles, key=lambda vehicle: vehicle[0], reverse=False)  # 按距离的升序排序
            for dis, vehicle, veloc in vehicles:
                if dis > 200:
                    continue
                self.__info_text += ["%s%3.1f m  %3.1f km/h" % (vehicle.strip()[8:].ljust(20), dis, veloc)]
        # show the info
        bar_width = int(WIDTH / 3)
        info_surface = pygame.Surface((bar_width + 40, HEIGHT))  # 默认生成纯黑背景
        info_surface.set_alpha(40)  # 设置透明度，值越低越透明
        self.screen.blit(info_surface, (0, 0))  # 将文字背景渲染到屏幕上

        # 该变量返回当前操作系统的类型，当前只注册了3个值：分别是posix, nt, java，
        # 对应linux / windows / java虚拟机
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]  # choose target font
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]  # 如果默认字体不在选择的字体中，取选择的字体中的第一个
        mono = pygame.font.match_font(mono)
        # 这里的22是一个经验值，目的是使字体能跟着屏幕大小自适应， scale value
        font = pygame.font.Font(mono, int(bar_width / 21) if os.name == 'nt' else int(bar_width / 21) + 2)
        v_offset = 4

        for item in self.__info_text:
            # render(文字内容, True, 文字颜色， 背景颜色（可以不设）)
            # 第二个True类似是文字是否平滑之类的，固定就是True了
            text = font.render(item, True, (255, 255, 255))
            self.screen.blit(text, (8, v_offset))
            v_offset += int(HEIGHT / 40)

    def Init_world(self):
        self.print_hi("welcome to the carla 0.9.12!")
        "initialize the virtual environment"
        self.is_synchronous = True  # set the synchronous mode
        # self.load_world(world_name="Town01")

        self.load_world(world_name="Town05")
        'set the synchronous mode after carla world initialization'
        self.set_synchronous_mode(is_synchronous=self.is_synchronous)
        self.generate_vehicle()
        # initialize the sensors
        self.radar_sensor = Radar_sensor(self)
        self.radar_sensor.create_radar()
        self.rgb_camera = RGB_camera(self)
        self.rgb_camera.create_camera()
        self.semantic_camera = Semantic_seg_camera(self)
        self.semantic_camera.create_camera()
        self.collision_sensor = Collision_sensor(self)
        self.collision_sensor.create_sensor()

        # 获取车辆视角
        self.spectator = self.world.get_spectator()

        "generate other traffic participants"
        self.traffic_manager(is_synchronous=self.is_synchronous, vehicle_number=25, walker_number=50)

        self.pygame_clock = pygame.time.Clock()  # get the clock of pygame

    def destroy(self):
        for sensor in self.sensor_list:
            # sensor.stop()
            sensor.destroy()

        self.ego_vehicle.destroy()


def main_loop():
    """
    run the pragrame
    :return: None
    """
    # create an animation window to observe our ego-vehicle's behaviour.
    pygame.init()  # init pygame
    pygame.font.init()  # init the font in pygame
    display = pygame.display.set_mode(size=(WIDTH, HEIGHT))  # set the window size or resolution of pygame
    pygame.display.set_caption("This is my autopilot simulation")
    display.fill((1, 1, 1))
    pygame.display.flip()
    # initiate the simulation world
    my_world = World(display)
    time_begin = time.time()
    print("Begin my simulation!")
    while True:
        # update the position of ego-vehicle's spectator
        my_world.update_spectator_transform(my_world.ego_vehicle.get_transform())
        # judge the synchronous mode
        if my_world.is_synchronous:

            my_world.world.tick()  # 设置为同步模式之后，模拟器的每一次动作均由tick()方法触发,参数为最大等待时间
            my_world.pygame_clock.tick_busy_loop(60)  #
            print("One tick in simulation world!")
            snapshot = my_world.world.get_snapshot()  # type: carla.WorldSnapshot
            world_frequency = snapshot.timestamp.delta_seconds
            print("world frequency:", int(1/world_frequency))
            # 获取当前的模拟场景基本信息，帧编号，帧编号，当前场景时间戳
            print(snapshot.id, snapshot.frame, snapshot.timestamp)
            my_world.sim_timestamp = snapshot.timestamp.elapsed_seconds

        else:
            pass_time = my_world.pygame_clock.tick(
                20)  # control the loop frequency, can match the camera tick frequency
            print(my_world.pygame_clock.get_fps())
            print("passed time:", pass_time)
            pass

        my_world.screen.blit(my_world.semantic_camera.image, dest=(0, 0))
        my_world.real_timestamp = time.time() - time_begin
        my_world.show_infomation()
        pygame.display.update()
        pygame.event.pump()  # 让 Pygame 内部自动处理事件
        # pygame.display.flip() 更新整个待显示的Surface对象到屏幕上
        # pygame.display.update() 更新部分内容显示到屏幕上，如果没有参数，则与flip功能相同
        if my_world.keyboard_control():
            setting = my_world.world.get_settings()  # type: carla.WorldSettings
            setting.synchronous_mode = False  # synchronous is true
            my_world.world.apply_settings(setting)
            my_world.destroy()
            break


def main():
    main_loop()
    print("Simulation is over!")
    pygame.quit()


if __name__ == '__main__':
    main()
