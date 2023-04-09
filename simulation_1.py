# -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : simulation_1.py

"""
功能
1.实现了基本要素的配置，以及通过按键切换ego-vehicle的运动控制（自动驾驶和手动控制的切换）
2.理解并实现了同步异步模式的差异
3.各项功能实现了基础的模块化
存在的问题：
1.未实现车辆行驶数据在pygame中的呈现，下一版着重解决这个问题
2.目前的运动控制都是基于作者开发的各种控制模块，还未着手自己实现
3.目前自动驾驶状态下，车辆的跟随状态不是很稳定，会出现急刹车现象，以及遇到路边行人也会出现减速现象
4.车辆在转向的过程中存在提前转向的问题，导致车辆会碾压马路边缘
"""

import carla
import random
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

WIDTH = 720
HEIGHT = 560



print_flag = True


class my_agent():
    def __init__(self):
        self.client = None  # type: carla.Client
        self.world = None  # type: carla.World
        self.map = None  # type: carla.Map
        self.blueprint_lib = None  # type: carla.BlueprintLibrary
        self.ego_vehicle = None  # type: carla.Vehicle
        self.camera = None  # type: carla.Actor
        self.agent = None
        self.rgb_image = None
        self.spectator = None
        self.Auto_mode = False
        self.keys = None
        self.is_synchronous = None

    def print_hi(self, sentence):
        """
        print the greeting sentence
        :return: None
        """
        print(sentence)

    def load_world(self, world_name="Town05"):
        """
        argument:
        world_name: the exist world that we want to use
        :return: None
        """
        # 1.create a client and connect to the server
        self.client = carla.Client("localhost", port=2000)  # type: carla.Client
        self.client.set_timeout(10)

        # 2.load the desired world
        self.world = self.client.load_world(world_name)  # type: carla.World

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
        if is_synchronous:
            # setting the mode of synchronous and time step
            setting = self.world.get_settings()  # type: carla.WorldSettings
            setting.synchronous_mode = True  # synchronous is true
            setting.fixed_delta_seconds = 0.05  # the fixed time interval is 0.00125, 100 Hz
            # this mean that the simulator will take twenty steps (1/0.05) to recreate one second of the simulated world.

            # 由于物理控制要求比较小的时间间隔才能实现准确的控制，因此当仿真环境中的fix_delta_time相对较大时，这时物理控制可能出现较大的偏差
            # 因此需要为仿真环境中的物理控制模块提供最小的控制间隔，从而实现精确控制对象的姿态
            setting.substepping = True  # enable the sub-stepping mode
            setting.max_substep_delta_time = 0.01  # the max sub-step time is 0.002, 20Hz
            setting.max_substeps = 10  # the max sub-steps are 10
            # Note（***）: The condition to be fulfilled is: fixed_delta_seconds <= max_substep_delta_time * max_substeps
            # in real simulation fixed_delta_seconds equals substep_delta_time * substeps
            # 这里说明一下，我们设置的最大子步骤间隔时间和最大子步骤数，知识给定了一个上限。意味着实际仿真环境的一个主线时间步骤最多包含10个子步骤，
            # 当前的配置下，子时间步骤的间隔是0.005，也就是仿真环境每过0.05秒刷新一次，控制模块是每隔0.005秒就实现了一次姿态调整
            # 每个子步骤的时间间隔是主线时间步骤的十分之一，配置过程中一定要满足给定的条件***
            self.world.apply_settings(setting)

    def traffic_manager(self, is_synchronous=False, vehicle_number=50, walker_number=20):
        """
        set other traffic participation
        :param vehicle_number: int, the number of vehicles wanted in the traffic
        :param walker_number: int, the number of walkers wanted in the traffic
        :return: None
        """
        tm = self.client.get_trafficmanager()  # type: carla.TrafficManager
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
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicle_list), len(walkers_list)))

    def generate_vehicle(self, vehicle_type="vehicle.tesla.model3"):
        """
        argument: generate a ego vehicle in the world
        :param vehicle_type: the desired vehicle you want to use
        :return: None
        """
        ego_vehicle_bp = self.blueprint_lib.find(vehicle_type)  # type: carla.ActorBlueprint
        ego_vehicle_bp.set_attribute("color", "255,0,0")  # set its color
        ego_vehicle_bp.set_attribute("role_name", "ego-vehicle")  # appoint a name to the vehicle
        if print_flag:
            print("show the attributes of ego-vehicle:")
            for attr in ego_vehicle_bp:
                print('  - {}'.format(attr))
        spwan_point = random.choice(self.map.get_spawn_points())  # choose a spawn point for this vehicle
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, spwan_point)  # spawn this actor

    def create_camera(self, camera_type="sensor.camera.rgb", image_size=(WIDTH, HEIGHT)):
        """
        create a camera attached to ego-vehicle
        :param camera_type: the type of desired camera, default is "sensor.camera.rgb"
        :param image_size: the size of image camera captures
        :return: None
        """
        camera_bp = self.blueprint_lib.find(camera_type)  # type: carla.ActorBlueprint
        camera_bp.set_attribute("image_size_x", str(image_size[0]))
        camera_bp.set_attribute("image_size_y", str(image_size[1]))
        camera_bp.set_attribute("fov", "90")  # Horizontal field of view in degrees.
        camera_bp.set_attribute("sensor_tick", "0.025")  # capture frequency is 40, 经测试该传感器的最大捕获频率是40Hz

        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)

        self.camera.listen(lambda data: self.camera_callback(data))

    def camera_callback(self, data: carla.Image):
        """
        call back function of rgb camera
        :param data: the raw data the camera captured
        :return: None
        """
        # print(data)
        array = np.frombuffer(data.raw_data, dtype='uint8')  # transfer byte type to int type
        array = np.reshape(array, newshape=(data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]  # 摄像机的显示模式是RGB, carla 仿真环境的图像色彩是BGR模式，因此这里需要一个转换
        print("Capture one frame image:", array.shape)
        self.rgb_image = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def update_spectator_transform(self, v_trans: carla.Transform):
        """
        update the position of spectator for tracking the ego-vehicle
        :param v_trans: ego-vehicle's latest position transform
        :return: None
        """
        transform = carla.Transform(v_trans.location+carla.Location(z=40),
                                    carla.Rotation(pitch=-70, yaw=v_trans.rotation.yaw, roll=v_trans.rotation.roll))
        self.spectator.set_transform(transform)

    def autopilot_mode(self):
        """
        set auto pilot mode
        instance a agent to provide auto pilot function for ego-vehicle
        :return: None
        """
        self.agent = BehaviorAgent(self.ego_vehicle)
        destination = random.choice(self.map.get_spawn_points()).location
        self.agent.set_destination(destination)

    def manual_model(self):
        """
        generate the manual control signal
        :return: control-> the control
        """
        control = carla.VehicleControl()
        if self.keys[K_w] or self.keys[K_UP]:
            print("press W")
            control.throttle = 1  # range (0, 1)
            control.reverse = False  # forward and backward is controlled by reverse parameters
        elif self.keys[K_s] or self.keys[K_DOWN]:
            control.throttle = 1
            control.reverse = True
        else:
            control.throttle = 0

        if self.keys[K_a] or self.keys[K_LEFT]:
            control.steer = -0.5  # range (-1, 1), control the the direction of vehicle
        elif self.keys[K_d] or self.keys[K_RIGHT]:
            control.steer = 0.5
        else:
            control.steer = 0
        # use K_SPACE to control the brake of vehicle # range (0, 1)
        control.brake = 1 if self.keys[K_SPACE] else 0
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

    def main_loop(self):
        """
        run the pragrame
        :return: None
        """
        self.is_synchronous = True
        self.load_world(world_name="Town01")
        self.generate_vehicle()
        self.create_camera()
        self.spectator = self.world.get_spectator()
        # set the synchronous mode after carla world initialization
        self.traffic_manager(vehicle_number=50, walker_number=25)

        self.set_synchronous_mode(is_synchronous=self.is_synchronous)
        pygame.init()  # init pygame
        pygame.font.init()  # init the font in pygame
        screen = pygame.display.set_mode(size=(WIDTH, HEIGHT))  # set the window size or resolution of pygame
        pygame.display.set_caption("This is my autopilot simulation")
        pygame_clock = pygame.time.Clock()  # get the clock of pygame

        while True:
            # update the position of ego-vehicle's spectator
            self.update_spectator_transform(self.ego_vehicle.get_transform())

            # judge the synchronous mode
            if self.is_synchronous:
                setting = self.world.get_settings()  # type: carla.WorldSettings
                world_frequency = 1/setting.fixed_delta_seconds
                print("world frequency:", int(world_frequency))
                self.world.tick()  # 设置为同步模式之后，模拟器的每一次动作均由tick()方法触发,由于同步状态下Carla世界的仿真频率是40Hz
                # 因此两次触发的间隔是固定的, 所以也可以起到控制循环的频率，即保证主循环以40Hz的频率运行
                # pygame_clock.tick(40)也可以起到控制主循环频率的作用，在同步模式下，是冗余的。异步模式下可以采用该方法控制主循环的频率
            else:
                pygame_clock.tick(40)  # control the loop frequency, can match the camera tick frequency
            # 这里控制的是pygame的刷新频率，由于每一次刷新更新一帧图像，
            # 因此如果刷新频率大于传感器的捕获频率，那么会存在反复刷新同一帧数据的情况，消耗不必要的运算量
            # 可以合理匹配刷新频率和捕获频率实现二者同步
            # Note：说明一下，为什么self.world.tick()和pygame_clock.tick(40)分别对仿真环境和pygame动画环境起作用，
            # 二者为什么都能起到控制主循环频率的作用？原因是虽然作用的对象不同，但是他们的原理是相同的，
            # 一次tick完成后要等待设定的时间间隔才会接收下一次状态更新，将二者放到主循环中，无论循环执行多么快，一次结束后都要进行等待，
            # 虽然这样会拖慢程序指令更新速度，但是可以保证较好的时序关系，有助于复杂场景的重复验证（同步模式的优缺点）
            # 异步模式一般不采用控制主循环刷新频率，在程序中采用了是为了在异步条件下匹配相机捕获频率，实现捕获的每一帧都可以显示到pygame

            print("one pygame clock")
            screen.blit(self.rgb_image, dest=(0, 0))
            pygame.display.update()
            pygame.event.pump()  # 让 Pygame 内部自动处理事件
            # pygame.display.flip() 更新整个待显示的Surface对象到屏幕上
            # pygame.display.update() 更新部分内容显示到屏幕上，如果没有参数，则与flip功能相同
            if self.keyboard_control():
                setting = self.world.get_settings()  # type: carla.WorldSettings
                setting.synchronous_mode = False  # synchronous is true
                self.world.apply_settings(setting)
                break


def main():
    agent = my_agent()
    agent.print_hi("welcome to the carla 0.9.12!")
    agent.main_loop()
    print("Simulation is over!")
    pygame.quit()


if __name__ == '__main__':
    main()
