import carla
import numpy as np
import cv2 as cv


"""
Take a shot of the world from their point of view. 
For cameras that return carla.Image, 
you can use the helper class carla.ColorConverter to modify the image to represent different information.
Retrieve data every simulation step.
"""
print_flag = False

WIDTH = 1080
HEIGHT = 720


class Optical_flow_camera(object):
    def __init__(self, parent_actor):
        self._parent = parent_actor
        self.optical_flow_camera = None
        self.image = None

    def create_camera(self, camera_type="sensor.camera.optical_flow", image_size=(WIDTH, HEIGHT)):
        camera_bp = self._parent.blueprint_lib.find(camera_type)  # type: carla.ActorBlueprint
        camera_bp.set_attribute("image_size_x", str(image_size[0]))
        camera_bp.set_attribute("image_size_y", str(image_size[1]))
        camera_bp.set_attribute("fov", "90")  # Horizontal field of view in degrees.
        camera_bp.set_attribute("sensor_tick", "0.025")  # capture frequency is 40,同步情况下取决于客户端的tick频率

        camera_transform = carla.Transform(carla.Location(x=2.5, z=2.8), carla.Rotation(pitch=-15))
        self.optical_flow_camera = self._parent.world.spawn_actor(camera_bp, camera_transform,
                                                                  attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.optical_flow_camera)
        self.optical_flow_camera.listen(lambda data: self.camera_callback(data))

    def camera_callback(self, data: carla.Image):
        data = data.get_color_coded_flow()
        array = np.frombuffer(data.raw_data, dtype='uint8')  # transfer byte type to int type
        array = np.reshape(array, newshape=(data.height, data.width, 4))
        array = array[:, :, :3].copy()  # .copy()方法就很奇怪，有时候不用就会报错
        array = array[:, :, ::-1]  # 摄像机的显示模式是RGB, carla 仿真环境的图像色彩是BGR模式，因此这里需要一个转换
        array = array.swapaxes(0, 1)  # pygame显示的模式是（width, height, channel)；
        # 直接捕获的图片和经过opencv处理后的图片数据都是（height, width, channel)形式，所以需要进行维度转换
        # self._parent.sensor_data_queue.put((data.frame, pygame.surfarray.make_surface(array)))
        if print_flag:
            print("Capture one frame semantic image:", array.shape)
        self.image = array


class Semantic_seg_camera(object):
    def __init__(self, parent_actor):
        self.semantic_camera = None
        self._parent = parent_actor
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
        camera_bp.set_attribute("sensor_tick", "0.025")  # capture frequency is 40

        camera_transform = carla.Transform(carla.Location(x=-2.5, z=2.8), carla.Rotation(pitch=-15))
        self.semantic_camera = self._parent.world.spawn_actor(camera_bp, camera_transform,
                                                              attach_to=self._parent.ego_vehicle)
        self._parent.sensor_list.append(self.semantic_camera)

        self.semantic_camera.listen(lambda data: self.camera_callback(data))

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
        if print_flag:
            print("Capture one frame semantic image:", array.shape)
        self.image = array


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
        camera_bp.set_attribute("sensor_tick", "0.025")  # capture frequency is 40, 经测试该传感器的最大捕获频率是40Hz

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
        if print_flag:
            print("Capture one frame rgb image:", array.shape)
        self.image = array

    def load_detection_model(self):
        """load yolov3 model"""  # 加载训练好的车辆识别模型
        config_path = "../yolo module/yolov3.cfg"
        weights_path = "../yolo module/yolov3.weights"
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
