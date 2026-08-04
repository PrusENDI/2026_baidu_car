from cv2 import sort
from smartcar import logger, CountRecord
from car_wrap_2026 import MyCar, kill_other_python
import time
import math

"""
- cylinder_3
- h_jin_zhen_gu
- h_tu_dou
- h_xi_lan_hua
- h_dou_jiao
- h_you_cai
- animal
- h_qin_cai
- cylinder_2
- cylinder_1
- cylinder_set
- h_qing_jiao
- h_fan_qie
- water_l3
- water_l2
- water
- ball_blue
- ball_yellow
- storage
- lable_yellow
- order
- h_mo_gu
- name
- lable_blue
"""


class _ArmCompatibilityAdapter:
    """让 7_17 任务安全使用工作树 ArmController 的返回值和异常语义。"""

    def __init__(self, arm):
        object.__setattr__(self, "_arm", arm)

    def __getattr__(self, name):
        return getattr(self._arm, name)

    def __setattr__(self, name, value):
        if name == "_arm":
            object.__setattr__(self, name, value)
        else:
            setattr(self._arm, name, value)

    @staticmethod
    def _require_success(result, action):
        if result is False:
            raise RuntimeError(f"机械臂动作失败，停止当前任务: {action}")
        return result

    def reset_position(self, *args, **kwargs):
        return self._require_success(
            self._arm.reset_position(*args, **kwargs),
            f"reset_position args={args!r}, kwargs={kwargs!r}",
        )

    def move_x_position(self, *args, **kwargs):
        return self._require_success(
            self._arm.move_x_position(*args, **kwargs),
            f"move_x_position args={args!r}, kwargs={kwargs!r}",
        )

    def move_y_position(self, *args, **kwargs):
        return self._require_success(
            self._arm.move_y_position(*args, **kwargs),
            f"move_y_position args={args!r}, kwargs={kwargs!r}",
        )

    def goto_position(self, *args, **kwargs):
        return self._require_success(
            self._arm.goto_position(*args, **kwargs),
            f"goto_position args={args!r}, kwargs={kwargs!r}",
        )

    def set_arm_angle(self, *args, **kwargs):
        return self._require_success(
            self._arm.set_arm_angle(*args, **kwargs),
            f"set_arm_angle args={args!r}, kwargs={kwargs!r}",
        )

    def set_arm_pose(self, x=None, y=None, arm=None, hand=None):
        """保持 7_17 的直线轴 -> arm -> hand 顺序，并检查每步结果。"""
        if x is not None or y is not None:
            self.goto_position(x=x, y=y)
        if arm is not None:
            self.set_arm_angle(arm)
            time.sleep(1)
        if hand is not None:
            self._arm.set_hand_angle(hand)
        return True


def _install_arm_compatibility(car):
    car.arm = _ArmCompatibilityAdapter(car.arm)
    original_adjust_arm_position = car.adjust_arm_position

    def checked_adjust_arm_position(*args, **kwargs):
        result = original_adjust_arm_position(*args, **kwargs)
        return _ArmCompatibilityAdapter._require_success(
            result,
            f"adjust_arm_position args={args!r}, kwargs={kwargs!r}",
        )

    car.adjust_arm_position = checked_adjust_arm_position


def init():
    time.sleep(1)
    global my_car
    my_car = MyCar()
    _install_arm_compatibility(my_car)
    my_car.STOP_PARAM = False
    my_car.beep()
    time.sleep(1)
    my_car.arm.reset_position()
    my_car.reset_position()  #

def auto_lane_tracing(speed=0.3, dis_hold=0.85):
    my_car.lane_dis_offset(speed=speed, dis_hold=dis_hold)
    print(f"巡线停止的位置：{my_car.get_odometry()}")

def auto_seeding():
    x_length = 0.46  # 基地前方转角的位置，用于计算播种位置
    dis = 0.55  # 转角后第一个播种点的距离
    heading = math.pi / 4  # 车子的方向 45°
    sin45 = math.sin(heading)  # sin45°
    # 正对播种点车子的理论位置
    cylinder_loc = {
        "cylinder_3": [x_length + dis * sin45 - 0.01, dis * sin45 - 0.01, 0.74],
        "cylinder_2": [x_length + (dis + 0.15) * sin45 - 0.01, (dis + 0.15) * sin45 - 0.01, 0.74],
        "cylinder_1": [x_length + (dis + 0.3) * sin45 - 0.01, (dis + 0.3) * sin45 - 0.01, 0.74],
    }
    cylinder_list = ["cylinder_3", "cylinder_2", "cylinder_1"]
    cylinder_set_list = {}

    # 设置机械臂初始状态
    my_car.arm.set_arm_pose(0.0, 0.2, "LEFT", "UP")
    my_car.lane_dis_offset(speed=0.3, dis_hold=0.85)
    time.sleep(0.5)
    print(f"巡线停止的位置：{my_car.get_odometry()}")

    for i in range(3):
        my_car.move_to_position(cylinder_loc[cylinder_list[i]])
        # 对齐目标
        my_car.move_to_detection_target(-0.11,0)
        x, y, z = my_car.get_odometry()
        pose = [x, y, z, my_car.arm.x_get_position()]
        print(f"第{i}个播种位置{pose}")
        cylinder_set_list[cylinder_list[i]] = pose
        my_car.beep()
    print("实际播种位置：")
    print(cylinder_set_list)

    for i in range(3):
        # 开始工作
        # 移动手臂到右侧高处
        my_car.arm.move_y_position(0.2)
        my_car.arm.move_x_position(0.255)
        time.sleep(1)
        my_car.arm.set_arm_pose(arm="RIGHT")
        time.sleep(2)
        # 对齐目标，识别目标
        my_car.move_to_position(cylinder_loc[cylinder_list[i]])
        time.sleep(1)
        my_car.arm.set_arm_pose(arm="RIGHT")#再次确认
        time.sleep(1)
        # 识别目标
        cls_id, label = my_car.move_to_detection_target(0.16,0)
        
        #直到识别到目标
        if not label:
            for i in range(5):
                my_car.arm.set_arm_pose(arm="RIGHT")#再次确认
                time.sleep(0.5)
                cls_id, label = my_car.move_to_detection_target(0.16,0)
                if label:
                    break
        if not label:
            print("未识别到目标，跳过本次抓取")
            continue
        print(f"识别到目标{cls_id}-{label}")
        my_car.beep()
        pose = cylinder_set_list[label]

        # 调整气泵吸嘴对齐目标，补偿摄像头和吸嘴的距离
        my_car.adjust_arm_position()
        time.sleep(0.5)

        # 下降后再吸取目标
        my_car.arm.move_y_position(0.01)
        time.sleep(1)
        my_car.arm.grasp(True)
        time.sleep(2)
        my_car.arm.move_y_position(0.2)
        time.sleep(2)

        # 移动到目标播种处
        my_car.arm.move_x_position(0.01)
        time.sleep(2)
        my_car.arm.set_arm_pose(arm="LEFT")
        time.sleep(0.5)
        my_car.arm.move_x_position(pose[3])
        time.sleep(0.5)
        my_car.arm.set_arm_pose(arm="LEFT")
        time.sleep(0.5)
        #再次确认
        my_car.move_to_position(pose[:3])#这里会移动车，上面可以加一个底盘移动
        time.sleep(0.5)
        my_car.arm.set_arm_pose(arm="LEFT")
        time.sleep(0.5)
        #再次确认
        my_car.adjust_arm_position()#补偿摄像头和吸嘴的距离
        time.sleep(0.5)
        my_car.arm.set_arm_pose(arm="LEFT")
        time.sleep(0.5)
        #再次确认
        # 下降后释放目标，再抬起   
        my_car.arm.move_y_position(0.04)
        time.sleep(0.5)
        my_car.arm.grasp(False)
        time.sleep(0.5)
        my_car.arm.move_y_position(0.2)
        time.sleep(0.5)

    my_car.arm.move_y_position(0.1)
    my_car.arm.set_arm_pose(hand="DOWN")
    my_car.arm.move_x_position(0.15)
    my_car.move_to_position(cylinder_loc[cylinder_list[0]])
    print("播种完成")
    my_car.beep()
    my_car.beep()
    my_car.get_odometry(True)
    my_car.get_distance(True)


# 动物识别阶段使用的机械臂姿态、巡线距离和扫描参数集中放在这里。
# 后续现场标定时只需要调整本字典，不必进入任务函数查找分散的常量。
TARGET_SHOOTING_DETECTION_POSES = {
    # 侧摄像头搜索动物标签时的机械臂姿态。
    "initial_search": {
        "arm": "LEFT",
        "hand": "UP",
        "x": 0.05,
        "y": 0.05,
    },
    # 从当前任务起点进入动物识别区域的巡线参数。
    "task_entry": {
        "speed": 0.30,
        "debug_distance": 0.40,
        "run_distance": 1.45,
    },
    # 进入识别区域后只修正车辆航向，保持当前平面位置不变。
    "heading_correction": {
        "offset_xy": [0.0, 0.0],
    },
    # 四个动物标签按固定间距排列；入口位置直接对应第一个扫描点。
    "animal_scan": {
        "count": 4,
        "speed": 0.30,
        "step_distance": 0.15,
    },
    # delta_y=None 表示识别时不驱动机械臂 X 轴跟随纵向偏差。
    "alignment": {
        "delta_y": None,
    },
    # 独立调试完成后返回的局部里程计原点。
    "debug_return": {
        "position": [0.0, 0.0, 0.0],
    },
}


def target_shooting_detection(debug=False) -> list:
    """依次识别四个动物标签，并按物理射击靶位顺序返回识别结果。"""
    scan_pose = TARGET_SHOOTING_DETECTION_POSES["initial_search"]
    task_entry = TARGET_SHOOTING_DETECTION_POSES["task_entry"]
    heading_correction = TARGET_SHOOTING_DETECTION_POSES["heading_correction"]
    animal_scan = TARGET_SHOOTING_DETECTION_POSES["animal_scan"]
    alignment = TARGET_SHOOTING_DETECTION_POSES["alignment"]
    debug_return = TARGET_SHOOTING_DETECTION_POSES["debug_return"]

    # None 表示本靶位没有获得可信结果。不能把识别失败默认当作有害动物，
    # 否则后续射击任务可能在没有可靠判断时执行射击。
    animal_list = [None] * animal_scan["count"]

    # 先摆好侧摄像头搜索姿态，再根据调试模式选择进入任务区的距离。
    my_car.arm.set_arm_pose(**scan_pose)
    task_distance = (
        task_entry["debug_distance"]
        if debug
        else task_entry["run_distance"]
    )
    my_car.lane_dis_offset(
        speed=task_entry["speed"],
        dis_hold=task_distance,
    )

    # 读取当前航向角并做反向补偿，使后续扫描沿靶位排列方向直行。
    _x, _y, _z = my_car.get_odometry(True)
    my_car.get_distance(True)
    my_car.move_for([
        heading_correction["offset_xy"][0],
        heading_correction["offset_xy"][1],
        -_z,
    ])
    time.sleep(3)

    # 第一个动物位于任务入口，不再额外前移；从第二个动物开始，
    # 每次按固定间距巡线到下一个扫描位置后再执行视觉对齐和分析。
    for i in range(animal_scan["count"]):
        if i > 0:
            my_car.lane_dis_offset(
                speed=animal_scan["speed"],
                dis_hold=animal_scan["step_distance"],
            )
        time.sleep(0.5)
        cls_id, label = my_car.move_to_detection_target(
            delta_y=alignment["delta_y"]
        )
        if label == "animal":
            res, analysis = my_car.animal_image_analysis()
            if res is not None:
                my_car.beep()
                animal_list[i] = res

    # 扫描结束后用双蜂鸣提示任务完成，并保留里程计和测距读取动作。
    time.sleep(0.5)
    my_car.beep()
    my_car.beep()
    my_car.get_odometry(True)
    my_car.get_distance(True)
    if debug:
        my_car.move_to_position(debug_return["position"])

    # 扫描行进方向与物理射击靶位 0 -> 3 的顺序相反，因此只在任务边界
    # 统一反转一次；射击函数接收到的列表始终按物理靶位编号排列。
    target_order_animal_list = list(reversed(animal_list))
    print(f"动物识别最终结果：{target_order_animal_list}")
    return target_order_animal_list


def water_tower_task(debug=True):
    """
    水塔灌溉任务

    :param debug: 是否开启调试模式
    :return: None

    核心逻辑：
    1. 全部6个水方块按最大需求（3个）提前分配给两个水塔， 并提供预对齐位置参数列表
    2. 初始化机械臂搜索姿态
    2. 根据是否为调试模式，设置不同的距离，自动巡航到第一个水塔附近
    3. 利用视觉识别功能，对齐到水塔标，并获取该水塔需要的水方块的数量
    4. 根据该水塔需要的水方块的数量，从预对齐位置参数列表中截取对应数量的位置参数
    5. 遍历每个位置参数，机械臂和车身移动到该位置，搜索水方块，并进行视觉对齐
    6. 吸取并放置该水方块，放置时根据遍历次序调整机械臂高度
    7. 前进到下一个水塔位置进行相同操作，为避免弯角影响，拆分移动为巡航移动+相对坐标移动
    9. 停止点为机械臂正对最后一个水塔位置
    """
    ################################# 常量定义 ##################################

    # 用于搜寻水方块的机械臂位置参数
    ARM_Y  = 0.19                                              # 搜寻水方块时的y坐标（相机高度）设为常量，方便调试时统一修改
    INNER_OFFSET = 0.11                                        # 搜索靠近赛道内圈的水方块相对于车子的x轴偏移
    OUTER_OFFSET = 0.01                                        # 搜索靠近赛道外圈的水方块相对于车子的x轴偏移
    DISTANCE_BETWEEN_SLOTS = 0.28                              # 两个水槽之间的距离

    # 预对齐位置参数列表
    tower1_search_prealign_params = [
        {
            'arm_pose':{
                'x': INNER_OFFSET,
                'y': ARM_Y,
                'arm': "LEFT",
                'hand': "DOWN"
            },
            'car_offset': [0, 0, 0]
        },
        {
            'arm_pose':{
                'x': OUTER_OFFSET,
                'y': ARM_Y,
                'arm': "LEFT",
                'hand': "DOWN"
            },
            'car_offset': [0, 0, 0]
        },
        {
            'arm_pose':{
                'x': INNER_OFFSET,
                'y': ARM_Y,
                'arm': "LEFT",
                'hand': "DOWN"
            },
            'car_offset': [DISTANCE_BETWEEN_SLOTS, 0, 0]
        },
    ]

    tower2_search_prealign_params = [
        {
            'arm_pose':{
                'x': INNER_OFFSET,
                'y': ARM_Y,
                'arm': "LEFT",
                'hand': "DOWN"
            },
            'car_offset': [0, 0, 0]
        },
        {
            'arm_pose':{
                'x': OUTER_OFFSET,
                'y': ARM_Y,
                'arm': "LEFT",
                'hand': "DOWN"
            },
            'car_offset': [0, 0, 0]
        },
        {
            'arm_pose':{
                'x': OUTER_OFFSET,
                'y': ARM_Y,
                'arm': "LEFT",
                'hand': "DOWN"
            },
            'car_offset': [-DISTANCE_BETWEEN_SLOTS, 0, 0]
        },
    ]

    SEARCHING_POSE ={
        'x': 0.01,
        'y': 0.00,
        'arm': "RIGHT",
        'hand': "UP"
    }

    PLACINGING_POSE ={
        'x': 0.01,
        'y': 0.00,
        'arm': "RIGHT",
        'hand': "UP"
    }
    
    ################################# 函数定义 ##################################

    def get_water_block_needs_by_label(label):
        """根据识别到的标签确定水塔需要的水方块数量"""
        water_dict = {
            'water_l1': 1,
            'water_l2': 2,
            'water_l3': 3,
        }
        # 默认如果识别失败或识别结果不在字典中，则认为不需要水方块
        return water_dict.get(label, 0)
    
    def pick_and_place_water_block(pos_params):
        """根据预设的位置参数列表进行水方块的拾取和放置"""
        for i, pos_param in enumerate(pos_params):
            # 记录绝对坐标，搜索获取水方块后会回到这个位置
            loc = my_car.get_odometry(True)
            
            # 根据位置参数，移动到对应的搜寻位置进行预对齐
            x, y, arm, hand = pos_param['arm_pose'].values()
            my_car.arm.set_arm_pose(x=x, y=y, arm=arm, hand=hand)
            car_offset = pos_param['car_offset']
            my_car.move_for(car_offset)
            
            # 对齐视野中心的方块
            my_car.move_to_detection_target(-0.16,0)
            my_car.beep()

            #补偿摄像头和吸嘴的距离
            my_car.adjust_arm_position()
            # 开启气泵，准备拾取水方块
            my_car.arm.grasp(True)
            my_car.arm.move_y_position(0.08)
            
            # 适当抬高放置碰撞
            my_car.arm.move_y_position(0.18)
            my_car.arm.set_arm_pose(arm="LEFT")
            my_car.arm.move_x_position(0.01)
            my_car.arm.set_arm_pose(arm="LEFT")
            time.sleep(1)
            # 设置为放置姿态
            x, y, arm, hand = PLACINGING_POSE.values()
            my_car.arm.set_arm_pose(arm=arm, hand=hand)
            time.sleep(3)
            my_car.arm.set_arm_pose(x=x, y=y+0.035*i)
            
            # 回到记录的绝对坐标位置，确保每次放置水方块时车子的位置都是一致的，从而提高放置的准确性
            my_car.move_to_position(loc)
            
            # 移动到水塔内侧，确保放置位置正确
            my_car.arm.move_x_position(0.17)
            my_car.arm.grasp(False)
            time.sleep(2)
            my_car.arm.move_x_position(0.01)

    ################################# 任务流程 ##################################
    # 预备姿态
    my_car.arm.set_arm_pose(**SEARCHING_POSE)

    # 沿着车道线移动到第一个水塔附近
    # 调试时缩短沿车道线移动的距离，从而快速进入水塔任务核心部分
    distance = 0.6 if debug else 2.0
    my_car.lane_dis_offset(speed=0.2, dis_hold=distance)      

    # 识别需求标志，确定需要拾取的水方块数量
    cls_id, label = my_car.move_to_detection_target(delta_x=0.14, delta_y=None)
    # 根据识别结果确定第一个水塔需要的水方块数量，调试时默认3个都需要
    tower1_need = 3 if debug else get_water_block_needs_by_label(label)
    # 根据第一个水塔的需求数量，从预设的位置参数列表中取对应数量的位置参数进行拾取和放置
    pick_and_place_water_block(tower1_search_prealign_params[:tower1_need])    # 回到预备姿态
    my_car.arm.set_arm_pose(**SEARCHING_POSE)
    
    # 沿着车道线移动到第二个水塔附近
    # 沿车道线移动一小段距离，调整位置和姿态，为移动到第二个水塔做准备
    my_car.lane_dis_offset(speed=0.2, dis_hold= 0.4)
    # [关键点] 直接向前移动到第二个水塔附近，避免前方弯道导致的车身偏转影响识别
    my_car.move_for([0.24, 0, 0])

    # 识别需求标志，确定需要拾取的水方块数量
    cls_id, label = my_car.move_to_detection_target(delta_x=0.14, delta_y=None)  
    # 根据识别结果确定第二个水塔需要的水方块数量，调试时默认3个都需要
    tower2_need = 3 if debug else get_water_block_needs_by_label(label)
    # 根据第二个水塔的需求数量，从预设的位置参数列表中取对应数量的位置参数进行拾取和放置
    pick_and_place_water_block(tower2_search_prealign_params[:tower2_need])
    my_car.arm.move_x_position(0)
    ################################# 调试模式 ##################################
    my_car.move_to_position([0.0, 0.0, 0.0]) if debug else None



# 动物射击阶段使用的机械臂姿态、靶位间距、视觉关联和击倒确认参数。
# 参数集中管理可以避免射击流程中出现难以追踪的现场标定常量。
TARGET_SHOOTING_POSES = {
    # 机械臂先完成方向和末端姿态切换，再移动直线轴，保持安全动作顺序。
    "initial_orientation": {
        "arm": "LEFT",
        "hand": "UP",
    },
    "initial_linear": {
        "x": 0.24,
        "y": 0.02,
    },
    # 入口距离已经包含旧流程中的后退补偿，并使车辆接近物理第 0 靶。
    "task_entry": {
        "speed": 0.30,
        "debug_distance": 0.34,
        "run_distance": 2.64,
    },
    # 四个靶位等间距排列，course_distance 是从第 0 靶到第 3 靶的总距离。
    "targets": {
        "count": 4,
        "step_distance": 0.16,
        "course_distance": 0.48,
        "lane_speed": 0.30,
    },
    "alignment": {
        # 侧摄像头画面中用于射击的目标横向位置。
        "delta_x": 0.36,
        "delta_y": None,
        "sort_y": 0.0,
        # 对齐期间限制目标框横向跳变，防止漏检后误选相邻靶位。
        "max_delta_x_error": 0.20,
        "target_x_direction": "increasing",
        # 物理靶位 0 -> 3 在画面中按 dx 从大到小排列。
        "target_order_direction": "descending",
        # 首次进入射击区时要求四靶连续多帧稳定，随后锁定固定编号。
        "calibration_stable_frames": 3,
        "calibration_timeout": 15.0,
        "calibration_frame_interval": 0.10,
        "calibration_max_dx_shift": 0.12,
        "selected_max_dx_jump": 0.12,
        "alignment_attempts": 3,
        "alignment_timeout": 10.0,
    },
    "shot_confirmation": {
        # 单靶最多重复射击四次，每次射击后等待靶体稳定再检查是否倒下。
        "max_shots_per_target": 4,
        "post_shot_settle_seconds": 5.0,
        "confirm_frames": 3,
        "verification_timeout": 10.0,
        "verification_frame_interval": 0.10,
        "stationary_dx_tolerance": 0.15,
    },
    # 独立调试完成后返回的局部里程计原点。
    "debug_return": {
        "position": [0.0, 0.0, 0.0],
    },
}


def target_shooting(
    animal_list=[1, 1, 1, 0],
    debug=False,
    shooting_delta_x=None,
):  # noqa: E741
    """按识别结果锁定物理靶位，射击有害动物并确认靶体是否倒下。"""
    # 清除上一次任务可能遗留的固定靶号显示，避免本次校准沿用旧映射。
    my_car.clear_shooting_target_display()

    initial_orientation = TARGET_SHOOTING_POSES["initial_orientation"]
    initial_linear = TARGET_SHOOTING_POSES["initial_linear"]
    task_entry = TARGET_SHOOTING_POSES["task_entry"]
    targets = TARGET_SHOOTING_POSES["targets"]
    alignment = TARGET_SHOOTING_POSES["alignment"]
    shot_confirmation = TARGET_SHOOTING_POSES["shot_confirmation"]
    debug_return = TARGET_SHOOTING_POSES["debug_return"]

    # 默认使用现场标定值；调试调用方可以临时覆盖横向对齐位置。
    target_delta_x = (
        alignment["delta_x"]
        if shooting_delta_x is None
        else float(shooting_delta_x)
    )
    if not -1.0 <= target_delta_x <= 1.0:
        raise ValueError(
            "射击图像横向目标必须位于归一化范围 [-1.0, 1.0]: "
            f"shooting_delta_x={target_delta_x}"
        )

    # 将输入结果规范为固定四个靶位。缺失或非法值保持为 None，
    # 后续生成射击计划时会安全跳过，绝不把未知结果当作有害动物。
    target_count = targets["count"]
    if animal_list is None:
        animal_list = [None] * target_count
    animal_results = list(animal_list[:target_count])
    animal_results.extend([None] * (target_count - len(animal_results)))

    def ordered_animal_detections():
        """读取当前动物框，并按固定物理靶位方向排序。"""
        detections = [
            item
            for item in my_car.get_detection_results()
            if item[2] == "animal"
        ]
        detections.sort(
            key=lambda item: item[4],
            reverse=alignment["target_order_direction"] == "descending",
        )
        return detections

    def calibrate_target_slots():
        """等待四个靶框连续稳定，建立物理靶号与视觉顺序的初始映射。"""
        stable_samples = []
        deadline = time.monotonic() + alignment["calibration_timeout"]
        while time.monotonic() < deadline:
            detections = ordered_animal_detections()
            current_dx = [item[4] for item in detections]

            # 数量不等于四时无法建立完整映射，丢弃此前的连续稳定计数。
            if len(detections) != target_count:
                stable_samples = []
            elif stable_samples and any(
                abs(now - previous) > alignment["calibration_max_dx_shift"]
                for now, previous in zip(current_dx, stable_samples[-1])
            ):
                # 靶框位置突变时从当前帧重新计数，防止编号发生跨靶跳转。
                stable_samples = [current_dx]
            else:
                stable_samples.append(current_dx)

            if len(stable_samples) >= alignment["calibration_stable_frames"]:
                locked_dx = [
                    round(
                        sum(sample[index] for sample in stable_samples)
                        / len(stable_samples),
                        3,
                    )
                    for index in range(target_count)
                ]
                return [
                    {
                        "index": index,
                        "animal_result": animal_results[index],
                        "initial_dx": locked_dx[index],
                        "status": "standing",
                    }
                    for index in range(target_count)
                ]
            time.sleep(alignment["calibration_frame_interval"])
        return None

    def wait_for_standing_snapshot(standing_indices):
        """等待画面中的动物框数量与仍站立的物理靶位数量一致。"""
        deadline = time.monotonic() + alignment["alignment_timeout"]
        expected_count = len(standing_indices)
        while time.monotonic() < deadline:
            detections = ordered_animal_detections()
            if len(detections) == expected_count:
                return detections
            time.sleep(alignment["calibration_frame_interval"])
        return None

    def frame_matches_slots(detections, slot_indices, before_map):
        """判断当前帧中的剩余靶框是否仍对应射击前的固定靶位。"""
        if len(detections) != len(slot_indices):
            return False
        tolerance = shot_confirmation["stationary_dx_tolerance"]
        return all(
            abs(detection[4] - before_map[index][4]) <= tolerance
            for index, detection in zip(slot_indices, detections)
        )

    def verify_knockdown(target_index, standing_indices, before_detections):
        """连续检查目标靶是否消失，区分倒靶、仍站立和画面不确定。"""
        before_map = dict(zip(standing_indices, before_detections))
        remaining_indices = [
            index for index in standing_indices if index != target_index
        ]
        absent_count = 0
        present_count = 0
        deadline = time.monotonic() + shot_confirmation["verification_timeout"]

        while time.monotonic() < deadline:
            detections = ordered_animal_detections()
            if frame_matches_slots(detections, remaining_indices, before_map):
                absent_count += 1
                present_count = 0
            elif frame_matches_slots(detections, standing_indices, before_map):
                present_count += 1
                absent_count = 0
            else:
                # 框数或位置无法稳定关联时，两种连续计数都重新开始。
                absent_count = 0
                present_count = 0

            if absent_count >= shot_confirmation["confirm_frames"]:
                return "knocked_down"
            if present_count >= shot_confirmation["confirm_frames"]:
                return "still_standing"
            time.sleep(shot_confirmation["verification_frame_interval"])
        return "verification_ambiguous"

    # 只把可信的有害动物结果 0 加入射击计划。计划中的移动距离是相对
    # 上一个待射击靶位的距离，因此可以跳过无需射击的靶位。
    shot_plan = []
    last_index = -1
    skipped_indices = []
    for idx, value in enumerate(animal_results):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in (0, 1)
        ):
            skipped_indices.append(idx)
            continue
        if value == 0:
            if last_index == -1:
                dist = idx * targets["step_distance"]
            else:
                dist = (idx - last_index) * targets["step_distance"]
            shot_plan.append({"target_index": idx, "move_distance": dist})
            last_index = idx
        else:
            skipped_indices.append(idx)

    # 机械臂先完成方向切换，再移动 X/Y 直线轴，最后巡线进入射击区。
    my_car.arm.set_arm_pose(**initial_orientation)
    my_car.arm.set_arm_pose(**initial_linear)
    task_distance = (
        task_entry["debug_distance"]
        if debug
        else task_entry["run_distance"]
    )
    my_car.lane_dis_offset(
        speed=task_entry["speed"],
        dis_hold=task_distance,
    )

    # 只有存在待射击靶位时才校准四靶映射；没有射击计划时直接完成赛道补偿。
    target_slots = calibrate_target_slots() if shot_plan else []
    standing_indices = list(range(target_count))
    successful_indices = []
    failed_targets = []
    shots_by_target = {index: 0 for index in range(target_count)}
    traveled_plan_distance = 0.0

    if target_slots is not None and shot_plan:
        my_car.set_shooting_target_display(
            standing_indices,
            alignment["target_order_direction"],
        )

    if target_slots is None:
        # 初始四靶映射无法建立时禁止盲目射击，并记录所有计划靶位为失败。
        for item in shot_plan:
            failed_targets.append({
                "target_index": item["target_index"],
                "reason": "calibration_timeout",
                "shots": 0,
            })
    else:
        for plan_item in shot_plan:
            target_index = plan_item["target_index"]
            dis = plan_item["move_distance"]
            if dis > 0:
                my_car.lane_dis_offset(
                    speed=targets["lane_speed"],
                    dis_hold=dis,
                )
            traveled_plan_distance += dis

            target_succeeded = False
            target_failure_reason = None
            while (
                shots_by_target[target_index]
                < shot_confirmation["max_shots_per_target"]
            ):
                before_detections = None
                alignment_failure = None
                fixed_rank = standing_indices.index(target_index)

                # 每次射击前重新对齐固定编号靶位。expected_detection_count 和
                # fixed_order_num 共同阻止相邻靶在漏检后被误认为当前目标。
                for alignment_attempt in range(
                    1, alignment["alignment_attempts"] + 1
                ):
                    cls_id, label = my_car.move_to_detection_target(
                        delta_x=target_delta_x,
                        delta_y=alignment["delta_y"],
                        sort_pos=(target_delta_x, alignment["sort_y"]),
                        label="animal",
                        time_out=alignment["alignment_timeout"],
                        max_delta_x_error=alignment["max_delta_x_error"],
                        target_x_direction=alignment["target_x_direction"],
                        require_alignment=True,
                        fixed_order_num=fixed_rank,
                        fixed_order_direction=alignment[
                            "target_order_direction"
                        ],
                        expected_detection_count=len(standing_indices),
                        max_selected_dx_jump=alignment["selected_max_dx_jump"],
                        target_index=target_index,
                    )
                    diagnosis = getattr(
                        my_car,
                        "last_detection_alignment_status",
                        {"reason": "alignment_timeout"},
                    )
                    if cls_id is not None and label == "animal":
                        before_detections = wait_for_standing_snapshot(
                            standing_indices
                        )
                        if before_detections is not None:
                            break
                        alignment_failure = "association_timeout"
                    else:
                        alignment_failure = diagnosis.get(
                            "reason", "alignment_timeout"
                        )

                if before_detections is None:
                    target_failure_reason = (
                        alignment_failure or "alignment_timeout"
                    )
                    break

                # 射击命令异常不会直接假定靶体状态，仍通过视觉确认结果决定
                # 是否重试；这样可以兼容命令已下发但调用端收到异常的情况。
                shots_by_target[target_index] += 1
                try:
                    my_car.beep()
                    my_car.shooting()
                except Exception:
                    pass

                time.sleep(shot_confirmation["post_shot_settle_seconds"])
                verification = verify_knockdown(
                    target_index,
                    standing_indices,
                    before_detections,
                )
                if verification == "knocked_down":
                    standing_indices.remove(target_index)
                    my_car.set_shooting_target_display(
                        standing_indices,
                        alignment["target_order_direction"],
                    )
                    target_slots[target_index]["status"] = "shot"
                    successful_indices.append(target_index)
                    target_succeeded = True
                    break
                if verification == "verification_ambiguous":
                    target_failure_reason = "verification_ambiguous"
                    break
                # still_standing 会进入下一轮，在达到单靶射击上限前继续尝试。

            if not target_succeeded:
                if target_failure_reason is None:
                    target_failure_reason = "max_shots_exhausted"
                target_slots[target_index]["status"] = "failed"
                failed_targets.append({
                    "target_index": target_index,
                    "reason": target_failure_reason,
                    "shots": shots_by_target[target_index],
                })

    # 无论实际射击了哪些靶位，都补足从第 0 靶到第 3 靶的剩余赛道距离。
    remaining_course_distance = max(
        0.0,
        targets["course_distance"] - traveled_plan_distance,
    )
    if remaining_course_distance > 0:
        my_car.lane_dis_offset(
            speed=targets["lane_speed"],
            dis_hold=remaining_course_distance,
        )

    # 仅保留一次最终结果输出；详细校准、对齐和逐次射击遥测不在本测试版打印。
    final_result = {
        "planned": [item["target_index"] for item in shot_plan],
        "success": successful_indices,
        "failed": failed_targets,
        "skipped": skipped_indices,
        "shots_by_target": shots_by_target,
    }
    print(f"射击最终结果：{final_result}")
    my_car.clear_shooting_target_display()
    if debug:
        my_car.move_to_position(debug_return["position"])


def crop_harvesting(debug=True):
    """
    作物采收
    """

    # 调整机械臂
    my_car.arm.move_y_position(0.2)

    my_car.set_storage(True)  # 抬起储物架。
    my_car.arm.set_arm_pose(arm="LEFT")
    my_car.arm.set_hand_angle("DOWN")
    # 移动到任务位置
    task_distance = 0.70 if debug else 2.3
    my_car.lane_dis_offset(speed=0.3, dis_hold=task_distance)
    my_car.arm.move_y_position(0.17)
    first_label = None
    for i in range(8):
        # 调整机械臂
        # 修改前仅按编码器移动到 x=0.0，且忽略停滞结果；现在每轮抓取前
        # 低速触达物理回收端并重新建立零点，只有寻零成功才允许转到 LEFT。
        # 末端手腕在初始化或上一轮放球后已经保持 DOWN；这里只翻转到 LEFT，
        # 修改前会在翻转完成后重复下发一次 hand="DOWN"。
        my_car.arm.set_arm_pose(arm="LEFT")
        my_car.arm.set_hand_angle("DOWN")

        # 对齐目标
        alignment_timeout = 6.0
        alignment_start = time.time()
        cls_id, label = my_car.move_to_detection_target(delta_x=-0.14, delta_y=-0.06, time_out=alignment_timeout,)
        alignment_elapsed = time.time() - alignment_start
        alignment_timed_out = alignment_elapsed >= alignment_timeout

        if i == 0:
            first_label = label

        print(f"发现第{i + 1}个作物，目标为{label}")
        time.sleep(0.5)

        # 补偿
        my_car.adjust_arm_position()

        # 抓取
        my_car.arm.grasp(True)
        time.sleep(0.3)

        # 抓球高度抬高 5 cm：修改前为 0.045 m，当前调试值为 0.095 m。
        my_car.arm.move_y_position(0.09)  # 吸取
        time.sleep(1)
        my_car.arm.move_y_position(0.20)  # 抬起机械臂
        time.sleep(0.3)

        my_car.arm.set_arm_angle(-113, 40)
        time.sleep(2)

        if label == first_label:  # 黄球在一号位
            target_x = 0.06
        else:
            target_x = 0.0
        
        my_car.arm.move_x_position(target_x)
        time.sleep(0.5)
        my_car.arm.set_arm_angle(-113, 40)
        my_car.arm.set_hand_angle(-35)
        my_car.beep()
        time.sleep(0.5)
        my_car.arm.grasp(False)
        time.sleep(1.5)
        my_car.arm.set_hand_angle("DOWN")
        my_car.arm.set_arm_pose(arm="LEFT")
        my_car.lane_dis_offset(speed=0.2, dis_hold=0.04)
        time.sleep(0.5)

    my_car.set_storage(False)  # 放下存储架
    if debug:
        my_car.move_to_position([0.0, 0.0, 0.0])

    return first_label
    
def sort_and_store(first_label, ball_count_dict={"first_label": 2, "second_label": 2}, debug=True):
    """
    分类存储

    :param ball_count_dict: 包含拾取的球的标签和数量的字典，格式为{'first_label': 4, 'second_label': 4}
    :param debug: 是否开启调试模式
    :return: None

    核心逻辑：
    1. 初始化机械臂搜索姿态
    2. 根据是否为调试模式，设置不同的距离，自动巡航到高球仓位置
    3. 识别球仓上的标签, 判断高球仓放置球的颜色，确定 first_ball_color 以及 second_ball_color
    4. 根据 first_ball_color 和 second_ball_color以及 ball_count_dict 中的球数量，进行球的拾取和放置
    6. 将高球仓的球放完后，向后移动到低球仓的位置继续将剩下的球放完
    7. 停止点为机械臂正对最后一个球座位置
    """

    ################################## 常量定义 ##################################
    SEARCHING_POSE ={
        'x': 0.24,
        'y': 0.05,
        'arm': "LEFT",
        'hand': "UP"
    }

    ARM_PARAMS_FOR_PICKING = {
        "second_label": {
                "x": 0.01,
                "y": 0.14,
                "arm": -108,
                "hand": -5,

        },
        "first_label": {
                "x": 0.005,
                "y": 0.20,
                "arm": -113,
                "hand": -35,
        },
    }

    ARM_PARAMS_FOR_PLACING = [
        {
                "x": 0.23,
                "y": 0.20,
                "arm": "LEFT",
                "hand": "UP",
        },
        {
                "x": 0.23,
                "y": 0.05,
                "arm":"LEFT",
                "hand": "UP",
        },

    ]

    DISTANCE_BETWEEN_BALL_SLOTS = 0.16
       
    ################################## 函数定义 ##################################
    def pick_and_place_ball_by_params(arm_params_for_picking, arm_params_for_placing):
        """根据预设的位置参数进行球的拾取和放置"""
        my_car.arm.grasp(True)
        my_car.arm.move_y_position(0.18)
        my_car.arm.set_arm_pose(arm="RIGHT")
        my_car.arm.set_hand_angle("DOWN")
        time.sleep(1)
        my_car.arm.set_arm_pose(**arm_params_for_picking)
        if arm_params_for_placing == ARM_PARAMS_FOR_PLACING[1]:
            my_car.arm.move_y_position(0.11)
        else:
            my_car.arm.move_y_position(0.15)
        
        time.sleep(2)
        my_car.arm.move_y_position(0.19)
        my_car.arm.set_hand_angle("UP")
        my_car.arm.set_arm_pose(arm="RIGHT")
        time.sleep(1)
        my_car.arm.move_x_position(0.20)
        my_car.arm.set_arm_pose(arm="LEFT")
        time.sleep(1)
        my_car.arm.set_arm_pose(**arm_params_for_placing)
        time.sleep(2)
        my_car.arm.grasp(False)
        time.sleep(1)
    ################################## 任务流程 ##################################
    # 初始化机械臂搜索姿态,并且放下球仓
    my_car.arm.move_y_position(0.05)
    my_car.arm.set_arm_pose(**SEARCHING_POSE)
    time.sleep(1)
    my_car.set_storage(False)

    # 根据是否为调试模式，设置不同的距离，自动巡航到较高球仓位置
    distance = 0.35 if debug else 2.05
    my_car.lane_dis_offset(speed=0.2, dis_hold=distance)
    my_car.move_for([0.20, 0, 0])
    
    cls_id, label = my_car.move_to_detection_target(delta_x=-0.26, delta_y=None)
    print(cls_id, label)

    if first_label == "ball_blue":
        first_label = "lable_blue"
    else:
        first_label = "lable_yellow"

    # 确认抓取姿势
    if label == first_label:
        label_for_first = "first_label"
        label_for_second = "second_label"
    else:
        label_for_first = "second_label"
        label_for_second = "first_label"
    
    # 移动到first_label对应的货舱
    if label != first_label:
        my_car.move_for([-DISTANCE_BETWEEN_BALL_SLOTS, 0, 0])

    arm_params_for_picking = ARM_PARAMS_FOR_PICKING[label_for_first]
    arm_params_for_placing = ARM_PARAMS_FOR_PLACING[1]
    print(arm_params_for_picking, arm_params_for_placing) 

    for _ in range(ball_count_dict[label_for_first]):
        pick_and_place_ball_by_params(arm_params_for_picking, arm_params_for_placing)

    # 移动到second_label对应的货舱
    my_car.arm.set_arm_pose(arm="RIGHT")
    my_car.arm.set_hand_angle("UP")
    if label == first_label:
        my_car.move_for([-DISTANCE_BETWEEN_BALL_SLOTS, 0, 0])
    else:
        my_car.move_for([DISTANCE_BETWEEN_BALL_SLOTS, 0, 0])

    arm_params_for_picking = ARM_PARAMS_FOR_PICKING[label_for_second]
    arm_params_for_placing = ARM_PARAMS_FOR_PLACING[0]
    print(arm_params_for_picking, arm_params_for_placing)
    for _ in range(ball_count_dict[label_for_second]):
        pick_and_place_ball_by_params(arm_params_for_picking, arm_params_for_placing)
    my_car.set_storage(True)

    ################################# 调试模式 ##################################
    my_car.move_to_position([0.0, 0.0, 0.0]) if debug else None

# 寻找货物的程序
def find_goods(label, dy=-0.6):
    time.sleep(1)
    cls_id, det_label = my_car.move_to_detection_target(
        label=label, delta_x=0.22, delta_y=dy, detector=my_car.goods_det
    )
    if det_label is not None:
        logger.info("1找到")
        return det_label

    my_car.arm.move_x_position(0.24)
    time.sleep(1)
    cls_id, det_label = my_car.move_to_detection_target(
        label=label, delta_x=0.22, delta_y=dy, detector=my_car.goods_det
    )
    if det_label is not None:
        logger.info("2找到")
        return det_label

    my_car.move_for([0.14, 0, 0])
    time.sleep(1)
    cls_id, det_label = my_car.move_to_detection_target(
        label=label, delta_x=0.22, delta_y=dy, detector=my_car.goods_det
    )
    if det_label is not None:
        logger.info("3找到")
        return det_label

    my_car.arm.move_x_position(0.15)
    time.sleep(1)
    cls_id, det_label = my_car.move_to_detection_target(
        label=label, delta_x=0.22, delta_y=dy, detector=my_car.goods_det
    )
    if det_label is not None:
        logger.info("4找到")
        return det_label
    else:
        logger.info("未找到")

def get_order():
    goods_dict = {
        "青椒": "h_qing_jiao",
        "蘑菇": "h_mo_gu",
        "芹菜": "h_qin_cai",
        "番茄": "h_fan_qie",
        "油菜": "h_you_cai",
        "豆角": "h_dou_jiao",
        "西兰花": "h_xi_lan_hua",
        "土豆": "h_tu_dou",
        "金针菇": "h_jin_zhen_gu",
    }

    order_list = []
    push_rod_y = 0.01
    goods_pickup_left_offset = 0.00

    # 启动阶段已完成 X 轴寻零，任务内避免重复触碰限位。
    my_car.arm.reset_position(rehome_x=False)
    my_car.lane_dis_offset(speed=0.2, dis_hold=1.5)
    my_car.move_to_detection_target(delta_y=None)

    # 推杆 
    # 保持推杆工作高度，避免水平推动时与机构干涉。
    my_car.arm.move_y_position(push_rod_y)
    my_car.move_for([0.065, 0, 0])
    my_car.arm.move_x_position(0.20)
    my_car.arm.move_x_position(0.03, out_time=4.0)
    time.sleep(0.5)

    # 识别第一个订单
    my_car.move_for([-0.06, 0, 0])
    my_car.move_to_detection_target(delta_y=None)
    time.sleep(0.5)
    order_list.append(my_car.analyze_task_image(task="order", label="order"))
    my_car.beep()

    # 识别固定订单
    my_car.arm.move_y_position(0.2)
    my_car.arm.move_x_position(0.18)
    my_car.arm.set_hand_angle("MID")
    my_car.arm.set_arm_angle("RIGHT")
    time.sleep(1.0)
    my_car.move_to_detection_target(label="order", time_out=5.0)
    time.sleep(1)
    # 固定订单直接使用整帧，避免对齐后下一帧丢失检测框而中断识别。
    order_list.append(
        my_car.analyze_task_image(
            task="order",
            label="order",
            full_frame=True,
        )
    )
    my_car.beep()

    # 移动到赛道货架
    order_list.sort(key=lambda x: x["address"])
    my_car.lane_dis_offset(speed=0.2, dis_hold=0.20)
    my_car.arm.set_hand_angle(angle="DOWN")
    my_car.move_for([0.0, -0.04, 0])
    time.sleep(0.5)
    loc = my_car.get_odometry(True)

    #my_car.set_storage(False)
    my_car.arm.move_y_position(0.16)

    # 从固定起点搜索，底盘位移仅由 find_goods() 控制。
    my_car.arm.move_x_position(0.15)
    goods_now = order_list[1]["goods"]
    find_goods(goods_dict[goods_now])
    time.sleep(0.5)

    # 视觉对齐后回缩，修正吸盘与识别中心的固定偏差。
    #pickup_x = max(0.0, my_car.arm.x_get_position() - goods_pickup_left_offset)
    #my_car.arm.move_x_position(pickup_x)
    my_car.arm.grasp(True)
    my_car.arm.move_y_position(0.05)
    time.sleep(0.5)
    my_car.arm.move_y_position(0.2)
    my_car.arm.move_x_position(0.005)
    my_car.arm.set_arm_angle(-91)
    my_car.arm.move_y_position(0.09)
    time.sleep(0.5)
    my_car.arm.grasp(False)
    my_car.arm.set_arm_angle("RIGHT")

    my_car.move_for([0.0, 0.04, 0])
    time.sleep(0.5)
    my_car.move_to_position(loc)
    my_car.move_for([0.0, -0.04, 0])
    my_car.arm.move_y_position(0.16)
    # 第二件货物从同一固定起点重新搜索。
    my_car.arm.move_x_position(0.18)
    goods_now = order_list[0]["goods"]
    find_goods(goods_dict[goods_now])
    time.sleep(0.5)

    # 视觉对齐后回缩，修正吸盘与识别中心的固定偏差。
    #pickup_x = max(0.0, my_car.arm.x_get_position() - goods_pickup_left_offset)
    #my_car.arm.move_x_position(pickup_x)
    my_car.arm.grasp(True)
    my_car.arm.move_y_position(0.05)
    time.sleep(0.5)
    my_car.arm.move_y_position(0.2)
    my_car.arm.move_x_position(0.005)
    my_car.arm.set_arm_angle(-91)
    # 保留原仓位布局，确保与当前配送取货动作匹配。
    my_car.arm.move_y_position(0.14)
    time.sleep(0.5)
    my_car.arm.grasp(False)
    my_car.arm.set_arm_angle("RIGHT")

    my_car.move_to_position(loc)
    return order_list

# 我们不会用他
def find_name(name="name"):
    name_list = []
    for i in range(3):
        my_car.move_to_detection_target(delta_y=None)
        time.sleep(1)
        dets = my_car.get_detection_results(sort_pos=(0, 0.5), limit_x=0.3)
        for j, det in enumerate(dets):
            text = my_car.get_det_ocr(det)
            print(f'第{i}列第{j}行的姓名：{text}')
            time.sleep(5)
            if text == name:
                return i, j  # i为0 是下层，为上层
        if i < 2:
            my_car.lane_dis_offset(speed=0.3, dis_hold=0.11)


def order_delivery(
        order_list = [
            {"name": "李四", "goods": "芹菜", "address": 2},
            {"name": "钱七", "goods": "青椒", "address": 2},
        ],
        debug=True
     ):
    """
    订单配送

    :param order_list: 订单列表，每个订单包含姓名、商品和地址
    :param debug: 是否开启调试模式
    :return: 配送及返航成功返回True，否则返回False

    核心逻辑：
    1. 初始化机械臂搜索姿态
    2. 根据是否为调试模式，设置不同的距离，自动巡航到第一栋楼附近
    3. 依次遍历2栋楼所有房间
    4. 按照拿取顺序的倒序将货物进行配送
    5. 任务结束后先前一直巡航回到基地
    """    
    ################################## 常量定义 ##################################
    # 搜索时的机械臂姿态参数
    SEARCHING_POSE ={
        'x': 0.20,
        'y': 0.15,
        'arm': "LEFT",
        'hand': "UP"
    }

    # 搜索时的机械臂姿态参数
    ARM_X_FOR_SEARCHING = 0.22            # 搜索时的机械臂x轴位置
    ARM_Y_FOR_SEARCHING_UPPER = 0.14      # 搜索时的机械臂y轴位置（上层）
    ARM_Y_FOR_SEARCHING_LOWER = 0.05      # 搜索时的机械臂y轴位置（下层）
    ROOM_WIDTH = 0.12                     # 房间的宽度

    # 搜索时的机械臂姿态参数，用于遍历2栋楼的所有房间
    PARAMS_FOR_SEARCHING_BUILDING = [

        {
            'arm_x':ARM_X_FOR_SEARCHING,
            'arm_y':ARM_Y_FOR_SEARCHING_UPPER,
            'offset_x': 0.0
        },
        {
            'arm_x':ARM_X_FOR_SEARCHING,
            'arm_y':ARM_Y_FOR_SEARCHING_UPPER,
            'offset_x': ROOM_WIDTH
        },
        {
            'arm_x':ARM_X_FOR_SEARCHING,
            'arm_y':ARM_Y_FOR_SEARCHING_UPPER,
            'offset_x': ROOM_WIDTH
        },
        {
            'arm_x':ARM_X_FOR_SEARCHING,
            'arm_y':ARM_Y_FOR_SEARCHING_LOWER,
            'offset_x': 0.0
        },
        
        {
            'arm_x':ARM_X_FOR_SEARCHING,
            'arm_y':ARM_Y_FOR_SEARCHING_LOWER,
            'offset_x': -ROOM_WIDTH
        },
        {
            'arm_x':ARM_X_FOR_SEARCHING,
            'arm_y':ARM_Y_FOR_SEARCHING_LOWER,
            'offset_x': -ROOM_WIDTH
        },
    ]

    ################################## 函数定义 ##################################
    def grasp_and_place_good(is_first_good_picked, floor):
        """拾取商品并放置"""
        my_car.arm.grasp(True)
        my_car.arm.move_y_position(0.17)
        my_car.arm.set_arm_pose(arm="RIGHT", hand="DOWN")
        my_car.arm.set_arm_pose(x=0.005, y=0.09  if is_first_good_picked else 0.14)
        my_car.arm.move_y_position(0.18 if floor == "upper" else 0.10)
        my_car.arm.move_x_position(0.22)
        my_car.arm.set_arm_pose(arm="LEFT", hand="UP")
        my_car.arm.move_y_position(0.12)
        my_car.arm.move_y_position(0.08)
        my_car.arm.set_arm_pose(arm="LEFT", hand="UP")
        my_car.arm.grasp(False)

    def search_building_by_names(names):
        # 初始化状态变量，用于控制搜索过程
        first_name = names[0]           # 第一个商品的收货人姓名
        second_name = names[1]          # 第二个商品的收货人姓名
        logger.info(f"first_name: {first_name}, second_name: {second_name}")

        is_first_good_placed = False    # 第一个商品是否已经配送
        is_second_good_placed = False   # 第二个商品是否已经配送

        second_good_loc = None          # 第二个商品的位置
        second_good_arm_y = None        # 第二个商品的机械臂y轴位置
        second_good_floor = None        # 第二个商品所在楼层

        # 遍历2栋楼的所有房间
        for building_id in range(2):
            if my_car._stop_flag:
                return False
            if is_first_good_placed and is_second_good_placed:
                return True
            # 如果遍历索引值为1，说明当前是第二栋楼，需要向前移动到第二栋楼的位置
            if building_id == 1 and not my_car.move_for([0.56, 0, 0]):
                logger.error("移动到第二栋楼失败")
                return False
            # 遍历当前栋楼的所有房间
            for i, params in enumerate(PARAMS_FOR_SEARCHING_BUILDING):
                if my_car._stop_flag:
                    return False
                arm_x, arm_y = params["arm_x"], params["arm_y"]
                my_car.arm.set_arm_pose(x=arm_x, y=arm_y)
                offset_x = params["offset_x"]
                if not my_car.move_for([offset_x, 0, 0]):
                    logger.error(f"移动到房间失败 building={building_id + 1}, room={i + 1}")
                    return False
                _, aligned_label = my_car.move_to_detection_target(
                    delta_x=None,
                    delta_y=None,
                    label="name",
                )
                if my_car._stop_flag:
                    return False
                if aligned_label == "name":
                    name = my_car.get_ocr(label="name")
                else:
                    logger.warning(
                        f"姓名标签检测失败，改用整帧OCR building={building_id + 1}, room={i + 1}"
                    )
                    name = my_car.get_ocr(full_frame=True)
                if my_car._stop_flag:
                    return False
                if name is None:
                    logger.warning(
                        f"裁框OCR失败，改用整帧OCR building={building_id + 1}, room={i + 1}"
                    )
                    name = my_car.get_ocr(full_frame=True)
                logger.info(f"name: {name}")
                if name is None:
                    logger.warning(f"姓名识别失败 building={building_id + 1}, room={i + 1}")
                    continue

                # 处理第一个要配送的货物
                # 直接配送第一个商品，然后检查是否有第二个商品的检索位置，如果有则配送第二个商品
                if name == first_name:
                    logger.info("人1")
                    floor = "upper" if i//3 == 0 else "lower"
                    grasp_and_place_good(is_first_good_placed, floor)
                    is_first_good_placed = True

                    if second_good_loc is not None:
                        if not my_car.move_to_position(second_good_loc):
                            logger.error("返回第二位收货人位置失败")
                            return False
                        grasp_and_place_good(is_second_good_placed, second_good_floor)
                        is_second_good_placed = True
                        return True
                    continue
                
                # 处理第二个要配送的货物
                # 如果第一个已经配送了，则直接配送第二个商品，否则记录第二个商品的检索位置
                if name == second_name:
                    logger.info("人2")
                    floor = "upper" if i//3 == 0 else "lower"
                    if is_first_good_placed:
                        grasp_and_place_good(is_second_good_placed, floor)
                        is_second_good_placed = True
                        if is_first_good_placed and is_second_good_placed:
                            return True
                    else:
                        second_good_loc = my_car.get_odometry(True)
                        second_good_floor = floor
                        continue
        logger.error(
            f"配送未完成 first={is_first_good_placed}, second={is_second_good_placed}"
        )
        return False
    ################################## 任务流程 ##################################
    my_car.arm.set_arm_pose(**SEARCHING_POSE)

    # 根据是否为调试模式，设置不同的距离，自动巡航到第一栋楼附近
    distance = 0.60 if debug else 2.30
    my_car.lane_dis_offset(speed=0.2, dis_hold=distance )

    names = [order["name"] for order in order_list]
    if len(names) < 2:
        logger.error("订单配送至少需要两条订单")
        return False
    if my_car._stop_flag:
        my_car.stop()
        return False
    if not search_building_by_names(names):
        my_car.stop()
        logger.error("订单配送失败，停止后续返航动作")
        return False

    ################################# 调试模式 ##################################
    if my_car._stop_flag:
        my_car.stop()
        return False
    if debug:
        if not my_car.move_to_position([0.0, 0.0, 0.0]):
            logger.error("配送完成后返回原点失败")
            return False
    else:
        my_car.lane_dis_offset(speed=0.3, dis_hold=4.0)
    return not my_car._stop_flag
