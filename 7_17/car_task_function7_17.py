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


def init():
    time.sleep(1)
    global my_car
    my_car = MyCar()
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


def target_shooting_detection() -> list:

    animal_list = [0, 0, 0, 0]
    my_car.arm.set_arm_pose(x=0.05, y=0.05, arm="LEFT", hand="UP")
    my_car.lane_dis_offset(speed=0.3, dis_hold=1.45)

    _x, _y, _z = my_car.get_odometry(True)
    my_car.get_distance(True)
    my_car.move_for([0, 0, 0 - _z])
    time.sleep(3)

    for i in range(4):
        my_car.lane_dis_offset(speed=0.3, dis_hold=0.15)
        time.sleep(0.5)
        cls_id, label = my_car.move_to_detection_target(delta_y=None)
        if label == "animal":
            res, analysis = my_car.animal_image_analysis()
            if res is not None:
                my_car.beep()
                print(f"第{i}个动物分析结果：{res}，{analysis}")
                animal_list[i] = res
    time.sleep(0.5)
    my_car.beep()
    my_car.beep()
    my_car.get_odometry(True)
    my_car.get_distance(True)
    return animal_list


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



def target_shooting(animal_list=[0, 0, 0, 0]):  # noqa: E741

    step = 0.16  # 每个目标间距
    relative_loc = []  # 记录相对运动距离
    last_index = -1  # 记录上一个打击点的索引，初始为-1
    d_x = 0.2  # 对齐参数

    for idx, value in enumerate(animal_list):
        if value == 0:  # 遇到需要打击的点
            if last_index == -1:
                # 第一个打击点：相对距离 = 从起点走到这里
                dist = idx * step
            else:
                # 后续打击点：相对距离 = 两个点之间的间隔数 * 0.16
                dist = (idx - last_index) * step

            relative_loc.append(dist)
            last_index = idx  # 更新上一个打击点位置
    print(relative_loc)

    # 射击任务
    my_car.arm.set_arm_pose(arm="LEFT", hand="UP")
    my_car.arm.set_arm_pose(x=0.3, y=0.02)

    my_car.lane_dis_offset(speed=0.2, dis_hold=3.0)
    my_car.move_for([-0.2, 0, 0])
    # 对齐第一个目标
    my_car.move_to_detection_target(delta_x=d_x, delta_y=None, sort_pos=(d_x, 0))

    for dis in relative_loc:
        my_car.lane_dis_offset(speed=0.3, dis_hold=dis)
        cls_id, label = my_car.move_to_detection_target(
            delta_x=d_x, delta_y=None, sort_pos=(d_x, 0)
        )
        time.sleep(5)
        my_car.beep()
        my_car.shooting()
        time.sleep(5)

    my_car.lane_dis_offset(
        speed=0.3, dis_hold=0.48 - sum(relative_loc)
    )  # 距离补偿到最后一个目标


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
def find_goods(label, dy=-0.2):
    # #region debug-point A:find-goods-entry
    import json, urllib.request; urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:7777/event", data=json.dumps({"sessionId":"goods-align-none-setpoint","runId":"post-fix","hypothesisId":"A","location":"car_task_function.py:find_goods","msg":"[DEBUG] find_goods entry","data":{"label":label,"delta_x":0.0,"delta_y":dy}}).encode(), headers={"Content-Type":"application/json"})).read()
    # #endregion
    cls_id, det_label = my_car.move_to_detection_target(label=label, delta_x=0, delta_y=dy)
    # #region debug-point B:first-attempt
    import json, urllib.request; urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:7777/event", data=json.dumps({"sessionId":"goods-align-none-setpoint","runId":"post-fix","hypothesisId":"B","location":"car_task_function.py:find_goods:first","msg":"[DEBUG] first attempt returned","data":{"cls_id":cls_id,"det_label":det_label,"arm_x":my_car.arm.x_get_position()}}).encode(), headers={"Content-Type":"application/json"})).read()
    # #endregion
    if det_label is not None:
        return det_label

    my_car.arm.move_x_position(0.25)
    time.sleep(1)
    cls_id, det_label = my_car.move_to_detection_target(label=label, delta_x=0, delta_y=dy)
    # #region debug-point C:second-attempt
    import json, urllib.request; urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:7777/event", data=json.dumps({"sessionId":"goods-align-none-setpoint","runId":"post-fix","hypothesisId":"C","location":"car_task_function.py:find_goods:second","msg":"[DEBUG] second attempt returned","data":{"cls_id":cls_id,"det_label":det_label,"arm_x":my_car.arm.x_get_position()}}).encode(), headers={"Content-Type":"application/json"})).read()
    # #endregion
    if det_label is not None:
        return det_label

    my_car.move_for([0.10, 0, 0])
    time.sleep(1)
    # #region debug-point D:third-attempt-before
    import json, urllib.request; urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:7777/event", data=json.dumps({"sessionId":"goods-align-none-setpoint","runId":"post-fix","hypothesisId":"D","location":"car_task_function.py:find_goods:third","msg":"[DEBUG] before third attempt","data":{"label":label,"delta_x":None,"delta_y":dy,"arm_x":my_car.arm.x_get_position()}}).encode(), headers={"Content-Type":"application/json"})).read()
    # #endregion
    cls_id, det_label = my_car.move_to_detection_target(label=label, delta_x=None, delta_y=dy)
    if det_label is not None:
        return det_label


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
    goods_pickup_left_offset = 0.06

    # 启动阶段已完成 X 轴寻零，任务内避免重复触碰限位。
    my_car.arm.reset_position(rehome_x=False)
    my_car.lane_dis_offset(speed=0.2, dis_hold=1.5)
    my_car.move_to_detection_target(delta_y=None)

    # 保持推杆工作高度，避免水平推动时与机构干涉。
    my_car.arm.move_y_position(push_rod_y)
    my_car.move_for([0.065, 0, 0])
    my_car.arm.move_x_position(0.20)
    my_car.arm.move_x_position(0.03, out_time=4.0)
    time.sleep(0.5)

    my_car.move_for([-0.06, 0, 0])
    my_car.move_to_detection_target(delta_y=None)
    time.sleep(0.5)
    order_list.append(my_car.analyze_task_image(task="order", label="order"))
    my_car.beep()

    my_car.arm.move_y_position(0.2)
    my_car.arm.move_x_position(0.18)
    my_car.arm.set_hand_angle("MID")
    my_car.arm.set_arm_angle("RIGHT")
    time.sleep(0.5)
    _, fixed_label = my_car.move_to_detection_target(
        label="order", time_out=5.0
    )
    time.sleep(1)
    # 检测超时后使用整帧，避免因缺少检测框中断订单识别。
    order_list.append(
        my_car.analyze_task_image(
            task="order",
            label="order",
            full_frame=fixed_label != "order",
        )
    )
    my_car.beep()

    # 配送流程依赖该顺序确定取货次序。
    order_list.sort(key=lambda x: x["address"])
    my_car.lane_dis_offset(speed=0.2, dis_hold=0.20)
    my_car.arm.set_hand_angle(angle="DOWN")
    loc = my_car.get_odometry(True)

    my_car.set_storage(True)
    my_car.arm.move_y_position(0.2)
    # 从固定起点搜索，底盘位移仅由 find_goods() 控制。
    my_car.arm.move_x_position(0.18)
    goods_now = order_list[1]["goods"]
    if find_goods(goods_dict[goods_now]) is None:
        raise RuntimeError(f"订单第一件货物搜索失败，禁止盲抓: goods={goods_now}")
    time.sleep(0.5)

    # 视觉对齐后回缩，修正吸盘与识别中心的固定偏差。
    pickup_x = max(0.0, my_car.arm.x_get_position() - goods_pickup_left_offset)
    if not my_car.arm.move_x_position(pickup_x):
        raise RuntimeError("订单第一件货物抓取位置调整失败")
    my_car.arm.grasp(True)
    my_car.arm.move_y_position(0.05)
    time.sleep(0.5)
    my_car.arm.move_y_position(0.2)
    if not my_car.arm.move_x_position(0.0):
        raise RuntimeError("订单第一件货物未到达存储位置，禁止释放")
    my_car.arm.move_y_position(0.09)
    time.sleep(0.5)
    my_car.arm.grasp(False)

    my_car.move_to_position(loc)
    my_car.arm.move_y_position(0.2)
    # 第二件货物从同一固定起点重新搜索。
    my_car.arm.move_x_position(0.18)
    goods_now = order_list[0]["goods"]
    if find_goods(goods_dict[goods_now]) is None:
        raise RuntimeError(f"订单第二件货物搜索失败，禁止盲抓: goods={goods_now}")
    time.sleep(0.5)

    # 视觉对齐后回缩，修正吸盘与识别中心的固定偏差。
    pickup_x = max(0.0, my_car.arm.x_get_position() - goods_pickup_left_offset)
    if not my_car.arm.move_x_position(pickup_x):
        raise RuntimeError("订单第二件货物抓取位置调整失败")
    my_car.arm.grasp(True)
    my_car.arm.move_y_position(0.05)
    time.sleep(0.5)
    my_car.arm.move_y_position(0.2)
    if not my_car.arm.move_x_position(0.0):
        raise RuntimeError("订单第二件货物未到达存储位置，禁止释放")
    # 保留原仓位布局，确保与当前配送取货动作匹配。
    my_car.arm.move_y_position(0.14)
    time.sleep(0.5)
    my_car.arm.grasp(False)

    my_car.move_to_position(loc)
    return order_list


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


def order_delivery(    order_list = [
        {"name": "李四", "goods": "芹菜", "address": 2},
        {"name": "钱七", "goods": "青椒", "address": 2},
    ]):


    my_car.lane_dis_offset(speed=0.3, dis_hold=3.25)

    time.sleep(1)
    my_car.arm.move_y_position(0.2)
    my_car.arm.move_x_position(0.3)
    my_car.arm.set_arm_pose(arm="LEFT", hand=-70)
    time.sleep(1)
    cls_id, label = my_car.move_to_detection_target(delta_y=None)
    if label is None:
        my_car.lane_dis_offset(speed=0.3, dis_hold=0.12)
    time.sleep(1)
    # 记录1号楼起始位置
    loc_flag = 1
    loc = my_car.get_odometry(True)

    for i, order in enumerate(order_list):
        my_car.move_to_position(loc)
        if order["address"] > loc_flag:
            my_car.lane_dis_offset(speed=0.3, dis_hold=0.56)
            loc_flag = 2
            loc = my_car.get_odometry(True)
        time.sleep(0.5)

        # 调节识别高度
        my_car.arm.move_y_position(0.13)
        my_car.arm.move_x_position(0.3)
        my_car.arm.set_arm_pose(arm="LEFT", hand='UP')

        _x, y = find_name(order["name"])
        my_car.arm.set_arm_pose(arm="RIGHT", hand="DOWN")
        my_car.arm.move_x_position(0.0)
        my_car.arm.grasp(True)
        my_car.arm.move_y_position(0.135 - i * 0.05)
        my_car.arm.move_y_position(0.155 - i * 0.05)
        my_car.arm.move_x_position(0.2)
        my_car.arm.set_arm_pose(arm="LEFT", hand=-70)
        my_car.arm.move_y_position(y * 0.09)
        my_car.arm.move_x_position(0.1)
        my_car.arm.grasp(False)
        time.sleep(1)
        my_car.arm.move_x_position(0.15)
        my_car.arm.set_arm_pose(arm="LEFT", hand=-80)
        time.sleep(0.5)
        my_car.arm.move_x_position(0.2)
    
    if loc_flag == 1:
        my_car.lane_dis_offset(speed=0.3, dis_hold=1.7)
    else:
        my_car.lane_dis_offset(speed=0.3, dis_hold=1.1)
