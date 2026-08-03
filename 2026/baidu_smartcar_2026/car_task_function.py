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
    # 程序启动时执行本次运行唯一一次 X 轴机械归零。
    my_car.arm.reset_position()
    my_car.reset_position()  #


def auto_lane_tracing(speed=0.3, dis_hold=0.85):
    """同步自 Orin 2026-07-17 副本的独立巡线调试入口。"""
    my_car.lane_dis_offset(speed=speed, dis_hold=dis_hold)
    print(f"巡线停止的位置：{my_car.get_odometry()}")


def auto_seeding(debug=False):

    # 遥测范围说明：以下两个局部函数只读取状态并打印，不调用任何运动、抓取、释放或停车控制。
    # latest_detection_offsets() 会在视觉对齐返回后额外读取一帧检测结果，用于记录最后可见的 dx/dy；
    # 该额外读取不会改变 PID、速度、阈值、超时、标签选择或保存位姿逻辑。
    def print_arm_telemetry(stage, loop=None, expected_key=None, label=None,
                            cls_id=None, dx=None, dy=None, saved_pose=None):
        """打印播种抓取/放置阶段的最小遥测，不改变控制行为。"""
        odom = my_car.get_odometry()
        arm_x = my_car.arm.x_get_position()
        arm_y = my_car.arm.y_get_position()
        print(
            "[AUTO_SEEDING_TELEMETRY] "
            f"stage={stage} loop={loop} expected_key={expected_key} "
            f"label={label} cls_id={cls_id} dx={dx} dy={dy} "
            f"arm_x={arm_x:.4f} arm_y={arm_y:.4f} side={my_car.arm.side} "
            f"odom={odom} saved_pose={saved_pose}",
            flush=True,
        )

    def latest_detection_offsets():
        """读取对齐后的最新检测偏差，仅用于遥测。"""
        detections = my_car.get_detection_results()
        if not detections:
            return None, None
        return detections[0][4], detections[0][5]

    # 同步自 Orin /home/jetson/workspaces/baidu_smart_2026_7_17（2026-07-23）：
    # 现场把基地前方转角基准向前修正 1 cm；该参数不改变后续机械臂寻零和失败保护。
    x_length = 0.46  # 基地前方转角的位置，用于计算播种位置
    dis = 0.55  # 转角后第一个播种点的距离
    heading = math.pi / 4  # 车子的方向 45°
    sin45 = math.sin(heading)  # sin45°
    # 同步自同一 Orin 现场版本：目标姿态使用实测 0.74 rad，位置投影仍按理论 45° 计算。
    seeding_target_heading = 0.74
    # 正对播种点车子的理论位置
    cylinder_loc = {
        "cylinder_3": [x_length + dis * sin45, dis * sin45, seeding_target_heading],
        "cylinder_2": [x_length + (dis + 0.15) * sin45, (dis + 0.15) * sin45, seeding_target_heading],
        "cylinder_1": [x_length + (dis + 0.3) * sin45, (dis + 0.3) * sin45, seeding_target_heading],
    }
    cylinder_list = ["cylinder_3", "cylinder_2", "cylinder_1"]
    cylinder_set_list = {}

    # 右侧播种专用参数：第二轮三次抓取均在视觉识别前向车体右侧横移 0.02 m。
    RIGHT_SEEDING_PRE_DETECTION_OFFSET = [0.0, -0.02, 0.0]
    RIGHT_SEEDING_ARM_X_PRESET = 0.25
    RIGHT_SEEDING_ARM_Y_PRESET = 0.20
    RIGHT_SEEDING_ARM_X_BOUNDS = (0.24, 0.260)

    # 播种时摄像头需要朝下；全局姿态语义中 DOWN 对应物理朝下。
    my_car.arm.move_y_position(0.2)
    if not my_car.arm.retract_x_safe():
        raise RuntimeError("播种任务初始化水平轴安全回收失败，禁止旋转到 LEFT")
    my_car.arm.set_arm_pose(
        my_car.arm.x_safe_position, 0.2, "LEFT", "DOWN"
    )
    task_distance = 0.5 if debug else 0.85
    my_car.lane_dis_offset(speed=0.3, dis_hold=task_distance)
    time.sleep(0.5)
    print(f"巡线停止的位置：{my_car.get_odometry()}")

    for i in range(3):
        # 第一轮仍按原流程保存 odometry 和 arm_x；新增日志用于核对循环 key、识别 label 与保存位姿。
        my_car.move_to_position(cylinder_loc[cylinder_list[i]])
        # Orin 2026-07-17 最新标定：图像目标偏置为 (-0.1, -0.05)。
        # 修改前代码：first_cls_id, first_label = my_car.move_to_detection_target()
        first_cls_id, first_label = my_car.move_to_detection_target(
            delta_x=-0.1, delta_y=-0.05
        )
        first_dx, first_dy = latest_detection_offsets()
        print_arm_telemetry(
            "save_pose_after_alignment",
            loop=i,
            expected_key=cylinder_list[i],
            label=first_label,
            cls_id=first_cls_id,
            dx=first_dx,
            dy=first_dy,
        )
        x, y, z = my_car.get_odometry()
        corrected_x = x + 0.05 * math.cos(z)
        corrected_y = y + 0.05 * math.sin(z)
        pose = [corrected_x, corrected_y, z, my_car.arm.x_get_position()]
        print(f"第{i}个播种位置{pose}")
        cylinder_set_list[cylinder_list[i]] = pose
        my_car.beep()
    print("实际播种位置：")
    print(cylinder_set_list)

    for i in range(3):
        # 第二轮仍按原流程执行右侧抓取、恢复保存位姿、左侧放置；各日志点只标记控制动作前后状态。
        # 移动手臂到右侧高处
        # 先抬高机械臂，再在当前一侧收回水平轴，避免伸出状态旋转时碰撞限位。
        my_car.arm.move_y_position(RIGHT_SEEDING_ARM_Y_PRESET)
        # 本次启动已建立机械零点；旋转前只返回 1.5 cm 并确认开关释放。
        if not my_car.arm.retract_x_safe():
            raise RuntimeError("右侧抓取前水平轴安全回收失败，禁止旋转到 RIGHT")

        # 只有水平轴收回成功后才旋转到 RIGHT；set_arm_pose() 内部会等待舅机动作。
        my_car.arm.set_arm_pose(arm="RIGHT")

        # 旋转完成后再伸到右侧播种专用预抓取位置，其他任务的水平位置不受影响。
        if not my_car.arm.move_x_position(RIGHT_SEEDING_ARM_X_PRESET):
            raise RuntimeError(
                f"旋转到 RIGHT 后水平轴预伸出失败: "
                f"target={RIGHT_SEEDING_ARM_X_PRESET}"
            )
        print_arm_telemetry(
            "right_pre_alignment",
            loop=i,
            expected_key=cylinder_list[i],
        )

        # 对齐目标，识别
        # 完整执行原有到点轨迹；三次抓取均在到点停车后、视觉识别前修正车体坐标。
        base_pose = cylinder_loc[cylinder_list[i]]
        my_car.move_to_position(base_pose)
        time.sleep(0.5)
        correction_before = my_car.get_odometry()
        print(
            "[AUTO_SEEDING_COORD_CORRECTION] "
            f"stage=right_pre_detection_offset loop={i} "
            f"expected_key={cylinder_list[i]} "
            f"offset={RIGHT_SEEDING_PRE_DETECTION_OFFSET} "
            f"odom={correction_before}",
            flush=True,
        )
        my_car.move_for(RIGHT_SEEDING_PRE_DETECTION_OFFSET)
        correction_after = my_car.get_odometry()
        print(
            "[AUTO_SEEDING_COORD_CORRECTION] "
            f"stage=right_post_detection_offset loop={i} "
            f"expected_key={cylinder_list[i]} "
            f"offset={RIGHT_SEEDING_PRE_DETECTION_OFFSET} "
            f"odom={correction_after}",
            flush=True,
        )
        time.sleep(0.5)
        cls_id, label = my_car.move_to_detection_target(
            arm_x_bounds=RIGHT_SEEDING_ARM_X_BOUNDS
        )
        dx, dy = latest_detection_offsets()
        print_arm_telemetry(
            "right_after_alignment",
            loop=i,
            expected_key=cylinder_list[i],
            label=label,
            cls_id=cls_id,
            dx=dx,
            dy=dy,
        )
        print(f"识别到目标{cls_id}-{label}")
        my_car.beep()
        pose = cylinder_set_list[label]

        # 调整气泵吸嘴对齐目标
        # 右侧抓取前尝试水平补偿；即使补偿未到位，也按现场要求继续抓取。
        my_car.adjust_arm_position()
        print_arm_telemetry(
            "right_after_adjust",
            loop=i,
            expected_key=cylinder_list[i],
            label=label,
            cls_id=cls_id,
            dx=dx,
            dy=dy,
            saved_pose=pose,
        )
        # 吸起目标
        print_arm_telemetry(
            "right_before_grasp",
            loop=i,
            expected_key=cylinder_list[i],
            label=label,
            cls_id=cls_id,
            dx=dx,
            dy=dy,
            saved_pose=pose,
        )
        my_car.arm.grasp(True)
        print_arm_telemetry(
            "right_after_grasp",
            loop=i,
            expected_key=cylinder_list[i],
            label=label,
            cls_id=cls_id,
            dx=dx,
            dy=dy,
            saved_pose=pose,
        )
        my_car.arm.move_y_position(0.01)
        time.sleep(0.5)
        # Orin 2026-07-21 现场调整：抓取后抬升到 0.2 m。
        my_car.arm.move_y_position(0.2)

        # 抓取后仍保持 RIGHT，返回 1.5 cm 并确认开关释放后才允许旋转。
        if not my_car.arm.retract_x_safe():
            raise RuntimeError("右侧抓取后水平轴安全回收失败，禁止旋转到 LEFT")
        my_car.arm.set_arm_pose(arm="LEFT")
        time.sleep(1)
        if not my_car.arm.move_x_position(pose[3]):
            raise RuntimeError(
                f"LEFT 侧水平轴伸出失败: target={pose[3]}"
            )
        time.sleep(1)
        print_arm_telemetry(
            "left_before_place_alignment",
            loop=i,
            expected_key=cylinder_list[i],
            label=label,
            cls_id=cls_id,
            saved_pose=pose,
        )
        my_car.move_to_position(pose[:3])
        if not my_car.adjust_arm_position():
            raise RuntimeError("LEFT 侧放置前水平补偿失败，禁止继续释放")
        place_detection = next((det for det in my_car.get_detection_results() if det[2] == label), None)
        place_dx, place_dy = place_detection[4:6] if place_detection else (None, None)
        print_arm_telemetry(
            "left_after_adjust_before_release",
            loop=i,
            expected_key=cylinder_list[i],
            label=label,
            cls_id=cls_id,
            dx=place_dx,
            dy=place_dy,
            saved_pose=pose,
        )
        my_car.arm.move_y_position(0.04)
        my_car.arm.grasp(False)
        print_arm_telemetry(
            "left_after_release",
            loop=i,
            expected_key=cylinder_list[i],
            label=label,
            cls_id=cls_id,
            saved_pose=pose,
        )
        time.sleep(1)

    my_car.arm.move_y_position(0.1)
    # Orin 2026-07-17 最新结束姿态为 DOWN；修改前 worktree 为 UP。
    # 修改前代码：my_car.arm.set_arm_pose(hand="UP")
    my_car.arm.set_arm_pose(hand="DOWN")
    my_car.arm.move_x_position(0.15)
    my_car.move_to_position(cylinder_loc[cylinder_list[0]])
    print("播种完成")
    my_car.beep()
    my_car.beep()
    my_car.get_odometry(True)
    my_car.get_distance(True)
    if debug:
        my_car.move_to_position([0.0, 0.0, 0.0])


def target_shooting_detection(debug=False) -> list:

    # None 表示未获得可信识别结果，不能默认当作有害动物执行射击。
    animal_list = [None, None, None, None]
    my_car.arm.set_arm_pose(x=0.05, y=0.05, arm="LEFT", hand="UP")
    task_distance = 0.40 if debug else 1.45
    my_car.lane_dis_offset(speed=0.3, dis_hold=task_distance)

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
    if debug:
        my_car.move_to_position([0.0, 0.0, 0.0])
    return animal_list


def water_tower_task(debug=True):
    """
    水塔灌溉任务。

    调试模式会缩短进入任务区的巡线距离、将两个水塔的需求固定为三个水块，
    并在任务结束后返回任务起点。

    任务流程：
    1. 将机械臂调整到水塔标签搜索姿态，并向右平移后驶向第一座水塔。
    2. 识别 water_l1/water_l2/water_l3 标签，确定需要放置的水块数量。
    3. 按预设位置逐个搜索、吸取水块，再返回水塔基准位置完成放置。
    4. 移动到第二座水塔并重复识别、拾取和放置流程。

    :param debug: True 时使用短距离入口、每座塔固定放置三个水块并返回原点。
    """
    # 水块搜索高度和机械臂相对车体的水平预对齐位置。
    ARM_Y = 0.18
    INNER_OFFSET = 0.10
    OUTER_OFFSET = 0.00
    DISTANCE_BETWEEN_SLOTS = 0.28
    # 修改前为 0.00 m：水块视觉对齐完成后直接吸取，没有水平回缩补偿。
    WATER_PICKUP_RETRACT = -0.03
    # 水块放入水塔时水平轴的伸出距离，可根据现场水塔前后位置单独调整。
    # 修改前在放置流程中直接写死为 0.23 m。
    WATER_TOWER_PLACEMENT_X = 0.20
    # 抓取后 hand 提前翻到 UP，再在水平归零前额外上抬 5 cm 进行避障。
    # 该值只影响归零前的临时安全高度，不参与后续水塔分层放置高度计算。
    HAND_FLIP_CLEARANCE_LIFT = 0.05

    # 第一座水塔对应的三个水槽位置：内侧、外侧、前方水槽内侧。
    tower1_search_prealign_params = [
        {
            "arm_pose": {
                "x": INNER_OFFSET,
                "y": ARM_Y,
                "arm": "LEFT",
                "hand": "DOWN",
            },
            "car_offset": [0, 0, 0],
        },
        {
            "arm_pose": {
                "x": OUTER_OFFSET,
                "y": ARM_Y,
                "arm": "LEFT",
                "hand": "DOWN",
            },
            "car_offset": [0, 0, 0],
        },
        {
            "arm_pose": {
                "x": INNER_OFFSET,
                "y": ARM_Y,
                "arm": "LEFT",
                "hand": "DOWN",
            },
            "car_offset": [DISTANCE_BETWEEN_SLOTS, 0, 0],
        },
    ]

    # 第二座水塔对应的三个水槽位置：内侧、外侧、后方水槽外侧。
    tower2_search_prealign_params = [
        {
            "arm_pose": {
                "x": INNER_OFFSET,
                "y": ARM_Y,
                "arm": "LEFT",
                "hand": "DOWN",
            },
            "car_offset": [0, 0, 0],
        },
        {
            "arm_pose": {
                "x": OUTER_OFFSET,
                "y": ARM_Y,
                "arm": "LEFT",
                "hand": "DOWN",
            },
            "car_offset": [0, 0, 0],
        },
        {
            "arm_pose": {
                "x": OUTER_OFFSET,
                "y": ARM_Y,
                "arm": "LEFT",
                "hand": "DOWN",
            },
            "car_offset": [-DISTANCE_BETWEEN_SLOTS, 0, 0],
        },
    ]

    # 标签识别姿态保持在竖直轴最低点，避免需求标签超出摄像头视野。
    searching_pose = {
        "x": my_car.arm.x_safe_position,
        "y": 0.00,
        "arm": "RIGHT",
        "hand": "UP",
    }
    # 水块放入水塔前使用的基础姿态；每层高度会在此基础上递增。
    placing_pose = {
        "x": my_car.arm.x_safe_position,
        "y": 0.02,
        "arm": "RIGHT",
        "hand": "UP",
    }
    # 修改前使用 0.02 + 0.06 * loop，三个放置点为 0.02、0.08、0.14 m；
    # 现场将第三个放置点下调 0.02 m 至 0.10 m。
    placing_heights = (0.02, 0.08, 0.10)
    # 水塔需求标签与需要放置的水块数量。
    water_num = {
        "water_l1": 1,
        "water_l2": 2,
        "water_l3": 3,
    }

    def read_detection_offsets():
        """只读取最新检测偏差用于遥测，不参与任何控制决策。"""
        detections = my_car.get_detection_results()
        if not detections:
            return None, None
        return detections[0][4], detections[0][5]

    def print_water_telemetry(stage, tower_index=None, loop=None, label=None,
                              cls_id=None, target_index=None, dx=None, dy=None,
                              extra=None):
        """打印水塔任务阶段状态；仅读取状态，不发送运动/抓取命令。"""
        odom = my_car.get_odometry()
        arm_x = my_car.arm.x_get_position()
        arm_y = my_car.arm.y_get_position()
        fields = (
            f"stage={stage} tower_index={tower_index} loop={loop} "
            f"target_index={target_index} label={label} cls_id={cls_id} "
            f"dx={dx} dy={dy} arm_x={arm_x:.4f} arm_y={arm_y:.4f} "
            f"side={my_car.arm.side} odom={odom}"
        )
        if extra is not None:
            fields += f" extra={extra}"
        print(f"[WATER_TOWER_TELEMETRY] {fields}", flush=True)

    def get_water_block_needs_by_label(label):
        """根据识别标签返回水塔需要的水块数量。"""
        return water_num.get(label, 0)

    def retract_before_rotation(target_side, stage):
        """切换机械臂方向前返回 1.5 cm，并确认微动开关已经释放。"""
        if not my_car.arm.retract_x_safe():
            raise RuntimeError(
                f"水塔任务水平轴安全回收失败，禁止旋转到 {target_side}: "
                f"stage={stage}"
            )

    def pick_and_place_water_block(pos_params, tower_index, tower_label):
        """按照预对齐参数依次拾取水块并放入当前水塔。"""
        print_water_telemetry(
            "tower_execution_begin",
            tower_index=tower_index,
            label=tower_label,
            extra={"water_count": len(pos_params)},
        )
        for loop, pos_param in enumerate(pos_params):
            # 将当前水塔位置设为本轮局部里程计原点，取水后按此位置返回。
            tower_pose = my_car.get_odometry(True)
            print_water_telemetry(
                "watering_cycle_begin",
                tower_index=tower_index,
                loop=loop,
                label=tower_label,
                target_index=loop,
                extra={"water_count": len(pos_params)},
            )

            arm_pose = pos_param["arm_pose"]
            # 旋转机械臂前返回 1.5 cm 并确认开关释放，避免转动时碰撞水塔。
            retract_before_rotation(
                target_side=arm_pose["arm"],
                stage=f"tower={tower_index},loop={loop},before_pickup",
            )
            my_car.arm.move_y_position(arm_pose["y"])
            my_car.arm.set_arm_pose(
                arm=arm_pose["arm"],
                hand=arm_pose["hand"],
            )
            # 转向完成后才允许水平轴伸到水槽预对齐位置。关键伸出失败时
            # 立即停止本任务，不能在未知水平位置继续视觉对齐和吸取。
            if not my_car.arm.move_x_position(arm_pose["x"]):
                raise RuntimeError(
                    "水塔取水预对齐水平伸出失败: "
                    f"tower={tower_index}, loop={loop}, target={arm_pose['x']}"
                )
            car_offset = pos_param["car_offset"]
            print_water_telemetry(
                "before_move_to_water_block",
                tower_index=tower_index,
                loop=loop,
                label=tower_label,
                target_index=loop,
                extra={"arm_pose": arm_pose, "car_offset": car_offset},
            )
            my_car.move_for(car_offset)
            print_water_telemetry(
                "after_move_to_water_block",
                tower_index=tower_index,
                loop=loop,
                label=tower_label,
                target_index=loop,
            )

            # 使用侧摄像头寻找水块，并将目标对齐到预设的纵向偏移位置。
            print_water_telemetry(
                "before_water_block_detection",
                tower_index=tower_index,
                loop=loop,
                label=tower_label,
                target_index=loop,
            )
            cls_id, detected_label = my_car.move_to_detection_target(delta_y=-0.4)
            dx, dy = read_detection_offsets()
            print_water_telemetry(
                "after_water_block_detection",
                tower_index=tower_index,
                loop=loop,
                label=detected_label,
                cls_id=cls_id,
                target_index=loop,
                dx=dx,
                dy=dy,
            )
            print_water_telemetry(
                "after_water_block_alignment",
                tower_index=tower_index,
                loop=loop,
                label=detected_label,
                cls_id=cls_id,
                target_index=loop,
                dx=dx,
                dy=dy,
            )
            my_car.beep()

            # 水块视觉对齐完成后，将 LEFT 侧水平轴从当前位置往回缩 2 cm，
            # 补偿吸盘相对摄像头的抓取位置；修改前回缩量为 0.00 m。
            pickup_x_before_retract = my_car.arm.x_get_position()
            my_car.arm.move_x_position(
                pickup_x_before_retract - WATER_PICKUP_RETRACT
            )

            # 开启气泵吸取水块，然后抬高机械臂，为转向水塔留出空间。
            print_water_telemetry(
                "before_grasp",
                tower_index=tower_index,
                loop=loop,
                label=detected_label,
                cls_id=cls_id,
                target_index=loop,
            )
            my_car.arm.grasp(True)
            print_water_telemetry(
                "after_grasp",
                tower_index=tower_index,
                loop=loop,
                label=detected_label,
                cls_id=cls_id,
                target_index=loop,
            )
            my_car.arm.move_y_position(0.08)
            my_car.arm.move_y_position(0.12)

            # 抓取并抬升后，在水平轴仍保持当前抓取位置时先将 hand 翻到 UP。
            # 修改前 hand=UP 与 arm=RIGHT 一起在水平归零后才下发，导致内侧
            # 水块以 DOWN 姿态横向收回，运动轨迹可能碰到水槽中的外侧水块。
            # 这里只提前翻转末端 hand，不提前旋转 LEFT/RIGHT 总线舵机。
            my_car.arm.set_hand_angle(placing_pose["hand"])

            # hand 完成避障翻转后，在水平轴归零前从当前实际高度相对上抬
            # HAND_FLIP_CLEARANCE_LIFT（当前为 0.05 m）。使用当前反馈位置
            # 计算临时目标，确保每轮都是实际上抬 5 cm，而不是写死某一高度。
            y_before_clearance_lift = my_car.arm.y_get_position()
            my_car.arm.move_y_position(
                y_before_clearance_lift + HAND_FLIP_CLEARANCE_LIFT
            )

            # 完成 hand 翻转和竖直避障抬升后，再返回 1.5 cm 并确认开关释放。
            # 此时水块仍在 LEFT 侧；只有确认水平轴安全回收后才允许机械臂
            # 转到 RIGHT，继续保留“旋转机械臂前先收回水平轴”的防碰撞措施。
            retract_before_rotation(
                target_side=placing_pose["arm"],
                stage=f"tower={tower_index},loop={loop},after_pickup",
            )
            # hand 已在回收前翻到 UP，此处只旋转机械臂方向，避免重复改变
            # 末端姿态，同时保证 RIGHT 转向发生在水平轴安全回收之后。
            my_car.arm.set_arm_pose(arm=placing_pose["arm"])
            # 转到 RIGHT 后使用绝对放置高度覆盖上面的临时 +5 cm 抬升，
            # 因此避障高度不会累计或改变三个水塔分层放置点。水平轴继续
            # 保持在 1.5 cm 安全位置，同时避免 goto_position() 联合移动在零点
            # 残差处超时并掩盖切侧结果。
            my_car.arm.move_y_position(placing_heights[loop])
            print_water_telemetry(
                "after_water_pickup",
                tower_index=tower_index,
                loop=loop,
                label=detected_label,
                cls_id=cls_id,
                target_index=loop,
            )

            alignment_extra = {
                "tower_pose": tower_pose,
                "mode": "odometry_return",
            }
            # 依靠局部里程计返回水塔基准位置，不再进行一次水塔视觉对齐。
            print_water_telemetry(
                "before_tower_alignment",
                tower_index=tower_index,
                loop=loop,
                label=tower_label,
                target_index=loop,
                extra=alignment_extra,
            )
            my_car.move_to_position(tower_pose)
            print_water_telemetry(
                "after_tower_alignment",
                tower_index=tower_index,
                loop=loop,
                label=tower_label,
                target_index=loop,
                extra=alignment_extra,
            )

            # 将水平轴按可调参数伸入水塔。修改前固定伸出 0.23 m；现在可通过
            # WATER_TOWER_PLACEMENT_X 单独标定，而不需要修改放置流程代码。
            if not my_car.arm.move_x_position(WATER_TOWER_PLACEMENT_X):
                raise RuntimeError(
                    "水塔放置水平伸出失败: "
                    f"tower={tower_index}, loop={loop}, "
                    f"target={WATER_TOWER_PLACEMENT_X}"
                )
            my_car.arm.grasp(False)
            print_water_telemetry(
                "after_release",
                tower_index=tower_index,
                loop=loop,
                label=tower_label,
                target_index=loop,
            )
            print_water_telemetry(
                "watering_cycle_end",
                tower_index=tower_index,
                loop=loop,
                label=tower_label,
                target_index=loop,
            )

    # 遥测函数只读取车辆、机械臂和视觉状态，不参与运动控制。
    print_water_telemetry("begin", extra={"water_num": water_num})
    if not my_car.arm.retract_x_safe():
        raise RuntimeError("水塔任务初始化水平轴安全回收失败，禁止旋转到 RIGHT")
    my_car.arm.set_arm_pose(**searching_pose)
    print_water_telemetry("after_arm_pose_right")

    # 起步前向左平移 3 cm，等待车身稳定后再沿车道线前进。
    initial_right_offset = [0, 0.01, 0]
    my_car.move_for(initial_right_offset)
    print_water_telemetry(
        "after_initial_right_offset",
        extra={"car_offset": initial_right_offset, "settle_delay": 1.0},
    )
    time.sleep(1.0)

    # debug 模式缩短进入任务区的距离；首次巡线使用较低速度方便定位。
    task_distance = 0.6 if debug else 2.0
    my_car.lane_dis_offset(speed=0.2, dis_hold=task_distance)

    # 识别第一座水塔的需求标签；debug 模式固定取满三个水块。
    print_water_telemetry("before_first_tower_detection", tower_index=0)
    cls_id, label = my_car.move_to_detection_target(delta_y=None)
    dx, dy = read_detection_offsets()
    print_water_telemetry(
        "after_first_tower_detection",
        tower_index=0,
        label=label,
        cls_id=cls_id,
        dx=dx,
        dy=dy,
    )
    first_tower_pose = my_car.get_odometry()
    print_water_telemetry(
        "saved_first_tower_pose",
        tower_index=0,
        label=label,
        cls_id=cls_id,
        extra={"pose": first_tower_pose},
    )
    tower1_need = 3 if debug else get_water_block_needs_by_label(label)
    pick_and_place_water_block(
        tower1_search_prealign_params[:tower1_need],
        tower_index=0,
        tower_label=label,
    )

    # 恢复标签搜索姿态，并通过短巡线加直线位移绕开弯道进入第二座水塔。
    if not my_car.arm.retract_x_safe():
        raise RuntimeError("第二座水塔前水平轴安全回收失败，禁止恢复搜索姿态")
    my_car.arm.set_arm_pose(**searching_pose)
    print_water_telemetry("after_arm_pose_right", tower_index=1)
    my_car.lane_dis_offset(speed=0.3, dis_hold=0.4)
    my_car.move_for([0.2, 0, 0])
    print_water_telemetry(
        "after_middle_lane_offset",
        tower_index=1,
        extra={"lane_distance": 0.4, "car_offset": [0.2, 0, 0]},
    )

    # 识别第二座水塔需求，并使用第二组预对齐位置执行灌溉。
    print_water_telemetry("before_second_tower_detection", tower_index=1)
    cls_id, label = my_car.move_to_detection_target(delta_y=None)
    dx, dy = read_detection_offsets()
    print_water_telemetry(
        "after_second_tower_detection",
        tower_index=1,
        label=label,
        cls_id=cls_id,
        dx=dx,
        dy=dy,
    )
    second_tower_pose = my_car.get_odometry()
    print_water_telemetry(
        "saved_second_tower_pose",
        tower_index=1,
        label=label,
        cls_id=cls_id,
        extra={"pose": second_tower_pose},
    )
    tower2_need = 3 if debug else get_water_block_needs_by_label(label)
    pick_and_place_water_block(
        tower2_search_prealign_params[:tower2_need],
        tower_index=1,
        tower_label=label,
    )
    # 任务结束时返回 1.5 cm 安全位置，避免机械臂保持外伸状态。
    if not my_car.arm.retract_x_safe():
        raise RuntimeError("水塔任务结束时水平轴安全回收失败")

    # 分离式调试结束后返回进入水塔任务时记录的局部原点。
    if debug:
        print_water_telemetry("before_debug_return_origin")
        my_car.move_to_position([0.0, 0.0, 0.0])
        print_water_telemetry("after_debug_return_origin")




def target_shooting(animal_list=None, debug=False):  # noqa: E741

    if animal_list is None:
        animal_list = [None, None, None, None]

    step = 0.16  # 每个目标间距
    relative_loc = []  # 记录相对运动距离
    last_index = -1  # 记录上一个打击点的索引，初始为-1
    d_x = 0.2  # 对齐参数

    for idx, value in enumerate(animal_list):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in (0, 1)
        ):
            print(
                "[TARGET_SHOOTING_SKIP] "
                f"index={idx} reason=invalid_or_unknown_result "
                f"value_type={type(value).__name__}",
                flush=True,
            )
            continue
        if (
            value == 0
        ):  # 只有可信的有害动物结果才进入射击列表
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

    task_distance = 0.70 if debug else 3.0
    my_car.lane_dis_offset(speed=0.3, dis_hold=task_distance)
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
    if debug:
        my_car.move_to_position([0.0, 0.0, 0.0])


def crop_harvesting(debug=True):
    """
    作物采收
    """
    # 作物抓取专用水平回缩补偿：视觉对齐和吸嘴补偿完成后往回缩 2 cm。
    # 修改前回缩量为 0.00 m；后续现场标定只需调整此常量。
    CROP_PICKUP_RETRACT = 0.005

    # 遥测范围说明：以下局部函数只读取车辆、机械臂和视觉状态并打印，
    # 不发送运动、抓取、释放或储物架控制命令。storage_command_angle
    # 表示本流程最近一次请求的逻辑角度，并非舵机位置回读。
    def latest_detection_offsets():
        """读取对齐后的最新检测偏差，仅用于遥测。"""
        detections = my_car.get_detection_results()
        if not detections:
            return None, None
        return detections[0][4], detections[0][5]

    def print_crop_telemetry(stage, loop=None, label=None, cls_id=None,
                             dx=None, dy=None, storage_state=None,
                             storage_command_angle=None, extra=None):
        """打印作物采收阶段状态，不改变任务控制行为。"""
        odom = my_car.get_odometry()
        arm_x = my_car.arm.x_get_position()
        arm_y = my_car.arm.y_get_position()
        fields = (
            f"stage={stage} loop={loop} label={label} cls_id={cls_id} "
            f"dx={dx} dy={dy} arm_x={arm_x:.4f} arm_y={arm_y:.4f} "
            f"side={my_car.arm.side} odom={odom} "
            f"storage_state={storage_state} "
            f"storage_command_angle={storage_command_angle}"
        )
        if extra is not None:
            fields += f" extra={extra}"
        print(f"[CROP_HARVESTING_TELEMETRY] {fields}", flush=True)

    print_crop_telemetry("begin", extra={"debug": debug})

    # 调整机械臂
    my_car.arm.move_y_position(0.2)
    # 启动时已经机械归零；进入任务时只返回 1.5 cm 并确认开关释放。
    if not my_car.arm.retract_x_safe():
        raise RuntimeError("作物采收初始化水平轴安全回收失败，禁止旋转到 LEFT")
    my_car.arm.set_arm_pose(arm="LEFT", hand="DOWN")
    print_crop_telemetry("after_initial_arm_pose")

    # 当前实际映射：set_storage(False) 下发 165°，对应物理“抬起/收起”储物架。
    # 修改前为 set_storage(True)，实际下发 -85°，对应物理“放下”。
    my_car.set_storage(False)  # 抬起储物架。
    print_crop_telemetry(
        "after_storage_false_raise",
        storage_state=False,
        storage_command_angle=my_car.servo_1_angle_list[0],
    )

    # 移动到任务位置
    task_distance = 0.70 if debug else 2.3
    my_car.lane_dis_offset(speed=0.3, dis_hold=task_distance)
    my_car.arm.move_y_position(0.17)
    print_crop_telemetry(
        "after_entry_lane",
        extra={"task_distance": task_distance},
    )

    # 对齐 Orin /home/jetson/workspaces/baidu_smart_2026_7_17/ 的仓位判定：
    # 记录第一轮识别到的作物类别，后续同类作物统一放到 x=0.06 m；
    # 修改前按固定标签分仓，ball_yellow 放到 0.0 m、ball_blue 放到 0.02 m。
    first_label = None
    for i in range(8):
        print_crop_telemetry("cycle_begin", loop=i)
        # 调整机械臂
        # 每轮抓取前返回 1.5 cm 并确认开关释放，只有成功才允许转到 LEFT。
        if not my_car.arm.retract_x_safe():
            raise RuntimeError(
                f"作物抓取前水平轴安全回收失败，禁止旋转到 LEFT: loop={i}"
            )
        # 末端手腕在初始化或上一轮放球后已经保持 DOWN；这里只翻转到 LEFT，
        # 修改前会在翻转完成后重复下发一次 hand="DOWN"。
        my_car.arm.set_arm_pose(arm="LEFT")
        # 前进一小段
        my_car.lane_dis_offset(speed=0.3, dis_hold=0.04)
        time.sleep(0.5)
        # 对齐目标
        alignment_timeout = 3.0
        alignment_start = time.time()
        cls_id, label = my_car.move_to_detection_target(
            delta_x=-0.05,
            time_out=alignment_timeout,
        )
        alignment_elapsed = time.time() - alignment_start
        alignment_timed_out = alignment_elapsed >= alignment_timeout
        dx, dy = latest_detection_offsets()
        # 与 Orin 旧目录一致，以第一轮实际识别结果作为 0.06 m 仓位类别基准；
        # 修改前不记录首轮类别，而是直接按 ball_yellow/ball_blue 标签选择仓位。
        if i == 0:
            first_label = label
        if alignment_timed_out:
            print(
                "[CROP_HARVESTING_TIMEOUT] "
                f"stage=detection_alignment loop={i} "
                f"timeout={alignment_timeout:.1f}s "
                f"elapsed={alignment_elapsed:.3f}s "
                f"label={label} cls_id={cls_id} dx={dx} dy={dy}",
                flush=True,
            )
        print_crop_telemetry(
            "after_detection_alignment",
            loop=i,
            label=label,
            cls_id=cls_id,
            dx=dx,
            dy=dy,
            extra={
                "alignment_status": (
                    "timeout" if alignment_timed_out else "success"
                ),
                "alignment_timeout": alignment_timeout,
                "alignment_elapsed": alignment_elapsed,
            },
        )
        print(f"发现第{i + 1}个作物，目标为{label}")
        time.sleep(0.5)
        my_car.adjust_arm_position()
        print_crop_telemetry(
            "after_arm_adjustment",
            loop=i,
            label=label,
            cls_id=cls_id,
            dx=dx,
            dy=dy,
        )
        # 视觉对齐和吸嘴水平补偿完成后，从当前实际位置往回缩可调距离；
        # 修改前直接在补偿后的位置抓球，没有额外回缩。
        pickup_x_before_retract = my_car.arm.x_get_position()
        pickup_x_target = pickup_x_before_retract - CROP_PICKUP_RETRACT
        if not my_car.arm.move_x_position(pickup_x_target):
            raise RuntimeError(
                "作物抓取前水平回缩失败: "
                f"loop={i}, before={pickup_x_before_retract:.6f}, "
                f"retract={CROP_PICKUP_RETRACT:.3f}, "
                f"target={pickup_x_target:.6f}"
            )
        print_crop_telemetry(
            "after_pickup_retract",
            loop=i,
            label=label,
            cls_id=cls_id,
            dx=dx,
            dy=dy,
            extra={
                "pickup_x_before_retract": pickup_x_before_retract,
                "pickup_retract": CROP_PICKUP_RETRACT,
                "pickup_x_target": pickup_x_target,
            },
        )
        my_car.arm.grasp(True)
        print_crop_telemetry(
            "after_grasp",
            loop=i,
            label=label,
            cls_id=cls_id,
        )
        time.sleep(0.3)
        # 对齐 Orin 旧目录的吸取高度为 0.09 m；修改前本地值为 0.10 m。
        my_car.arm.move_y_position(0.09)  # 吸取
        time.sleep(0.3)
        # 对齐 Orin 旧目录，抓取后抬升到 0.20 m；修改前本地值为 0.17 m。
        my_car.arm.move_y_position(0.20)  # 抬起机械臂
        time.sleep(0.3)
        # 水平轴先返回 1.5 cm 并确认开关释放，避免带着作物外伸翻转。
        if not my_car.arm.retract_x_safe():
            raise RuntimeError(
                f"作物抓取后水平轴安全回收失败，禁止旋转到放球位置: loop={i}"
            )
        # 将机械臂翻转到置物架放球方向 -115°；修改前现场值为 -110°。
        my_car.arm.set_arm_angle(-110)
        # 对齐 Orin 旧目录的大角度翻转等待时间 2 秒；修改前本地值为 1 秒。
        time.sleep(2)

        # 对齐 Orin 旧目录的动态分仓规则：首轮类别放到 x=0.06 m，其他类别
        # 放到 x=0.0 m；修改前固定为黄球 0.0 m、蓝球 0.02 m。
        target_x = 0.06 if label == first_label else 0.0
        # 保留本地现场安全保护：水平定位失败时禁止在未知位置释放作物。
        if not my_car.arm.move_x_position(target_x):
            raise RuntimeError(
                "作物放球水平定位失败: "
                f"loop={i}, label={label}, first_label={first_label}, "
                f"target={target_x}"
            )
        time.sleep(0.5)

        # 对齐 Orin 旧目录，在水平仓位到位后、调整手腕和释放前重复发送一次
        # 放球方向；修改前本地仅在水平移动前发送一次。首次翻转目标 -115°
        # 未在本次点名参数范围内，因此重复发送同一个本地现场目标值。
        my_car.arm.set_arm_angle(-110)
        # 对齐 Orin 旧目录的放球手腕角度 -35°；修改前本地值为 -12°。
        my_car.arm.set_hand_angle(-35)
        if label == first_label:
            my_car.beep()
        else:
            my_car.beep()
            my_car.beep()
        time.sleep(1)
        print_crop_telemetry(
            "before_release",
            loop=i,
            label=label,
            cls_id=cls_id,
        )
        my_car.arm.grasp(False)
        # 放球完成后立即将末端手腕恢复并保持 DOWN；修改前会保持 -110°，
        # 直到下一轮翻转到 LEFT 后才设置 DOWN，导致翻转过程中手腕姿态不一致。
        my_car.arm.set_hand_angle("DOWN")
        print_crop_telemetry(
            "after_release",
            loop=i,
            label=label,
            cls_id=cls_id,
        )
        time.sleep(1)
    my_car.set_storage(False)  # 放下存储架
    print_crop_telemetry(
        "after_storage_false",
        storage_state=False,
        storage_command_angle=my_car.servo_1_angle_list[0],
    )
    if debug:
        print_crop_telemetry("before_debug_return_origin")
        my_car.move_to_position([0.0, 0.0, 0.0])
        print_crop_telemetry("after_debug_return_origin")


def sort_and_store(
    first_label=None,
    ball_count_dict=None,
    debug=True,
):
    """从车载储物架取出黄、蓝球，并分别放入高、低球仓。

    Args:
        first_label: crop_harvesting() 第一轮识别到的球标签。该标签对应
            储物架外侧水平位置 x=0.06 m，另一种球对应 x=0.0 m。
            debug=True 且未传入时默认外侧为蓝球；非调试模式必须显式传入。
        ball_count_dict: 球标签到取球数量的映射；None 时黄、蓝球各 4 个。
            修改前使用可变字典作为默认参数，可能在被调用方修改后污染后续调用。
        debug: True 时缩短进入任务区的巡线距离，并在结束后返回局部原点。
    """

    if ball_count_dict is None:
        ball_count_dict = {"ball_yellow": 4, "ball_blue": 4}

    # 入口搜索姿态。实际执行由统一的分步函数完成，不再组合设置四个轴。
    SEARCHING_POSE = {
        "x": 0.25,
        "y": 0.05,
        "arm": "LEFT",
        "hand": "UP",
    }

    # 采收任务把 first_label 放在外侧 x=0.06 m，另一颜色放在 x=0.0 m。
    # 单任务调试约定外侧为蓝球；正式任务禁止猜测，必须传入采收首轮标签。
    valid_ball_labels = ("ball_yellow", "ball_blue")
    if first_label is None:
        if debug:
            first_label = "ball_blue"
        else:
            raise ValueError(
                "非调试模式必须传入 crop_harvesting() 的 first_label，"
                "禁止在未知储物架仓位下取球"
            )
    if first_label not in valid_ball_labels:
        raise ValueError(
            "sort_and_store 的 first_label 必须是 "
            f"{valid_ball_labels} 之一，当前值为 {first_label!r}"
        )
    other_label = (
        "ball_blue" if first_label == "ball_yellow" else "ball_yellow"
    )
    storage_x_by_label = {
        first_label: 0.06,
        other_label: 0.0,
    }

    # 取球姿态：水平位置来自采收分仓映射；黄、蓝球使用各自翻转角度，
    # 手腕统一为现场标定的 -35°。
    ARM_PARAMS_FOR_PICKING = {
        "ball_yellow": {
            "x": storage_x_by_label["ball_yellow"],
            "y": 0.11,
            "arm": -110,
            "hand": -35,
        },
        "ball_blue": {
            "x": storage_x_by_label["ball_blue"],
            "y": 0.11,
            "arm": -110,
            "hand": -35,
        },
    }

    # 放球姿态：第 0 项为高仓，第 1 项为低仓；两者只在高度上不同。
    ARM_PARAMS_FOR_PLACING = [
        {
            "x": 0.20,
            "y": 0.20,
            "arm": "LEFT",
            "hand": "UP",
        },
        {
            "x": 0.20,
            "y": 0.10,
            "arm": "LEFT",
            "hand": "UP",
        },
    ]

    # 两球仓中心间距；修改前本地值为 0.155 m。
    DISTANCE_BETWEEN_BALL_SLOTS = 0.16

    # 左右翻转前的水平安全位置。修改前从取球侧转到放球侧时会先执行
    # reset_x()/回收到 x=0，再翻转到 LEFT；现场要求改为先平移到
    # x=0.20 m，再翻转，最后移动到实际放球位置。修改前安全位置为
    # x=0.25 m；放球结束后也保持/回到 x=0.20 m，避免下一轮在 LEFT
    # 侧直接回收时撞击放置目标。
    # 当前水平轴软件上限为 0.255 m，因此每次移动都必须检查返回值。
    ARM_ROTATION_SAFE_X = 0.20

    # 修改前没有独立吸附稳定等待，而是依赖“提前开启气泵后再移动机械臂”的
    # 隐式等待。现在改为到达取球位后才吸气，并显式等待 0.5 s 建立负压。
    GRASP_SETTLE_SECONDS = 0.5

    # 修改前放球后没有独立泄压等待。关闭气泵、打开泄压阀后等待 0.5 s，
    # 避免小球仍被残余负压吸住就开始执行放置侧脱离动作。
    RELEASE_SETTLE_SECONDS = 0.5

    # 标签统一使用正确拼写；旧模型配置仍可能返回 lable_*，仅在视觉结果
    # 进入本任务的边界处兼容转换，后续判断和遥测一律使用 label_*。
    # 修改前直接判断 label_yellow，收到旧拼写或未知标签时会误入蓝球分支。
    LABEL_ALIASES = {
        "lable_yellow": "label_yellow",
        "lable_blue": "label_blue",
    }
    VALID_STORAGE_LABELS = ("label_yellow", "label_blue")

    def print_sort_telemetry(
        stage,
        loop=None,
        ball_label=None,
        raw_label=None,
        normalized_label=None,
        target_slot=None,
        storage_state=None,
        storage_command_angle=None,
        extra=None,
    ):
        """打印分类存储关键阶段状态，不发送任何运动或吸取命令。"""
        odom = my_car.get_odometry()
        # 修改前每条遥测都会调用 x_get_position()/y_get_position()，从而向
        # MC602 额外发送两次串口查询。现场已出现串口 EIO，因此遥测改读
        # 各运动函数持续维护的缓存位置，避免诊断日志本身增加串口压力。
        arm_x = getattr(my_car.arm, "x_pose_now", 0.0)
        arm_y = getattr(my_car.arm, "y_pose_now", 0.0)
        fields = (
            f"stage={stage} loop={loop} ball_label={ball_label} "
            f"raw_label={raw_label} normalized_label={normalized_label} "
            f"target_slot={target_slot} arm_x={arm_x:.4f} "
            f"arm_y={arm_y:.4f} side={my_car.arm.side} "
            f"arm_angle={getattr(my_car.arm, '_arm_angle_last', None)} "
            f"hand_angle={getattr(my_car.arm, '_hand_angle_last', None)} "
            f"odom={odom} storage_state={storage_state} "
            f"storage_command_angle={storage_command_angle}"
        )
        if extra is not None:
            fields += f" extra={extra}"
        print(f"[SORT_AND_STORE_TELEMETRY] {fields}", flush=True)

    def set_task_arm_pose_step_by_step(
        target_pose,
        stage_prefix,
        loop=None,
        ball_label=None,
        target_slot=None,
        rotate_from_safe_x=False,
        home_before_target_rotation=False,
    ):
        """按目标所在侧的安全顺序，分步设置取球或放球姿态。

        入口流程先安全回收再翻转。取球和放球流程传入
        rotate_from_safe_x=True，统一先移动到 x=0.20 m 再切侧；其中取球
        流程额外传入 home_before_target_rotation=True：先翻到储物架基准侧
        RIGHT，完成 1.5 cm 安全回收后才下发黄/蓝球的精确取球角度。
        """
        safe_y = 0.20

        # 修改前组合姿态会先同时移动 x/y，超时后仍继续翻转舵机。现在先
        # 抬到 0.20 m，再物理寻零，避免水平轴伸出时旋转。
        my_car.arm.move_y_position(safe_y)
        print_sort_telemetry(
            f"{stage_prefix}_after_safe_raise",
            loop=loop,
            ball_label=ball_label,
            target_slot=target_slot,
            extra={"target_y": safe_y, "target_pose": target_pose},
        )

        if rotate_from_safe_x:
            # 修改前取球流程会在 LEFT 放球侧直接 reset_x()/回收到 x=0；
            # 修改前放球流程也会在储物架侧先 reset_x()/回收到 x=0。
            # 现在两种转换都先到 x=0.20 m（修改前为 0.25 m），只有明确
            # 到位后才允许翻转。
            if not my_car.arm.move_x_position(ARM_ROTATION_SAFE_X):
                print_sort_telemetry(
                    f"{stage_prefix}_rotation_safe_x_failed",
                    loop=loop,
                    ball_label=ball_label,
                    target_slot=target_slot,
                    extra={"target_x": ARM_ROTATION_SAFE_X},
                )
                raise RuntimeError(
                    "分类存储机械臂翻转前移动到水平安全位置失败，禁止旋转: "
                    f"stage={stage_prefix}, loop={loop}, "
                    f"target_x={ARM_ROTATION_SAFE_X}"
                )
            print_sort_telemetry(
                f"{stage_prefix}_after_rotation_safe_x",
                loop=loop,
                ball_label=ball_label,
                target_slot=target_slot,
                extra={"target_x": ARM_ROTATION_SAFE_X},
            )
        else:
            if not my_car.arm.retract_x_safe():
                print_sort_telemetry(
                    f"{stage_prefix}_safe_retract_failed",
                    loop=loop,
                    ball_label=ball_label,
                    target_slot=target_slot,
                    extra={"target_pose": target_pose},
                )
                raise RuntimeError(
                    "分类存储机械臂分步姿态水平安全回收失败，禁止继续旋转: "
                    f"stage={stage_prefix}, loop={loop}, "
                    f"ball_label={ball_label}, target_slot={target_slot}"
                )
            print_sort_telemetry(
                f"{stage_prefix}_after_safe_retract",
                loop=loop,
                ball_label=ball_label,
                target_slot=target_slot,
            )

        if home_before_target_rotation:
            # 取球切侧不能在 LEFT 放球侧直接回收到 1.5 cm，否则仍可能沿回收
            # 路径撞击放置目标。因此先在 x=0.20 m 翻到储物架基准侧
            # RIGHT=-92°，再在储物架侧安全回收。修改前是先下发黄球
            # -100°/蓝球 -107°，随后才归零；现在精确取球角度会在
            # 安全回收完成后由下面的统一目标角度步骤下发。
            my_car.arm.set_arm_angle("RIGHT")
            time.sleep(1)
            print_sort_telemetry(
                f"{stage_prefix}_after_storage_base_rotation",
                loop=loop,
                ball_label=ball_label,
                target_slot=target_slot,
                extra={"base_arm": "RIGHT", "target_arm": target_pose["arm"]},
            )

            if not my_car.arm.retract_x_safe():
                print_sort_telemetry(
                    f"{stage_prefix}_safe_retract_before_target_rotation_failed",
                    loop=loop,
                    ball_label=ball_label,
                    target_slot=target_slot,
                    extra={"target_pose": target_pose},
                )
                raise RuntimeError(
                    "分类存储机械臂储物架侧水平安全回收失败，禁止设置精确取球角度: "
                    f"stage={stage_prefix}, loop={loop}, "
                    f"ball_label={ball_label}, target_slot={target_slot}"
                )
            print_sort_telemetry(
                f"{stage_prefix}_after_safe_retract_before_target_rotation",
                loop=loop,
                ball_label=ball_label,
                target_slot=target_slot,
                extra={"target_arm": target_pose["arm"]},
            )

        # 入口路径在水平轴完全回收后设置目标角度；放球路径在 x=0.20 m
        # 到位后直接翻转到 LEFT；取球路径则先翻到 RIGHT 并寻零，再设置
        # 黄球 -100°/蓝球 -107° 的精确取球角度。
        my_car.arm.set_arm_angle(target_pose["arm"])
        time.sleep(1)
        print_sort_telemetry(
            f"{stage_prefix}_after_arm_rotation",
            loop=loop,
            ball_label=ball_label,
            target_slot=target_slot,
            extra={"target_arm": target_pose["arm"]},
        )

        my_car.arm.set_hand_angle(target_pose["hand"])
        print_sort_telemetry(
            f"{stage_prefix}_after_hand_pose",
            loop=loop,
            ball_label=ball_label,
            target_slot=target_slot,
            extra={"target_hand": target_pose["hand"]},
        )

        # 翻转和手腕姿态稳定后再水平伸出；move_x_position() 会明确返回
        # 到位结果，失败时禁止下降到取球/放球高度。
        if not my_car.arm.move_x_position(target_pose["x"]):
            print_sort_telemetry(
                f"{stage_prefix}_move_x_failed",
                loop=loop,
                ball_label=ball_label,
                target_slot=target_slot,
                extra={"target_x": target_pose["x"]},
            )
            raise RuntimeError(
                "分类存储机械臂分步姿态水平伸出失败，禁止下降: "
                f"stage={stage_prefix}, loop={loop}, "
                f"target_x={target_pose['x']}"
            )
        print_sort_telemetry(
            f"{stage_prefix}_after_move_x",
            loop=loop,
            ball_label=ball_label,
            target_slot=target_slot,
            extra={"target_x": target_pose["x"]},
        )

        # 最后下降到参数表中的目标高度；修改前 x/y 由 goto_position()
        # 同时控制，现在可以从遥测中区分水平到位与竖直下降阶段。
        my_car.arm.move_y_position(target_pose["y"])
        print_sort_telemetry(
            f"{stage_prefix}_complete",
            loop=loop,
            ball_label=ball_label,
            target_slot=target_slot,
            extra={"target_pose": target_pose},
        )

    def pick_and_place_ball_by_params(
        arm_params_for_picking,
        arm_params_for_placing,
        ball_label,
        target_slot,
        loop,
    ):
        """按给定取球和放球姿态安全转运一个球。"""
        print_sort_telemetry(
            "cycle_begin",
            loop=loop,
            ball_label=ball_label,
            target_slot=target_slot,
            extra={
                "picking": arm_params_for_picking,
                "placing": arm_params_for_placing,
            },
        )
        # 修改前气泵会在进入取球姿态前开启，并持续到整个放置姿态完成；
        # 放置阶段一旦异常或被 Ctrl+C 中断，关闭命令就永远不会执行。
        # 现在先到达取球位置，再开启气泵，并用 finally 保证所有退出路径
        # 都至少尝试下发一次关闭气泵、打开泄压阀的命令。
        set_task_arm_pose_step_by_step(
            arm_params_for_picking,
            stage_prefix="picking_pose",
            loop=loop,
            ball_label=ball_label,
            target_slot=target_slot,
            rotate_from_safe_x=True,
            home_before_target_rotation=True,
        )

        active_transfer_error = None
        pump_may_be_on = False
        try:
            print_sort_telemetry(
                "before_grasp_on",
                loop=loop,
                ball_label=ball_label,
                target_slot=target_slot,
            )
            # 在调用前即标记“气泵可能已开启”：若 pump.set(True) 成功、
            # valve.set(False) 随后因串口异常失败，finally 仍会尝试关闭。
            pump_may_be_on = True
            my_car.arm.grasp(True)
            print_sort_telemetry(
                "after_grasp_on",
                loop=loop,
                ball_label=ball_label,
                target_slot=target_slot,
            )
            time.sleep(GRASP_SETTLE_SECONDS)

            # 放球分步函数会先把已吸取的小球直接抬到安全高度 0.20 m，
            # 因此继续省略修改前重复的两次 y=0.12 m 中间过渡。
            set_task_arm_pose_step_by_step(
                arm_params_for_placing,
                stage_prefix="placing_pose",
                loop=loop,
                ball_label=ball_label,
                target_slot=target_slot,
                # 修改前从取球侧转到放球侧时先 reset_x()/回收到 x=0；
                # 现在按现场要求先到 x=0.20 m（修改前为 0.25 m），再翻转
                # LEFT，随后由分步函数移动到实际放球位置 x=0.20 m 和
                # 对应仓位高度。
                rotate_from_safe_x=True,
            )
        except BaseException as exc:
            # 同时覆盖普通异常和 Ctrl+C；若随后关闭气泵也失败，保留这里的
            # 原始动作异常，关闭失败通过独立遥测输出，避免掩盖首要故障。
            active_transfer_error = exc
            raise
        finally:
            if pump_may_be_on:
                release_reason = (
                    "exception" if active_transfer_error is not None else "normal"
                )
                print_sort_telemetry(
                    "before_grasp_off",
                    loop=loop,
                    ball_label=ball_label,
                    target_slot=target_slot,
                    extra={"reason": release_reason},
                )
                try:
                    my_car.arm.grasp(False)
                except Exception as release_error:
                    print_sort_telemetry(
                        "grasp_off_failed",
                        loop=loop,
                        ball_label=ball_label,
                        target_slot=target_slot,
                        extra={
                            "reason": release_reason,
                            "error": repr(release_error),
                        },
                    )
                    # 正常放置时关闭失败必须中止任务；若已有动作异常，则继续
                    # 抛出原始异常，避免被二次串口错误覆盖。
                    if active_transfer_error is None:
                        raise
                else:
                    print_sort_telemetry(
                        "after_grasp_off",
                        loop=loop,
                        ball_label=ball_label,
                        target_slot=target_slot,
                        extra={"reason": release_reason},
                    )
                    time.sleep(RELEASE_SETTLE_SECONDS)

        # 只有放置姿态和释放命令均正常完成后才执行机械脱离；异常路径仅
        # 尝试关闭气泵，不在未知位置继续移动机械臂。
        my_car.arm.move_y_position(0.20)
        print_sort_telemetry(
            "after_release_safe_raise",
            loop=loop,
            ball_label=ball_label,
            target_slot=target_slot,
            extra={"target_y": 0.20},
        )
        if not my_car.arm.move_x_position(ARM_ROTATION_SAFE_X):
            print_sort_telemetry(
                "placing_side_escape_failed",
                loop=loop,
                ball_label=ball_label,
                target_slot=target_slot,
                extra={"target_x": ARM_ROTATION_SAFE_X},
            )
            raise RuntimeError(
                "分类存储机械臂放球后 LEFT 侧脱离失败，禁止进入下一轮: "
                f"loop={loop}, ball_label={ball_label}, "
                f"target_slot={target_slot}, target_x={ARM_ROTATION_SAFE_X}"
            )
        print_sort_telemetry(
            "placing_side_escape_complete",
            loop=loop,
            ball_label=ball_label,
            target_slot=target_slot,
            extra={"target_x": ARM_ROTATION_SAFE_X},
        )

    def process_ball_group(ball_label, placing_pose, target_slot):
        """按配置数量处理一种颜色，消除高、低仓重复循环。"""
        picking_pose = ARM_PARAMS_FOR_PICKING[ball_label]
        count = ball_count_dict[ball_label]
        print_sort_telemetry(
            "ball_group_begin",
            ball_label=ball_label,
            target_slot=target_slot,
            extra={
                "count": count,
                "picking_pose": picking_pose,
                "placing_pose": placing_pose,
            },
        )
        for loop in range(count):
            pick_and_place_ball_by_params(
                picking_pose,
                placing_pose,
                ball_label=ball_label,
                target_slot=target_slot,
                loop=loop,
            )
        print_sort_telemetry(
            "ball_group_end",
            ball_label=ball_label,
            target_slot=target_slot,
            extra={"count": count},
        )

    print_sort_telemetry(
        "begin",
        extra={
            "debug": debug,
            "first_label": first_label,
            "ball_count_dict": ball_count_dict,
            "storage_x_by_label": storage_x_by_label,
        },
    )

    # 修改前任务入口不会主动关闭气泵；若上一次运行在吸气状态下因异常或
    # Ctrl+C 退出，控制器输出可能保持为开启，导致新任务从入口起就持续
    # 吸气。现在在任何机械臂和车辆动作前明确恢复“泵关闭、泄压阀打开”。
    # 该命令若因串口故障抛出异常，任务会立即中止，不在泵状态未知时继续。
    print_sort_telemetry("before_initial_grasp_off")
    my_car.arm.grasp(False)
    print_sort_telemetry("after_initial_grasp_off")

    # 入口使用“抬高、寻零、翻转、伸出、下降”的常规顺序；取球阶段则
    # 使用 LEFT 侧先到 x=0.20 m（修改前为 0.25 m）、翻转到储物架侧后
    # 再寻零的避障顺序。
    # 入口任一步失败都会在进入巡线之前直接中止。
    set_task_arm_pose_step_by_step(
        SEARCHING_POSE,
        stage_prefix="initial_pose",
    )

    # 当前入口实际命令为 True；本地映射下会下发 -85°。修改前注释仍
    # 描述 False/0°，与代码不一致，现在按实际命令记录储物架遥测。
    my_car.set_storage(True)
    print_sort_telemetry(
        "after_storage_true",
        storage_state=True,
        storage_command_angle=my_car.servo_1_angle_list[1],
    )

    # 本次未点名修改巡线距离，继续保留本地调试 0.55 m、正常 2.0 m。
    task_distance = 0.55 if debug else 2.0
    print_sort_telemetry(
        "before_entry_lane",
        extra={"task_distance": task_distance, "speed": 0.3},
    )
    my_car.lane_dis_offset(speed=0.3, dis_hold=task_distance)
    print_sort_telemetry(
        "after_entry_lane",
        extra={"task_distance": task_distance},
    )

    # 识别高球仓标签并决定两种颜色的处理顺序。修改前判断字符串为
    # 拼写错误的 "lable_blue"；现在按 Gitee 分支判断 "label_yellow"，
    # 未识别为黄色标签时按蓝色高仓处理。
    print_sort_telemetry("before_storage_label_detection")
    cls_id, raw_storage_label = my_car.move_to_detection_target(delta_y=None)
    label = LABEL_ALIASES.get(raw_storage_label, raw_storage_label)
    print_sort_telemetry(
        "after_storage_label_detection",
        raw_label=raw_storage_label,
        normalized_label=label,
        extra={"cls_id": cls_id},
    )
    if label not in VALID_STORAGE_LABELS:
        print_sort_telemetry(
            "invalid_storage_label",
            raw_label=raw_storage_label,
            normalized_label=label,
            extra={"cls_id": cls_id, "valid_labels": VALID_STORAGE_LABELS},
        )
        raise RuntimeError(
            "分类存储球仓标签无效，禁止按默认颜色继续: "
            f"raw_label={raw_storage_label!r}, normalized_label={label!r}"
        )
    # 球仓标签表示高仓应接收的颜色；另一种颜色自动分配到低仓。
    high_slot_ball = {
        "label_yellow": "ball_yellow",
        "label_blue": "ball_blue",
    }[label]
    low_slot_ball = (
        "ball_blue" if high_slot_ball == "ball_yellow" else "ball_yellow"
    )
    print_sort_telemetry(
        "storage_order_selected",
        raw_label=raw_storage_label,
        normalized_label=label,
        extra={
            "high_slot_ball": high_slot_ball,
            "low_slot_ball": low_slot_ball,
        },
    )

    # 先处理高仓颜色，再移动到低仓处理另一颜色。每组数量由参数字典控制。
    process_ball_group(
        high_slot_ball,
        ARM_PARAMS_FOR_PLACING[0],
        target_slot="HIGH",
    )

    # 从高球仓移动到低球仓；修改前后移距离为 0.155 m。
    print_sort_telemetry(
        "before_low_slot_offset",
        extra={"car_offset": [-DISTANCE_BETWEEN_BALL_SLOTS, 0, 0]},
    )
    my_car.move_for([-DISTANCE_BETWEEN_BALL_SLOTS, 0, 0])
    print_sort_telemetry(
        "after_low_slot_offset",
        extra={"car_offset": [-DISTANCE_BETWEEN_BALL_SLOTS, 0, 0]},
    )

    process_ball_group(
        low_slot_ball,
        ARM_PARAMS_FOR_PLACING[1],
        target_slot="LOW",
    )

    # 结束储物架命令对齐 Gitee 分支。修改前本地函数没有结束命令；
    # 当前本地映射下 True 实际下发 -85°。
    my_car.set_storage(True)
    print_sort_telemetry(
        "after_final_storage_true",
        storage_state=True,
        storage_command_angle=my_car.servo_1_angle_list[1],
    )

    if debug:
        print_sort_telemetry("before_debug_return_origin")
        my_car.move_to_position([0.0, 0.0, 0.0])
        print_sort_telemetry("after_debug_return_origin")


# 排序后的订单与车载置物架仓位一一对应。get_order() 存货和
# order_delivery() 取货必须共用同一组参数，避免左右侧或高度错配。
ORDER_STORAGE_SLOTS = (
    {
        "arm": "LEFT",              # 第一件订单对应左侧车载置物架。
        "x": 0.24,                  # 左侧仓位存货和配送取货的 X 轴绝对位置。
        "place_y": 0.16,            # get_order() 左侧仓位释放时的 Y 轴绝对高度。
        "pickup_y": (0.135, 0.155),  # 配送取货吸附后先到 0.135 m，再抬到 0.155 m。
    },
    {
        "arm": "RIGHT",             # 第二件订单对应右侧车载置物架。
        "x": 0.00,                  # 右侧仓位存货和配送取货的 X 轴绝对位置。
        "place_y": 0.16,            # get_order() 右侧仓位释放时的 Y 轴绝对高度。
        "pickup_y": (0.085, 0.105),  # 配送取货吸附后先到 0.085 m，再抬到 0.105 m。
    },
)

# 订单配送各阶段的机械臂姿态集中配置。动作流程只引用这里的参数，
# 现场标定时无需在 order_delivery() 中逐处查找 X/Y/arm/hand 数值。
ORDER_DELIVERY_POSES = {
    # 进入配送区域后，识别楼房标签时使用的机械臂姿态。
    "initial_search": {
        "arm": "LEFT",  # 楼房标签搜索时机械臂朝向。
        "hand": "UP",   # 楼房标签搜索时 hand 姿态；UP 当前对应 89°。
        "x": 0.245,     # 翻转机械臂前，X 轴先移动到的绝对位置。
        "y": 0.20,      # 移动 X 和翻转机械臂前先抬升到的 Y 轴高度。
    },
    # 到达目标楼房后，识别收货人姓名时使用的机械臂姿态。
    "name_search": {
        "arm": "LEFT",  # 姓名搜索阶段保持的机械臂朝向。
        "hand": "UP",   # OCR 识别姓名标签时的 hand 姿态；UP 当前对应 89°。
        "x": 0.245,       # 姓名标签搜索时的 X 轴绝对位置。
        "y": 0.13,       # 姓名标签搜索时的 Y 轴绝对高度。
    },
    # 从车载置物架取货后，向楼房货架移动并释放货物时使用的参数。
    "building_place": {
        "arm": "LEFT",       # 楼房侧放置货物时机械臂朝向。
        "hand": "UP",        # 楼房侧接近和释放货物时的 hand 姿态。
        "approach_x": 0.20,   # 翻转到楼房侧前，X 轴先移动到的接近位置。
        "row_y_step": 0.07,   # 每个 OCR 行号对应的 Y 轴层高增量。
        "release_x": 0.15,    # 下降到目标层后，释放货物时的 X 轴位置。
    },
    # 货物释放完成后，机械臂恢复到下一单准备姿态时使用的参数。
    "after_release": {
        "arm": "LEFT",          # 释放完成后保持的机械臂朝向。
        "hand": "DOWN",         # 释放完成后恢复的 hand 姿态；DOWN 当前对应 -3°。
        "transition_x": 0.15,    # 释放后、调整 hand 姿态前的 X 轴过渡位置。
        "ready_x": 0.20,         # hand 调整完成后，下一单开始前的 X 轴准备位置。
    },
}

# 订单配送中，水平轴停止、机械臂翻转和 hand 调整之间的可调等待时间。
ORDER_DELIVERY_TIMING = {
    "x_settle_before_flip": 0.5,   # X 到达目标位置后，发送翻转指令前等待时间。
    "arm_settle_before_hand": 1.5,  # arm 收到新响应后，调整 hand 前等待时间。
}


# 订单获取任务中的文字识别、蔬菜搜索和蔬菜抓取姿态集中配置。
# get_order() 和 find_goods() 只引用这里的参数；现场重新标定时可以按动作阶段
# 修改对应字段，不需要在识别、重试和两次抓取流程中逐处查找硬编码数值。
GET_ORDER_POSES = {
    # 从任务起点进入订单识别区域时使用的车辆巡线距离。
    "task_entry": {
        "debug_distance": 0.60,  # debug=True 时进入订单区的短距离，单位为米。
        "run_distance": 1.50,    # 正式运行时从任务起点进入订单区的距离，单位为米。
    },
    # 推开订单机构并识别随机订单文字时使用的机械臂和车辆位置。
    "random_order_label": {
        "push_rod_y": 0.025,        # 推动订单推杆时，机械臂 Y 轴保持的绝对高度。
        "push_car_offset": [0.065, 0, 0],  # 推动推杆前车辆沿 X 方向前移 6.5 cm。
        "push_x": 0.17,             # 接触推杆前，机械臂 X 轴先伸到的绝对位置。
        "return_x": 0.05,           # 推杆动作完成后，机械臂 X 轴回收到的绝对位置。
        "return_timeout": 4.0,       # X 轴从推杆位置回收时允许的最长动作时间，单位为秒。
        "camera_car_offset": [-0.06, 0, 0],  # 拍摄随机订单文字前车辆后退 6 cm。
    },
    # 识别固定订单文字前使用的机械臂完整姿态。
    "fixed_order_label": {
        "arm": "RIGHT",  # 固定订单标签位于车辆右侧，识别前机械臂转向 RIGHT。
        "hand": "MID",   # 固定订单文字识别时使用的 hand 中间姿态。
        "x": 0.16,        # 固定订单文字进入相机视野时的 X 轴绝对位置。
        "y": 0.20,        # 固定订单文字识别前机械臂抬升到的 Y 轴绝对高度。
    },
    # 完成两张订单文字识别后，进入蔬菜区域并开始视觉搜索时使用的位置。
    "goods_area": {
        "approach_distance": 0.20,  # 从订单识别区巡线进入蔬菜拿取区的距离。
        "search_x": 0.25,           # 每次搜索蔬菜前，机械臂 X 轴伸出的绝对位置。
        "search_y": 0.20,           # 每次搜索蔬菜前，机械臂 Y 轴保持的绝对高度。
    },
    # find_goods() 首次识别失败后依次执行的蔬菜视觉搜索位置。
    "goods_search": {
        "delta_y": -0.50,             # 蔬菜目标视觉对齐时使用的纵向图像偏移。
        "retry_x": 0.15,              # 第一次失败后回收到该 X 位置并再次识别。
        "retry_car_offset": [0.15, 0, 0],  # 第二次失败后车辆前移 15 cm 再次识别。
        "final_retry_x": 0.25,         # 第三次失败后伸到该 X 位置进行最后一次识别。
    },
    # 视觉找到蔬菜后，开启吸附并将蔬菜移出拿取区时使用的位置。
    "goods_pickup": {
        "left_offset": 0.02,  # 对齐蔬菜后，X 轴从实时位置向左回缩的抓取补偿距离。
        "lower_y": 0.05,      # 吸附蔬菜后先下降到的 Y 轴绝对高度。
        "lift_y": 0.20,       # 离开拿取位置并准备放入车载置物架时的 Y 轴高度。
        "second_flip_x": 0.24,  # 第二件蔬菜转向 LEFT 置物架前的 X 轴安全位置。
    },
}


# 寻找货物的程序
def find_goods(label, dy=GET_ORDER_POSES["goods_search"]["delta_y"]):
    # 订单、姓名等仍由 task_det 识别；只有蔬菜搜索使用 8_2 专用模型。
    # 四次搜索均显式传入同一检测器，避免重试阶段退回旧任务模型。
    goods_search_pose = GET_ORDER_POSES["goods_search"]
    cls_id, det_label = my_car.move_to_detection_target(
        label=label, delta_y=dy, detector=my_car.goods_det
    )
    if det_label is not None:
        return det_label

    my_car.arm.move_x_position(goods_search_pose["retry_x"])
    cls_id, det_label = my_car.move_to_detection_target(
        label=label, delta_y=dy, detector=my_car.goods_det
    )
    if det_label is not None:
        return det_label

    my_car.move_for(goods_search_pose["retry_car_offset"])
    cls_id, det_label = my_car.move_to_detection_target(
        label=label, delta_y=dy, detector=my_car.goods_det
    )
    if det_label is not None:
        return det_label

    my_car.arm.move_x_position(goods_search_pose["final_retry_x"])
    cls_id, det_label = my_car.move_to_detection_target(
        label=label, delta_y=dy, detector=my_car.goods_det
    )
    if det_label is not None:
        return det_label


def get_order(debug=True):
    task_entry_pose = GET_ORDER_POSES["task_entry"]
    random_order_label_pose = GET_ORDER_POSES["random_order_label"]
    fixed_order_label_pose = GET_ORDER_POSES["fixed_order_label"]
    goods_area_pose = GET_ORDER_POSES["goods_area"]
    goods_pickup_pose = GET_ORDER_POSES["goods_pickup"]

    # 标签对应关系
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

    order_list = []  # 千帆多模态模型直接返回的结构化订单

    # 本次启动已经建立 X 机械零点；这里只重置 Y，并让 X 返回 1.5 cm。
    my_car.arm.reset_position(rehome_x=False)
    task_distance = (
        task_entry_pose["debug_distance"]
        if debug
        else task_entry_pose["run_distance"]
    )
    my_car.lane_dis_offset(speed=0.3, dis_hold=task_distance)
    # 对齐订单
    cls_id, label = my_car.move_to_detection_target(delta_y=None)
    # 抬到配置的推杆高度并保持，直到后续固定标签阶段切换到识别高度。
    my_car.arm.move_y_position(random_order_label_pose["push_rod_y"])
    # 推动推杆
    my_car.move_for(random_order_label_pose["push_car_offset"])
    my_car.arm.move_x_position(random_order_label_pose["push_x"])
    my_car.arm.move_x_position(
        random_order_label_pose["return_x"],
        out_time=random_order_label_pose["return_timeout"],
    )
    time.sleep(0.5)
    # 识别随机标签
    my_car.move_for(random_order_label_pose["camera_car_offset"])
    cls_id, label = my_car.move_to_detection_target(delta_y=None)
    time.sleep(0.5)
    order_list.append(
        my_car.analyze_task_image(task="order", label="order")
    )
    my_car.beep()
    # 识别固定标签
    my_car.arm.move_y_position(fixed_order_label_pose["y"])
    my_car.arm.move_x_position(fixed_order_label_pose["x"])
    my_car.arm.set_hand_angle(fixed_order_label_pose["hand"])
    my_car.arm.set_arm_angle(fixed_order_label_pose["arm"])
    time.sleep(0.5)
    cls_id, label = my_car.move_to_detection_target()
    time.sleep(1)
    my_car.get_detection_results()
    order_list.append(
        my_car.analyze_task_image(task="order", label="order")
    )
    my_car.beep()

    print(order_list)
    # 对订单排序，先拿2号楼的
    order_list.sort(key=lambda x: x["address"])
    print(order_list)

    my_car.lane_dis_offset(
        speed=0.3,
        dis_hold=goods_area_pose["approach_distance"],
    )
    my_car.arm.set_hand_angle(angle="DOWN")

    loc = my_car.get_odometry(True)

    my_car.set_storage(True)  # 抬起存储架
    my_car.arm.move_y_position(goods_area_pose["search_y"])
    my_car.arm.move_x_position(goods_area_pose["search_x"])
    cls_id, label = my_car.move_to_detection_target(delta_y=None)
    first_order_index = 1
    first_storage_slot = ORDER_STORAGE_SLOTS[first_order_index]
    goods_now = order_list[first_order_index]["goods"]
    detected_good = find_goods(goods_dict[goods_now])
    if detected_good is None:
        raise RuntimeError(
            f"订单第一件货物搜索失败，禁止盲抓: goods={goods_now}"
        )
    print(f"正在拿取第一个货物：{goods_now}")
    time.sleep(0.5)
    # 第一件抓取时机械臂保持 RIGHT；视觉识别成功后，X 从当前实际位置
    # 向左回缩 goods_pickup.left_offset，再开启吸附。
    pickup_x_before_offset = my_car.arm.x_get_position()
    pickup_x_target = pickup_x_before_offset - goods_pickup_pose["left_offset"]
    if not my_car.arm.move_x_position(pickup_x_target):
        raise RuntimeError(
            "订单第一件货物抓取前向左微调失败: "
            f"before={pickup_x_before_offset:.6f}, "
            f"offset={goods_pickup_pose['left_offset']:.3f}, "
            f"target={pickup_x_target:.6f}"
        )
    my_car.arm.grasp(True)
    # 第一件抓取后按 goods_pickup.lower_y 和 lift_y 完成下降、抬升，
    # 再进入右侧车载置物架放置流程。
    my_car.arm.move_y_position(goods_pickup_pose["lower_y"])
    time.sleep(0.5)
    my_car.arm.move_y_position(goods_pickup_pose["lift_y"])
    if not my_car.arm.move_x_position(first_storage_slot["x"]):
        raise RuntimeError("订单第一件货物右侧放置水平轴未能到达目标位置，禁止下降释放")
    # 第一件右侧放置的 X/Y 由 ORDER_STORAGE_SLOTS 对应仓位决定；
    # 到位后将 hand 转到 -10° 并释放，再恢复抓取初始姿态 DOWN。
    my_car.arm.move_y_position(first_storage_slot["place_y"])
    my_car.arm.set_hand_angle(-10)
    time.sleep(0.5)
    my_car.arm.grasp(False)
    my_car.arm.set_hand_angle("DOWN")
    time.sleep(0.5)
    # 拿第二个货物
    my_car.move_to_position(loc)
    my_car.arm.move_y_position(goods_area_pose["search_y"])
    my_car.arm.move_x_position(goods_area_pose["search_x"])
    cls_id, label = my_car.move_to_detection_target(delta_y=None)
    second_order_index = 0
    second_storage_slot = ORDER_STORAGE_SLOTS[second_order_index]
    goods_now = order_list[second_order_index]["goods"]
    detected_good = find_goods(goods_dict[goods_now])
    if detected_good is None:
        raise RuntimeError(
            f"订单第二件货物搜索失败，禁止盲抓: goods={goods_now}"
        )
    print(f"正在拿取第二个货物：{goods_now}")
    time.sleep(0.5)
    # 第二件抓取时机械臂保持 RIGHT；视觉识别成功后，X 从当前实际位置
    # 向左回缩 goods_pickup.left_offset，再开启吸附。
    pickup_x_before_offset = my_car.arm.x_get_position()
    pickup_x_target = pickup_x_before_offset - goods_pickup_pose["left_offset"]
    if not my_car.arm.move_x_position(pickup_x_target):
        raise RuntimeError(
            "订单第二件货物抓取前向左微调失败: "
            f"before={pickup_x_before_offset:.6f}, "
            f"offset={goods_pickup_pose['left_offset']:.3f}, "
            f"target={pickup_x_target:.6f}"
        )
    my_car.arm.grasp(True)
    # 第二件抓取后按 goods_pickup.lower_y 和 lift_y 完成下降、抬升，
    # 为安全切换到 LEFT 留出高度。
    my_car.arm.move_y_position(goods_pickup_pose["lower_y"])
    time.sleep(0.5)
    my_car.arm.move_y_position(goods_pickup_pose["lift_y"])
    # 固定标签阶段机械臂保持在 RIGHT；放入左侧置物架前先将
    # 水平轴伸到 goods_pickup.second_flip_x，再翻转到 LEFT，避免翻转轨迹碰到置物架。
    if not my_car.arm.move_x_position(goods_pickup_pose["second_flip_x"]):
        raise RuntimeError(
            "订单第二件货物左侧翻转前水平轴未能到达安全位置，"
            f"target={goods_pickup_pose['second_flip_x']:.3f}，禁止旋转到 LEFT"
        )
    my_car.arm.set_arm_pose(arm=second_storage_slot["arm"])
    if not my_car.arm.move_x_position(second_storage_slot["x"]):
        raise RuntimeError("订单第二件货物左侧放置水平轴未能到达 0.20 m，禁止下降释放")
    # 第二件左侧放置按 goods_pickup.second_flip_x 完成翻转前避障，
    # 再使用 ORDER_STORAGE_SLOTS 对应仓位的 X/Y 位置；到位后将 hand
    # 转到 -18° 并释放，最后恢复抓取初始姿态 DOWN。
    my_car.arm.move_y_position(second_storage_slot["place_y"])
    my_car.arm.set_hand_angle(-18)
    time.sleep(0.5)
    my_car.arm.grasp(False)
    my_car.arm.set_hand_angle("DOWN")
    time.sleep(0.5)

    my_car.move_to_position(loc)
    if debug:
        my_car.move_to_position([0.0, 0.0, 0.0])
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


def order_delivery(
    order_list=[
        {"name": "李四", "goods": "芹菜", "address": 2},
        {"name": "钱七", "goods": "青椒", "address": 2},
    ],
    debug=True,
):

    initial_search_pose = ORDER_DELIVERY_POSES["initial_search"]
    name_search_pose = ORDER_DELIVERY_POSES["name_search"]
    building_place_pose = ORDER_DELIVERY_POSES["building_place"]
    after_release_pose = ORDER_DELIVERY_POSES["after_release"]

    def move_x_then_flip(target_x, target_arm, target_hand, stage):
        """按明确时序完成 X 定位、arm 翻转确认和 hand 姿态调整。"""
        if not my_car.arm.move_x_position(target_x):
            raise RuntimeError(
                f"订单配送翻转前水平定位失败: stage={stage}, "
                f"target_x={target_x}, target_arm={target_arm}"
            )
        time.sleep(ORDER_DELIVERY_TIMING["x_settle_before_flip"])

        if not my_car.arm.set_arm_angle(target_arm):
            raise RuntimeError(
                f"订单配送机械臂翻转无新响应，禁止继续: stage={stage}, "
                f"target_x={target_x}, target_arm={target_arm}"
            )
        time.sleep(ORDER_DELIVERY_TIMING["arm_settle_before_hand"])

        if target_hand is not None:
            my_car.arm.set_hand_angle(target_hand)

    task_distance = 0.60 if debug else 3.25
    my_car.lane_dis_offset(speed=0.3, dis_hold=task_distance)

    time.sleep(1)
    my_car.arm.move_y_position(initial_search_pose["y"])
    move_x_then_flip(
        target_x=initial_search_pose["x"],
        target_arm=initial_search_pose["arm"],
        target_hand=initial_search_pose["hand"],
        stage="initial_search",
    )
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

        # 姓名搜索姿态由 ORDER_DELIVERY_POSES["name_search"] 统一配置。
        my_car.arm.move_y_position(name_search_pose["y"])
        if not my_car.arm.move_x_position(name_search_pose["x"]):
            raise RuntimeError(f"订单配送姓名搜索水平伸出失败: order_index={i}")
        my_car.arm.set_hand_angle(name_search_pose["hand"])

        _x, y = find_name(order["name"])
        storage_slot = ORDER_STORAGE_SLOTS[i]
        move_x_then_flip(
            target_x=storage_slot["x"],
            target_arm=storage_slot["arm"],
            target_hand="DOWN",
            stage=f"storage_pickup[{i}]",
        )
        # 配送取货开启吸附时 Y 仍为姓名搜索高度；X 和 arm 由仓位决定：
        # order_list[0] 为 LEFT/X=0.24 m，order_list[1] 为 RIGHT/X=0.00 m。
        my_car.arm.grasp(True)
        pickup_y, lift_y = storage_slot["pickup_y"]
        # LEFT 仓取货沿 Y=0.135 -> 0.155 m；RIGHT 仓沿 Y=0.085 -> 0.105 m。
        my_car.arm.move_y_position(pickup_y)
        my_car.arm.move_y_position(lift_y)
        move_x_then_flip(
            target_x=building_place_pose["approach_x"],
            target_arm=building_place_pose["arm"],
            target_hand=building_place_pose["hand"],
            stage=f"building_place[{i}]",
        )
        # 楼房侧放置的接近位置、层高步距和释放位置统一由配置决定。
        room_y = y * building_place_pose["row_y_step"]
        my_car.arm.move_y_position(room_y)
        my_car.arm.move_x_position(building_place_pose["release_x"])
        my_car.arm.grasp(False)
        time.sleep(1)
        move_x_then_flip(
            target_x=after_release_pose["transition_x"],
            target_arm=after_release_pose["arm"],
            target_hand=after_release_pose["hand"],
            stage=f"after_release[{i}]",
        )
        time.sleep(0.5)
        my_car.arm.move_x_position(after_release_pose["ready_x"])
    
    if debug:
        my_car.move_to_position([0.0, 0.0, 0.0])
    elif loc_flag == 1:
        my_car.lane_dis_offset(speed=0.3, dis_hold=1.7)
    else:
        my_car.lane_dis_offset(speed=0.3, dis_hold=1.1)
