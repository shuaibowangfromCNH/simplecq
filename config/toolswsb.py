from typing import Dict,List
import numpy as np
import math

def parse_Output(state: List[Dict[str, any]]) -> dict:
    #example:如果要加别的输入数据，思考下如何修改，比如只想获取我车的distance参数。
    tmp = []
    Input = {}
    for input in state:
        for k, v in input.items():
            if k == 'carInfo':
                tmp.append(v)
            else:
                Input[k] = v
    Input['carInfo'] = tmp
    return Input

def process_raw_state(raw_state: Dict[str, any]) -> tuple:
    """
    将原始状态字典转换为状态向量
    
    根据文档设计的状态空间：
    [d, cos(θ_rel), sin(θ_rel), v_self, v_enemy, cos(Δψ), sin(Δψ), d_dot, ...]
    """
    try:
        # 提取车辆信息
        if raw_state['carInfo'][0]['baseInfo']['side'] == 1:
            self_car = raw_state['carInfo'][0]
            enemy_car = raw_state['carInfo'][1]
        else:
            self_car = raw_state['carInfo'][1]
            enemy_car = raw_state['carInfo'][0]
        
        # 提取位置和速度
        self_pos = self_car['position']
        enemy_pos = enemy_car['position']
        self_vel = self_car['velocity']
        enemy_vel = enemy_car['velocity']
        
        # 计算相对位置向量
        dx = enemy_pos['x'] - self_pos['x']
        dy = enemy_pos['y'] - self_pos['y']
        dz = enemy_pos['z'] - self_pos['z']
        
        # 距离
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 相对角度（目标相对于我车航向）
        # 假设有航向信息
        self_heading = self_car.get('heading', 0.0)  # 需要从状态中获取
        enemy_heading = enemy_car.get('heading', 0.0)
        
        # 相对角度计算
        target_angle = np.arctan2(dy, dx)
        theta_rel = target_angle - self_heading
        # 归一化到 [-π, π]
        theta_rel = (theta_rel + np.pi) % (2 * np.pi) - np.pi
        
        # 航向差
        delta_psi = enemy_heading - self_heading
        delta_psi = (delta_psi + np.pi) % (2 * np.pi) - np.pi
        
        # 速度信息
        v_self = np.sqrt(self_vel['x']**2 + self_vel['y']**2 + self_vel['z']**2)
        v_enemy = np.sqrt(enemy_vel['x']**2 + enemy_vel['y']**2 + enemy_vel['z']**2)
        
        # 相对速度
        v_rel_x = enemy_vel['x'] - self_vel['x']
        v_rel_y = enemy_vel['y'] - self_vel['y']
        
        # 接近率
        if distance > 0.001:
            d_dot = -(dx * v_rel_x + dy * v_rel_y) / distance
        else:
            d_dot = 0.0
        
        # 构建状态向量（根据文档设计）
        state_vector = np.array([
            distance,                   # 相对距离
            np.cos(theta_rel),           # 相对角度的余弦
            np.sin(theta_rel),           # 相对角度的正弦
            v_self,                     # 自身速度
            v_enemy,                    # 目标速度
            np.cos(delta_psi),          # 航向差余弦
            np.sin(delta_psi),          # 航向差正弦
            d_dot,                      # 接近率
            v_rel_x,                     # 相对速度x
            v_rel_y,                    # 相对速度y
            dx/distance if distance > 0 else 0,  # 归一化相对位置x
            dy/distance if distance > 0 else 0,  # 归一化相对位置y
        ], dtype=np.float32)
        
        # 状态归一化
        normalized_state_vector = normalize_state(state_vector.copy())
        
        return state_vector, normalized_state_vector
        
    except Exception as e:
        print(f"状态处理错误: {e}")
        # 返回零状态
        zero_vector = np.zeros(12, dtype=np.float32)
        return zero_vector, zero_vector
    
def normalize_state(state: np.ndarray) -> np.ndarray:
    """
    状态归一化
    """
    if len(state) != 12:
        raise ValueError(f"状态向量维度应为12，但得到{len(state)}")
    
    normalized = state.copy()
    
    # 1. 相对距离 [0] - 范围: [0, +∞)
    # 假设最大观测距离为10000米，使用对数压缩处理大范围距离
    if state[0] > 0:
        normalized[0] = np.log1p(state[0]) / np.log1p(10000)  # 对数归一化
    else:
        normalized[0] = 0.0
    
    # 2. 相对角度余弦 [1] - 范围: [-1, 1]，已经是归一化的
    # 不需要额外处理
    
    # 3. 相对角度正弦 [2] - 范围: [-1, 1]，已经是归一化的
    # 不需要额外处理
    
    # 4. 自身速度 [3] - 范围: [0, +∞)
    # 假设最大速度为20 m/s
    normalized[3] = np.tanh(state[3] / 20.0)  # 使用tanh限制在[-1,1]
    
    # 5. 目标速度 [4] - 范围: [0, +∞)
    normalized[4] = np.tanh(state[4] / 20.0)  # 使用tanh限制在[-1,1]
    
    # 6. 航向差余弦 [5] - 范围: [-1, 1]，已经是归一化的
    # 不需要额外处理
    
    # 7. 航向差正弦 [6] - 范围: [-1, 1]，已经是归一化的
    # 不需要额外处理
    
    # 8. 接近率 [7] - 范围: (-∞, +∞)
    # 接近率可能很大，使用tanh压缩
    normalized[7] = np.tanh(state[7] / 20.0)  # 除以20进行缩放
    
    # 9. 相对速度x [8] - 范围: (-∞, +∞)
    normalized[8] = np.tanh(state[8] / 20.0)  # 假设最大相对速度20 m/s
    
    # 10. 相对速度y [9] - 范围: (-∞, +∞)
    normalized[9] = np.tanh(state[9] / 20.0)  # 假设最大相对速度20 m/s
    
    # 11. 归一化相对位置x [10] - 范围: [-1, 1]，已经是归一化的
    # 确保在有效范围内
    normalized[10] = np.clip(state[10], -1.0, 1.0)
    
    # 12. 归一化相对位置y [11] - 范围: [-1, 1]，已经是归一化的
    normalized[11] = np.clip(state[11], -1.0, 1.0)
    
    return normalized

def parse_Input(action) -> str:
    
    try:
        # 直接提取原始动作值
        raw_velocity = float(action[0])  # PPO输出的原始速度值
        raw_direction = float(action[1])  # PPO输出的原始方向值
        
        # 在这里进行缩放
        # 假设PPO输出范围是 [-1, 1]，缩放到目标范围
        velocity = (raw_velocity + 1) * 10  # [-1,1] -> [0,20]
        direction = raw_direction * np.pi   # [-1,1] -> [-π, π]
        # 确保在合理范围内
        velocity = max(0, min(20, velocity))
        direction = max(-np.pi, min(np.pi, direction))
        
        velocity_str = f"{velocity:.4f}"
        direction_str = f"{direction:.4f}"
        
        cmd = f"<c><targetVel><float>{velocity_str}</float></targetVel><targetDir><float>{direction_str}</float></targetDir></c>"
        
        return cmd
        
    except:
        # 出错时返回零动作
        return "<c><targetVel><float>0.0</float></targetVel><targetDir><float>0.0</float></targetDir></c>"

def cal_Termination(state:Dict[str, any]) -> bool:
    #example
    if state['carInfo'][0]['baseInfo']['side'] == 1:
        self_pos = state['carInfo'][0]['position']
        enemy_pos = state['carInfo'][1]['position']
    elif state['carInfo'][1]['baseInfo']['side'] == 2:
        self_pos = state['carInfo'][1]['position']
        enemy_pos = state['carInfo'][0]['position']
    
    try:
        # 获取坐标值
        x1 = self_pos['x']
        y1 = self_pos['y']
        z1 = self_pos['z']
        x2 = enemy_pos['x']
        y2 = enemy_pos['y']
        z2 = enemy_pos['z']
        # 计算距离
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    except (KeyError, ValueError, TypeError) as e:
        # 如果数据格式有问题，不终止
        print(f"计算距离时出错: {e}")
        return False
    
    # 距离小于10米，终止
    if distance < 50:
        return True
    else:
        return False

def cal_reward_v1_5(state: dict) -> float:
    global prev_distance

    if 'carInfo' not in state or len(state['carInfo']) < 2:
        return 0.0

    # 区分敌我
    if state['carInfo'][0]['baseInfo']['side'] == 1:
        self_car = state['carInfo'][0]
        enemy_car = state['carInfo'][1]
    else:
        self_car = state['carInfo'][1]
        enemy_car = state['carInfo'][0]

    try:
        sx, sy, sz = self_car['position']['x'], self_car['position']['y'], self_car['position']['z']
        ex, ey, ez = enemy_car['position']['x'], enemy_car['position']['y'], enemy_car['position']['z']
    except KeyError:
        return 0.0

    dx, dy, dz = ex - sx, ey - sy, ez - sz
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)

    if 'prev_distance' not in globals():
        prev_distance = distance

    # ========== 核心奖励 ==========
    delta = prev_distance - distance

    # 非常重要：放大
    reward = 0.05 * delta    # 每接近 1 m +0.05

    # 防止原地摆烂
    reward -= 0.001          # 每步轻惩罚

    # 终止奖励
    if distance < 50:
        reward += 500.0

    prev_distance = distance
    return float(reward)
