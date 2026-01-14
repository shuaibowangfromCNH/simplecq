from toolswsb import parse_Output, process_raw_state, parse_Input, cal_reward_v1_5, cal_Termination
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PPO import PPO
from gym_interface import Agent, State

def compute_distance(state_dict):
    if 'carInfo' not in state_dict or len(state_dict['carInfo']) < 2:
        return None

    if state_dict['carInfo'][0]['baseInfo']['side'] == 1:
        self_car = state_dict['carInfo'][0]
        enemy_car = state_dict['carInfo'][1]
    else:
        self_car = state_dict['carInfo'][1]
        enemy_car = state_dict['carInfo'][0]

    dx = enemy_car['position']['x'] - self_car['position']['x']
    dy = enemy_car['position']['y'] - self_car['position']['y']
    dz = enemy_car['position']['z'] - self_car['position']['z']
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def scale_action(raw_action):
    raw_v, raw_d = float(raw_action[0]), float(raw_action[1])
    vel = (raw_v + 1.0) * 10.0      # -> [0, 20]
    direc = raw_d * math.pi         # -> [-pi, pi]
    return vel, direc

def run_one_episode(env, ppo_agent, decision_interval=10, max_steps=20000, deterministic=True):
    self_x, self_y, enemy_x, enemy_y = [], [], [], []
    raw_state = env.reset()
    state_dict = parse_Output(raw_state) if not isinstance(raw_state, dict) else raw_state
    log_state, current_state = process_raw_state(state_dict)

    raw_action = np.array([0.0, 0.0], dtype=np.float32)

    distances = []
    v_cmds = []
    dir_cmds = []
    raw_vs = []
    raw_ds = []

    for step in range(max_steps):
        if step % decision_interval == 0:
            # 用你新增的 act()（强烈推荐）
            raw_action = ppo_agent.act(current_state, deterministic=deterministic)

        next_raw_state, reward, done, _ = env.step(raw_action)
        if next_raw_state == '':
            continue

        state_dict = next_raw_state
        xy = get_xy(state_dict)
        if xy is not None:
            sx, sy, ex, ey = xy
            self_x.append(sx); self_y.append(sy)
            enemy_x.append(ex); enemy_y.append(ey)

        d = compute_distance(state_dict)
        if d is not None:
            distances.append(d)

        rv, rd = float(raw_action[0]), float(raw_action[1])
        vel, direc = scale_action(raw_action)

        raw_vs.append(rv); raw_ds.append(rd)
        v_cmds.append(vel); dir_cmds.append(direc)

        _, current_state = process_raw_state(state_dict)

        if done:
            print(f"[DONE] step={step}, distance={distances[-1] if len(distances) else None}, v={v_cmds[-1] if len(v_cmds) else None}, dir={dir_cmds[-1] if len(dir_cmds) else None}")
            break

    success = (len(distances) > 0 and distances[-1] < 50.0)
    return {
        "success": success,
        "distances": np.array(distances),
        "v_cmds": np.array(v_cmds),
        "dir_cmds": np.array(dir_cmds),
        "raw_vs": np.array(raw_vs),
        "raw_ds": np.array(raw_ds),
        "self_x": np.array(self_x),
        "self_y": np.array(self_y),
        "enemy_x": np.array(enemy_x),
        "enemy_y": np.array(enemy_y),
    }

def moving_avg(x, w=50):
    if len(x) < w: 
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid')

def get_xy(state_dict):
    if 'carInfo' not in state_dict or len(state_dict['carInfo']) < 2:
        return None

    if state_dict['carInfo'][0]['baseInfo']['side'] == 1:
        self_car = state_dict['carInfo'][0]
        enemy_car = state_dict['carInfo'][1]
    else:
        self_car = state_dict['carInfo'][1]
        enemy_car = state_dict['carInfo'][0]

    sx = self_car['position']['x']; sy = self_car['position']['y']
    ex = enemy_car['position']['x']; ey = enemy_car['position']['y']
    return sx, sy, ex, ey

def plot_2d(traj, title="episode"):
    plt.figure()
    plt.plot(traj["self_x"], traj["self_y"], label="self")
    plt.plot(traj["enemy_x"], traj["enemy_y"], label="enemy")
    plt.scatter(traj["self_x"][0], traj["self_y"][0], marker="o", label="self start")
    plt.scatter(traj["enemy_x"][0], traj["enemy_y"][0], marker="x", label="enemy start")
    plt.axis("equal")
    plt.grid(True)
    plt.title(f"{title} - 2D trajectory (XY)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def plot_episode(traj, title="episode"):
    d = traj["distances"]
    v = traj["v_cmds"]
    direc = traj["dir_cmds"]

    plt.figure()
    plt.plot(d)
    plt.title(f"{title} - distance")
    plt.xlabel("step")
    plt.ylabel("distance (m)")
    plt.grid(True)

    plt.figure()
    plt.plot(v)
    plt.title(f"{title} - target speed")
    plt.xlabel("step")
    plt.ylabel("speed cmd")
    plt.grid(True)

    plt.figure()
    plt.plot(direc)
    plt.title(f"{title} - target direction")
    plt.xlabel("step")
    plt.ylabel("direction (rad)")
    plt.grid(True)

    plt.figure()
    plt.plot(d, alpha=0.4, label="raw")
    plt.plot(moving_avg(d, 50), label="ma50")
    plt.legend()

    plt.show()

def main():
    ckpt = r"D:/codeproject/results/save/ppo_best_171目标不动.pth"
    port = 40029
    outputs_type = {
        "targetDir": "float",
        "targetVel": "float"
    }
    # TODO: 初始化 env + PPO（参数必须和训练一致）
    env = Agent(port=port, 
              outputs_type=outputs_type,
              process_input=parse_Input,
              process_output=parse_Output,
              reward_func=cal_reward_v1_5,
              end_func=cal_Termination)
    
    # PPO参数设置
    state_dim = 12  # 状态维度
    action_dim = 2  # 动作维度
    lr_actor = 3e-4
    lr_critic = 3e-4  # Critic学习率
    gamma = 0.995  # 折扣因子
    K_epochs = 10  # 每次更新的训练轮数
    eps_clipping = 0.2  # PPO裁剪参数
    is_continuous_action_space = True  # 连续动作空间
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clipping=eps_clipping,
        is_continuous_action_space=is_continuous_action_space
    )

    ppo_agent.load(ckpt)

    # 推理用 eval 模式（避免 dropout/bn）
    ppo_agent.policy.eval()
    ppo_agent.policy_old.eval()

    results = []
    for i in range(3):
        traj = run_one_episode(env, ppo_agent, decision_interval=10, deterministic=True)
        results.append(traj)
        print(f"episode {i}: success={traj['success']}, final_d={traj['distances'][-1] if len(traj['distances']) else None}")
        plot_episode(traj, title=f"episode {i}")
        plot_2d(traj, title=f"episode {i}")


    sr = sum(r["success"] for r in results) / len(results)
    print(f"success rate={sr:.2f}")

if __name__ == "__main__":
    main()
