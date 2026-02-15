"""这是Gemini写的一段可视化代码，其中的轨迹是随机生成的"""
import os
import numpy as np
import imageio
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path

def main():
    # 1. 初始化 Benchmark 和 任务
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10" 
    task_suite = benchmark_dict[task_suite_name]()

    # 获取第一个任务
    task_id = 0
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    print(f"[验证] 任务名称: {task_name}")
    print(f"[指令] 语言描述: {task_description}")

    # 2. 配置环境 (增加分辨率以获得更好的视觉效果)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256, # 增加分辨率
        "camera_widths": 256,
        "camera_names": ["agentview", "robot0_eye_in_hand"], # 同时开启两个视角
        "reward_shaping": True,
        "control_freq": 20,
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()

    # 设置预设的初始状态
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])

    # 3. 准备视频写入
    video_path = "libero_visualization.mp4"
    writer = imageio.get_writer(video_path, fps=20)
    print(f"开始渲染并收集图像至 {video_path}...")

    # 4. 运行环境并捕获帧
    for step in range(300):
        # 生成一个随机的微小动作，让机械臂看起来在动
        # 动作空间通常是 7 维: [x, y, z, roll, pitch, yaw, gripper]
        action = np.array([0.2,0.2,0.2,0.1,0.1,0.1,0.01])
        
        obs, reward, done, info = env.step(action)

        # 获取第三人称视角图像
        # 重点：Robosuite 渲染的图像坐标系在 Y 轴上是反的，需要用 [::-1] 翻转
        frame = obs['agentview_image'][::-1, :, :]
        
        # 将帧写入视频
        writer.append_data(frame)
        
        if step % 10 == 0:
            print(f"正在渲染第 {step} 步...")

    # 5. 清理
    writer.close()
    env.close()
    print(f"渲染完成！请查看当前目录下的 {video_path}")

if __name__ == "__main__":
    main()