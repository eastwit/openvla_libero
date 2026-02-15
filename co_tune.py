import os
# 环境与镜像设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import imageio
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import sys

# ==========================================
# 工具函数：对齐微调时的 90% Center Crop
# ==========================================
def get_openvla_input(raw_image):
    """
    1. 修正翻转
    2. 取中心 90% 区域 (这是微调时的规范)
    """
    # 修正 LIBERO 渲染颠倒
    corrected = np.flip(raw_image, axis=0)
    img = Image.fromarray(corrected.astype(np.uint8))
    
    width, height = img.size
    # 计算 90% 面积对应的边长比例 (约 0.9487)
    scale = 0.9487 
    new_w, new_h = int(width * scale), int(height * scale)
    
    left = (width - new_w) / 2
    top = (height - new_h) / 2
    right = (width + new_w) / 2
    bottom = (height + new_h) / 2
    
    # 裁剪并 Resize 到模型标准的 224
    input_pil = img.crop((left, top, right, bottom)).resize((224, 224), Image.LANCZOS)
    return input_pil, corrected

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
    VIDEO_PATH = "libero_spatial_optimized.mp4"
    
    # 1. 加载模型 (对齐 4-bit 和特定统计量)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto",
        load_in_4bit=True, 
        trust_remote_code=True
    )

    # 2. 环境初始化 (LIBERO-Spatial)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    TASK_ID = 1 # 你可以更换 ID
    task = task_suite.get_task(TASK_ID)
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview"],
        control_freq=20,
    )
    
    obs = env.reset()
    env.set_init_state(task_suite.get_task_init_states(TASK_ID)[0])

    # 3. 推理循环
    prompt = f"In: What action should the robot take to {task.language}?\nOut:"
    writer = imageio.get_writer(VIDEO_PATH, fps=20, format='FFMPEG', mode='I')

    print(f"🚀 正在执行对齐后的任务: {task.language}")

    try:
        for step in range(600):
            # 获取 90% Crop 后的输入
            input_pil, render_frame = get_openvla_input(obs['agentview_image'])
            
            with torch.inference_mode():
                prompt="In: What action should the robot take to {open the draw}?\nOut:"
                inputs = processor(prompt, input_pil).to("cuda", dtype=torch.float16)
                # 【关键】使用笔记中确定的 libero_spatial 统计量
                action = vla.predict_action(**inputs, unnorm_key="libero_spatial")

            # 动作执行
            scaled_action = action.astype(np.float64)
            # 夹爪逻辑对齐
            scaled_action[-1] = 1.0 if action[-1] > 0.5 else -1.0
            
            obs, reward, done, info = env.step(scaled_action)
            writer.append_data(render_frame)

            if step % 20 == 0: print(f"Step {step}...")
            if done: break

    finally:
        writer.close()
        env.close()
        print(f"✨ 录制完成: {VIDEO_PATH}")