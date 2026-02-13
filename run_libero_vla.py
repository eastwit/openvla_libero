import os
import sys
import torch
import numpy as np
from PIL import Image
import glob
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# ==========================================
# 1. 环境修复与路径初始化
# ==========================================
# 解决渲染器 GLEW 加载问题
glew_paths = glob.glob("/usr/lib/x86_64-linux-gnu/libGLEW.so*")
if glew_paths: 
    os.environ["LD_PRELOAD"] = glew_paths[-1]

# 使用镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("--- 正在初始化 LIBERO 环境 ---")
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.file_utils import get_bddl_path
    print("✅ LIBERO 导入成功！")
except Exception as e:
    print(f"❌ 导入失败: {e}"); sys.exit()

# ==========================================
# 2. OpenVLA 模型加载 (4-bit 极致优化)
# ==========================================
model_id = "openvla/openvla-7b"
print(f"--- 正在加载 OpenVLA 模型 (4-bit 模式) ---")

q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_id, 
    quantization_config=q_config,
    low_cpu_mem_usage=True, 
    trust_remote_code=True,
    device_map="cuda:0" 
)

# 显存优化
vla.config.use_cache = False 
print(f"✅ 模型加载成功！当前显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ==========================================
# 3. LIBERO 仿真环境启动 (修复 BDDL 路径)
# ==========================================
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]() 
task_id = 0 

task = task_suite.get_task(task_id)
task_description = task.language 
print(f"--- 任务指令: {task_description} ---")

# 关键修复：获取 BDDL 文件的绝对路径
actual_bddl_path = get_bddl_path(task.bddl_file)
print(f"📍 正在定位 BDDL: {actual_bddl_path}")

env_args = {
    "bddl_file_name": actual_bddl_path,
    "camera_height": 224,
    "camera_width": 224,
    "device_id": 0
}

print("--- 正在创建仿真环境 ---")
env = OffScreenRenderEnv(**env_args)
obs = env.reset()

# ==========================================
# 4. 闭环控制推理循环
# ==========================================
print("--- 机器人控制正式开始 ---")
try:
    for step in range(200):
        # 1. 图像预处理：翻转 + RGB 转换
        img_np = obs["agentview_image"]
        img_np = np.flipud(img_np) 
        img = Image.fromarray(img_np).convert("RGB")
        
        # 2. 准备模型输入
        inputs = processor(task_description, img, return_tensors="pt").to("cuda:0", dtype=torch.bfloat16)
        
        # 3. OpenVLA 推理动作
        with torch.inference_mode():
            # 显式指定输入字段，确保稳定性
            action = vla.predict_action(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                unnorm_key="bridge_orig"
            )
        
        # 4. 动作后处理
        if hasattr(action, 'cpu'):
            action = action.cpu().numpy()
        
        # 限制动作范围，防止仿真器因异常数值崩溃
        action = np.clip(action, -1.0, 1.0)
            
        # 5. 执行步进
        obs, reward, done, info = env.step(action)
        
        # 6. 状态反馈与日志
        if step % 5 == 0:
            vram = torch.cuda.memory_allocated()/1024**3
            print(f"Step {step}: 正在执行... 显存: {vram:.2f} GB | Reward: {reward}")
            img.save(f"step_{step}.png")
            # 释放缓存防止碎片化 OOM
            torch.cuda.empty_cache()
            
        if done or reward > 0: 
            print("🎉 任务成功达成！")
            break
            
except Exception as e:
    print(f"❌ 运行中出错: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'env' in locals(): 
        env.close()
    print("--- 仿真结束并安全退出 ---")