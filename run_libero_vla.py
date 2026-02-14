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
# 解决渲染器 GLEW 加载问题（Ubuntu 常见问题）
glew_paths = glob.glob("/usr/lib/x86_64-linux-gnu/libGLEW.so*")
if glew_paths: 
    os.environ["LD_PRELOAD"] = glew_paths[-1]

# 使用镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("--- 正在初始化 LIBERO 环境 ---")
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    print("✅ LIBERO 导入成功！")
except Exception as e:
    print(f"❌ 导入失败: {e}"); sys.exit()

# ==========================================
# 2. OpenVLA 模型加载 (极致显存优化)
# ==========================================
model_id = "openvla/openvla-7b"
print(f"--- 正在加载 OpenVLA 模型 (4-bit 模式) ---")

# 配置 4-bit 量化以适应 8GB 显存
q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # 4060 支持 bf16，速度更快
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 修复核心：显式指定 device_map 避免 transformers 内部 .to() 报错
vla = AutoModelForVision2Seq.from_pretrained(
    model_id, 
    quantization_config=q_config,
    low_cpu_mem_usage=True, 
    trust_remote_code=True,
    device_map="cuda:0" 
)

# 显存优化：禁用推理缓存
vla.config.use_cache = False 
print(f"✅ 模型加载成功！当前显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ==========================================
# 3. LIBERO 仿真环境启动 (API 适配版)
# ==========================================
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]() 
task_id = 0 

# 根据你的调试信息，Task 是个 NamedTuple，包含 bddl_file 等路径
task = task_suite.get_task(task_id)
task_description = task.language 
print(f"--- 任务指令: {task_description} ---")

# 构造环境参数 (适配你的 Task 对象属性)
env_args = {
    "bddl_file_name": task.bddl_file,
    "camera_height": 224, # OpenVLA 标准输入分辨率
    "camera_width": 224,
    "device_id": 0       # 明确指定渲染用的显卡 ID
}

# 如果你的 LIBERO 版本支持从初始状态文件加载
if hasattr(task, 'init_states_file'):
    env_args["initial_state_path"] = task.init_states_file

print("--- 正在创建仿真环境 ---")
env = OffScreenRenderEnv(**env_args)
obs = env.reset()

# ==========================================
# 4. 闭环控制推理循环
# ==========================================
print("--- 机器人控制正式开始 ---")
try:
    for step in range(200):
        # 图像处理：LIBERO 的 agentview 图像通常需要垂直翻转
        img_np = obs["agentview_image"]
        img_np = np.flipud(img_np) 
        img = Image.fromarray(img_np)
        
        # 准备模型输入
        inputs = processor(task_description, img, return_tensors="pt").to("cuda:0", dtype=torch.bfloat16)
        
        with torch.inference_mode():
            # OpenVLA 推理动作 (7维向量: [x, y, z, roll, pitch, yaw, gripper])
            action = vla.predict_action(**inputs, unnorm_key="bridge_orig")
        
        # 动作执行
        if hasattr(action, 'cpu'):
            action = action.cpu().numpy()
            
        obs, reward, done, info = env.step(action)
        
        # 每 10 步打印一次状态并保存图片
        if step % 10 == 0:
            vram = torch.cuda.memory_allocated()/1024**3
            print(f"Step {step}: AI 正在操控... 显存: {vram:.2f} GB")
            img.save(f"step_{step}.png")
            # 定期清理显存碎片防止 8GB 溢出
            torch.cuda.empty_cache()
            
        if done or reward > 0: 
            print("🎉 任务目标达成或环境终止！")
            break
            
except Exception as e:
    print(f"❌ 运行中出错: {e}")
finally:
    if 'env' in locals(): 
        env.close()
    print("--- 仿真结束并安全退出 ---")