import torch
import numpy as np
import imageio
import os
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import try_to_load_from_cache
from PIL import Image

# LIBERO 相关导入
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path

# 屏蔽 Tokenizer 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 解决 PyTorch 2.6+ 安全检查问题
try:
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
except AttributeError:
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# ==========================================
# 1. 模型加载函数 (4-bit 优化)
# ==========================================
def load_vla(img_path, model_id):
    os.environ["HF_ENDPOINT"] = img_path
    filepath = try_to_load_from_cache(model_id, "config.json")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        print("--- OpenVLA (4-bit) 加载成功！ ---")
        return vla, processor
    except Exception as e:
        print(f"加载失败: {e}")
        return None, None

# ==========================================
# 2. LIBERO 环境配置
# ==========================================
def setup_libero_env(task_suite_name, task_id):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[任务] {task.name} | [指令] {task.language}")

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_names": ["agentview"], 
        "reward_shaping": True,
        "control_freq": 20,
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])
    return env, task.language

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "openvla/openvla-7b"
    HF_MIRROR = "https://hf-mirror.com"
    VIDEO_PATH = "libero_openvla_demo.mp4"
    MAX_STEPS = 2000 
    ACTION_SCALE = 4.0  # 建议从 4.0 开始尝试，10.0 有点太大了

    vla, processor = load_vla(HF_MIRROR, MODEL_ID)
    env, prompt = setup_libero_env("libero_10", 0)
    
    # 强制使用 ffmpeg 写入，避免 Tiff 错误
    writer = imageio.get_writer(VIDEO_PATH, fps=20, format='FFMPEG', mode='I')

    obs = env.reset()

    print("🚀 启动控制循环...")
    try:
        for step in range(MAX_STEPS):
            # --- 修正视觉输入 ---
            # 针对截图中的颠倒问题，我们直接使用 np.flip 进行更彻底的翻转
            raw_image = obs['agentview_image']
            # 这种翻转方式确保画面底座在下，物体在上
            corrected_image = np.flip(raw_image, axis=0) 
            
            input_pil = Image.fromarray(corrected_image.astype(np.uint8))

            # --- VLA 推理 ---
            with torch.inference_mode():
                inputs = processor(prompt, input_pil).to("cuda", dtype=torch.float16)
                action = vla.predict_action(**inputs, unnorm_key="bridge_orig")

            # --- 动作缩放与执行 ---
            # 10倍可能太猛，这里用 ACTION_SCALE 控制
            scaled_action = action.astype(np.float64) * ACTION_SCALE
            # 夹爪动作 (最后一维) 通常不需要缩放，保持在原范围
            scaled_action[-1] = action[-1] 
            
            obs, reward, done, info = env.step(scaled_action)

            # --- 保存视频帧 ---
            writer.append_data(corrected_image)

            if step % 10 == 0:
                print(f"Step {step}/{MAX_STEPS} | 动作执行中...")
            
            if step % 5 == 0:
                torch.cuda.empty_cache()

            if done:
                print("🏁 任务完成！")
                break

    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        writer.close()
        env.close()
        print(f"✨ 视频已保存至: {os.path.abspath(VIDEO_PATH)}")