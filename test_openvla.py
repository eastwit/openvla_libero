import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import try_to_load_from_cache
from PIL import Image
import os

# 确保镜像站开启
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "openvla/openvla-7b"

filepath = try_to_load_from_cache(model_id, "config.json")

if isinstance(filepath, str):
    print(f"✅ 模型已存在！本地缓存路径为: {os.path.dirname(filepath)}")
    print(f"正在尝试加载模型......")

else:
    print("❌ 本地未找到完整模型，即将开始下载......")


try:
    # 1. 加载处理器
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # 2. 加载模型 
    # 注意：初次运行会下载约 26GB 的权重文件！
    vla = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        device_map="auto" 
    )
    print("--- 模型加载成功！ ---")

    # 3. 测试推理
    prompt = "In LIBERO, pick up the red block."
    image = Image.new('RGB', (224, 224), color = (73, 109, 137))
    inputs = processor(prompt, image, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
    
    # 预测动作
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig")
    print(f"生成的动作向量：\n{action}")

except Exception as e:
    print(f"运行中遇到问题: {e}")