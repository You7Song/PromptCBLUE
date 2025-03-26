import os  
import shutil  

# 指定源目录和目标目录  
src_dir = './models/llama'         # 替换为您的源文件夹路径  
target_dir = './models/llama_lora'   # 替换为您的目标文件夹路径  

# 确保目标目录存在  
os.makedirs(target_dir, exist_ok=True)  

# 遍历源目录中的文件  
for item in os.listdir(src_dir):  
    item_path = os.path.join(src_dir, item)  
    
    # 检查是否是文件，非.safetensors和.bin文件  
    if os.path.isfile(item_path) and not (item.endswith('.safetensors') or item.endswith('.bin')):  
        # 复制到目标目录  
        shutil.copy(item_path, target_dir)  

# 输出目标目录下的文件列表  
print("目标目录文件:", os.listdir(target_dir))  


import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

print(f"Loading the base model from {src_dir}")
base_model = AutoModelForCausalLM.from_pretrained(
    src_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)


lora_weights_path = "./peft_load"  # 替换为实际路径  
print(f"Loading the LoRA adapter from {lora_weights_path}")
lora_model = PeftModel.from_pretrained(base_model, lora_weights_path, torch_dtype=torch.float16,)  

print("Applying the LoRA")
model = lora_model.merge_and_unload()


print(f"Saving the target model to {target_dir}")
model.save_pretrained(target_dir)