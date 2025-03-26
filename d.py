from huggingface_hub import snapshot_download  

# 指定模型ID和本地保存路径  
model_id = "FlagAlpha/Llama3-Chinese-8B-Instruct"  
local_dir = "./models/llama"  

# 使用snapshot_download下载模型仓库的完整副本  
snapshot_download(repo_id=model_id, local_dir=local_dir)  