import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  
from peft import PeftModel, PeftConfig  

# 加载基础模型  
base_model_id = "./models/llama"  # 根据您使用的具体版本调整  
tokenizer = AutoTokenizer.from_pretrained(base_model_id)  
base_model = AutoModelForCausalLM.from_pretrained(  
    base_model_id,  
    torch_dtype=torch.float16,  # 可选，使用半精度加载  
    device_map="cpu"  
)  

print(base_model)

# 直接加载LoRA权重到基础模型上  
lora_weights_path = "./peft_load"  # 替换为实际路径  

model = PeftModel.from_pretrained(base_model, lora_weights_path)  

print("")
print("")
print(model)


# messages = [{"role": "system", "content": ""}]

# messages.append(
#                 {"role": "user", "content": "请问是什么意图类型？\n癫痫怎么治愈\n搜索意图选项：治疗方案，病情诊断，指标解读，病因分析，注意事项，功效作用，医疗费用"}
#             )

# prompt = tokenizer.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True
#     )




# terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

# # 现在可以直接使用应用了LoRA权重的模型进行推理  
# input_text = prompt  
# inputs = tokenizer(input_text, return_tensors="pt").to(model.device)  

# with torch.no_grad():  
#     outputs = model.generate(  
#         input_ids=inputs.input_ids,  
#         max_new_tokens=512,  
#         temperature=0.7,  
#         top_p=0.9  ,
#         eos_token_id=terminators,
#     )  

# response = tokenizer.decode(outputs[0], skip_special_tokens=True)  
# print(response)  