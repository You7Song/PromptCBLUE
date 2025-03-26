import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "./models/llama"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda:0",
    torch_dtype=torch.float16,
)


messages = [{"role": "system", "content": ""}]

messages.append(
                {"role": "user", "content": "介绍一下机器学习"}
            )

prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
print(repr(prompt))

terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
print(tokenizer.eos_token)

# output = model.generate(
#     **tokenizer(prompt, return_tensors="pt").to("cuda:1"),
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.9,
#     top_p=0.9,
#     top_k=40,
#     repetition_penalty=1.0,
#     num_return_sequences=1,
#     # eos_token_id=tokenizer.eos_token_id,
#     eos_token_id=terminators,
# )
# # print(list(output[0]))

# print(tokenizer.decode(output[0]))