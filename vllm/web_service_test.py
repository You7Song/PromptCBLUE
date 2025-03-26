# coding=utf-8
# Created by Michael Zhu
# DataSelect AI, 2023

import json
import time

import urllib.request

import sys
sys.path.append("./")
prefix = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
tail = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
def get_prompt_llama3_chinese(
    prompt: str,
) -> str:
    return prefix + prompt + tail


def test_service(prompt):
    header = {'Content-Type': 'application/json'}

    prompt = get_prompt_llama3_chinese(prompt)

    data = {
          "prompt": prompt,
          "stream" : False,
          "n" : 1,
          "best_of": 1, 
          "presence_penalty": 0.0, 
          "frequency_penalty": 0.2, 
          "temperature": 0.3, 
          "top_p" : 0.95, 
          "top_k": 50, 
          "beam_width": 4,
        #   "use_beam_search": False, 
          "stop": ["<|end_of_text|>", "<|eot_id|>"], 
          "ignore_eos" :False, 
          "max_tokens": 500, 
          "logprobs": None
    }
    request = urllib.request.Request(
        url='http://127.0.0.1:8090/generate',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )

    result = None
    try:
        response = urllib.request.urlopen(request, timeout=300)
        res = response.read().decode('utf-8')
        result = json.loads(res)
        # print(json.dumps(data, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(e)

    return result


if __name__ == "__main__":

    f_out = open("./vllm/output.json", "a", encoding="utf-8", buffering=1)
    with open("./datasets/PromptCBLUE/toy_examples/dev.json", "r", encoding="utf-8") as f:
        
        cnt = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = json.loads(line)

            t0 = time.time()
            result = test_service(line["input"])
            t1 = time.time()
            print("time cost: ", t1 - t0)

            f_out.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            
            cnt += 1
            if cnt >= 3:
                break
