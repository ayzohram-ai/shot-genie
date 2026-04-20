import openai
import re
import argparse
from airsim_wrapper import *
import math
import numpy as np
import os
import json
import time
import requests # sony
import subprocess
import base64

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="./multimodal/prompts/airsim_basic.txt")
parser.add_argument("--sysprompt", type=str, default="./multimodal/system_prompts/airsim_basic.txt")
args = parser.parse_args()

with open("./multimodal/config.json", "r") as f:
    config = json.load(f)

print("Initializing ChatGPT...")
# openai.api_key = config["OPENAI_API_KEY"]



with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

chat_history = [
    {
        "role": "system",
        "content": sysprompt
    },
    {
        "role": "user",
        "content": "move 10 units up"
    },
    {
        "role": "assistant",
        "content": """```python
aw.fly_to([aw.get_drone_position()[0], aw.get_drone_position()[1], aw.get_drone_position()[2]+10])
```

This code uses the `fly_to()` function to move the drone to a new position that is 10 units up from the current position. It does this by getting the current position of the drone using `get_drone_position()` and then creating a new list with the same X and Y coordinates, but with the Z coordinate increased by 10. The drone will then fly to this new position using `fly_to()`."""
    }
]


def ask(prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"  # sony
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['OPENAI_API_KEY']}"
    }  # sony
    data = {
        "model": "gpt-3.5-turbo",  # 和你原本一样
        "messages": chat_history,  # 和你原本一样
        "temperature": 0           # 和你原本一样
    }  # sony

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        completion = response.json()

        # Debugging: Log the response structure
        print("API Response:", completion)

        # Ensure the response contains the expected fields
        if "choices" in completion and len(completion["choices"]) > 0:
            assistant_content = completion["choices"][0]["message"]["content"]
        else:
            assistant_content = "API响应格式不正确，未找到'choices'字段。"  # sony

    except Exception as e:
        assistant_content = f"API请求失败: {str(e)}"  # sony

    chat_history.append(
        {
            "role": "assistant",
            "content": assistant_content,
        }
    )
    return chat_history[-1]["content"]


print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)


def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None


class colors:  # You may need to change color settings
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


print(f"Initializing AirSim...")
aw = AirSimWrapper()
print(f"Done.")

with open(args.prompt, "r") as f:
    prompt = f.read()



# ========= 新增：对外暴露的执行函数（最小侵入） =========
def llm_exec(text: str) -> str:
    """
    传入自然语言命令：
      - 调用 ask(text) 与 LLM 交互
      - 从回复中提取 ```python ...``` 代码块
      - 在本模块已初始化的 AirSimWrapper 实例 `aw` 上执行代码
    返回：LLM 原文回复（可用于上层日志/界面显示）
    """
    try:
        reply = ask(text)  # 走你原有的 chat_history / API 路径
    except Exception as e:
        return f"[intent_server.llm_exec][ERR] ask() failed: {e}"

    code = extract_python_code(reply)
    if code:
        print("Please wait while I run the code in AirSim...")
        # 只暴露必要对象，避免污染；aw 是关键
        _globals = {
            "aw": aw,
            "np": np,
            "math": math,
            "time": time,
            "os": os,
            "json": json,
            "base64": base64,
            # 如你的代码块会用到 airsim_wrapper 里的工具，可按需补充
        }
        try:
            exec(code, _globals, {})
            print("Done!\n")
        except Exception as ex:
            print(f"[intent_server.llm_exec][EXEC ERROR] {ex}")
            # 不中断外层调用，返回带错误的回复，便于上层记录
            return f"{reply}\n\n[EXEC ERROR] {ex}"
    else:
        print("[intent_server.llm_exec] no code block found in reply.")
    return reply


# （保持你前面的初始化/定义不变）
# ...
# 原有的 REPL 循环改成：
if __name__ == "__main__":
    ask(prompt)
    print("Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")

    while True:
        question = input(colors.YELLOW + "AirSim> " + colors.ENDC)
        if question == "!quit" or question == "!exit":
            break
        if question == "!clear":
            os.system("cls")
            continue

        response = ask(question)
        print(f"\n{response}\n")
        code = extract_python_code(response)
        if code is not None:
            print("Please wait while I run the code in AirSim...")
            exec(code, {"aw": aw, "np": np, "math": math, "time": time, "os": os, "json": json, "base64": base64}, {})
            print("Done!\n")
