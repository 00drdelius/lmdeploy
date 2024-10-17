from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import time
import base64
import time
 
client = OpenAI(
    api_key = "whatever u like",
    base_url = "http://localhost:23333/v1",
)

def encode_image(img_path):
    with open(img_path,'rb') as img:
        b64_encoded = base64.b64encode(img.read()).decode("utf8")
    return "data:image/jpeg;base64,{}".format(b64_encoded)

img_path="/home/gmcc/workspace/imagerecognition/test/results/1.png"
usr_p = """你是一名优秀的AI图像识别助手。
我会传给你一个身份证证照，请你按照给你的要求给出我需要的信息。
我需要的信息名列表：
[姓名，性别，民族，出生，住址，公民身份证号码，签发机关，有效期限]
要求：
1. 先列出图中未提及的信息名列表和提及的信息名列表
2. 对于未提及的信息名，在json字段中返回"未提及"；对于提及的信息名，在json字段中返回对应信息
注意：有些文字之间可能非常相似。如："5"跟"S"，"8"跟"B"，"Z"跟"2"。请你再三思考再做决定
返回模板：
图中未提及的信息名：[]
图中提及的信息名：[]
```json
{
  "姓名":"姓名",
  "性别":"性别",
  "民族":"民族",
  "出生":"出生",
  "住址":"住址",
  "公民身份证号码":"公民身份证号码",
  "签发机关":"签发机关",
  "有效期限":"有效期限"
}
```"""
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": usr_p,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image(img_path)
                },
            },
        ],
    }
]
temperature=0.0
max_tokens=512

start=time.time() 

def single():
    completion = client.chat.completions.create(
        model = "glm-4v-9b",
        messages = messages,
        temperature = temperature,
        max_tokens=max_tokens
    )
    print(completion.choices[0].message.content)

def conc():
    workers=6
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = [executor.submit(single) for _ in range(workers)]
        for fu in tasks:
            fu.result()

single()
print("\nTime consumed: ",time.time()-start)
