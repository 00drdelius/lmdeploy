from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
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

img_path=Path(__file__).parent.joinpath("jinzhe.jpg")
usr_p = """
这是一名游戏角色，
请你识别出这个游戏名称与这个角色名称，并返回给我json格式如下：
```json
{"游戏":"","角色":""}
```
注意：
1. 若是有字段不知道，可以填“不知道”。
"""
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
                    "url": encode_image(str(img_path))
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
