import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = '../models/12B'


def main():
    # 加载AICHI2LM模型
    tokenizer = AutoTokenizer.from_pretrained(PATH,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, device_map="auto",
                                                 torch_dtype=torch.float16)
    generate_config = GenerationConfig.from_pretrained(PATH)
    model.eval()

    # AICHI2LM多轮对话演示
    print("*" * 10 + "AICHI2LM多轮输入演示" + "*" * 10)
    question = "你是谁？"
    print("提问:", question)
    answer, history = model.chat(tokenizer = tokenizer, question=question, history=[], generation_config=generate_config,
                                 stream=False)
    print("回答:", answer)
    print("截至目前的聊天记录是:", history)

    question = "你是谁训练的"
    print("提问:", question)
    # 将history传入
    answer, history = model.chat(tokenizer, question=question, history=history, generation_config=generate_config,
                                 stream=False)
    print("回答是:", answer)
    print("截至目前的聊天记录是:", history)

    # 也可以这么调用传入history
    history = [
        {"role": "user", "content": "你是谁"},
        {"role": "bot", "content": "我是AICHI2LM"},
    ]

    question = "你是谁训练的"
    print("提问:", question)
    answer, history = model.chat(tokenizer, question=question, history=history, generation_config=generate_config,
                                 stream=False)
    print("回答是:", answer)
    print("截至目前的聊天记录是:", history)

    # AICHI2LM流式返回演示
    print("*" * 10 + "AICHI2LM流式输入演示" + "*" * 10)
    question = "你是谁？"
    print("提问:", question)
    gen = model.chat(tokenizer, question=question, history=[], generation_config=generate_config,
                     stream=True)
    for answer, history in gen:
        print("回答是:", answer)
        print("截至目前的聊天记录是:", history)


if __name__ == '__main__':
    main()
