from utils import get_completion_from_messages
import load_arxiv
import pandas as pd
def main():
    uc = load_arxiv.get_raw_text_arxiv(use_text=True)
    # messages = [
    #     {'role': 'system', 'content': 'You are a helpful assistant.'},
    #     {'role': 'user', 'content': '你是谁？'}]
    #
    # response = get_completion_from_messages(messages, model="qwen-turbo")
    # print(response)
    # different_prompt_try()


if __name__ == '__main__':
    main()