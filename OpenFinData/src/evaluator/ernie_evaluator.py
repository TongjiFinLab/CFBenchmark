from utils.ernie_utils import gpt_api


class ERNIEEvaluator:
    def __init__(self, model_type="eb4"):
        self.mode_type = model_type

    def answer(self, query, model_type="text-davinci-002"):
        return gpt_api(query, self.mode_type)


if __name__ == "__main__":
    evaluator = ERNIEEvaluator("eb4")
    print(evaluator.answer("你好"))
