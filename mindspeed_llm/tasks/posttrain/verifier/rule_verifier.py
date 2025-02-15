import re
from multiprocessing import Process, Queue

from mindspeed_llm.tasks.posttrain.verifier.math_eval_toolkit.parser import extract_answer
from mindspeed_llm.tasks.posttrain.verifier.math_eval_toolkit.grader import math_equal


def qwen_math_equal_subprocess(prediction, reference, timeout_seconds=10):

    def worker(q, prediction, reference):
        result = math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()
    
    p.join(timeout=timeout_seconds)
    
    if p.is_alive():
        p.terminate()
        p.join()  
        return False
        
    try:
        return q.get_nowait()
    except Exception as e:
        return False   


def preprocess_box_response_for_qwen_prompt(sequences, answers, **kwargs):
    scores = []

    for sequence, answer in zip(sequences, answers):
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', sequence, flags=re.DOTALL, count=1)
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
        for stop_word in stop_words:
            if stop_word in model_output:
                model_output = model_output.split(stop_word)[0].strip()
        ext_answer = extract_answer(model_output, data_name="math")

        if qwen_math_equal_subprocess(prediction=ext_answer, reference=answer):
            box_match = 1.0
        else:
            box_match = -0.5
            
        if "boxed" not in model_output:
            box_match = -1.0

        scores.append(box_match)
        
    return scores


def format_reward(sequences, **kwargs):
    """
    Reward function that checks if the completion has a specific format.

    Args:
        sequences: A list of sequences, where each completion is a tuple containing a list of dictionaries.
                     Each dictionary should have a "content" key with the text to be checked.

    Returns:
        A list of floats, where each float is 1.0 if the corresponding completion matches the required format,
        and 0.0 otherwise.

    Raises:
        ValueError: If the input sequences are not in the expected format.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    if not isinstance(sequences, list):
        raise ValueError("Input sequences must be a list.")

    rewards = []
    for completion in sequences:
        if re.match(pattern, completion, re.DOTALL | re.MULTILINE):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def reasoning_steps_reward(sequences, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    matches = [len(re.findall(pattern, content)) for content in sequences]

    return [min(1.0, count / 3) for count in matches]


def _test_rule_verifier():
    text = ["""<think>\nFirst, we use the property of a parallelogram that opposite sides are equal in length. 
    Therefore, we have:\n\\[AB = CD \\quad \\text{and} \\quad BC = AD\\]\n\nFrom the given measurements:\n\\
    [AB = 38 \\text{ cm}\\]\n\\[BC = 3y^3 \\text{ cm}\\]\n\\[CD = 2x + 4 \\text{ cm}\\]\n\\[AD = 24 \\text{ cm}\\]\n\n
    Setting \\(AB = CD\\), we get:\n\\[38 = 2x + 4\\]\n\nSetting \\(BC = AD\\), we get:\n\\[3y^3 = 24\\]\n\nNow, 
    we solve each equation for \\(x\\) and \\(y\\).\n</think>\n<answer>\nFirst, solve for \\(x\\):\n\\[38 = 2x + 4\\]
    \n\\[38 - 4 = 2x\\]\n\\[34 = 2x\\]\n\\[x = \\frac{34}{2} = 17\\]\n\nNext, solve for \\(y\\):\n\\[3y^3 = 24\\]\n\\
    [y^3 = \\frac{24}{3} = 8\\]\n\\[y = \\sqrt[3]{8} = 2\\]\n\nNow, find the product of \\(x\\) and \\(y\\):\n\\[xy = 
    17 \\times 2 = 34\\]\n\nTherefore, the product of \\(x\\) and \\(y\\) is \\(\\boxed{34}\\)."""]

    label = ['34']

    print('reward_verifier=', preprocess_box_response_for_qwen_prompt(text, label))
    print('format_verifier=', format_reward(text))
    print('step_verifier=', reasoning_steps_reward(text))


if __name__ == "__main__":
    _test_rule_verifier()
