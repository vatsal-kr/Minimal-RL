import re


def extract_boxed_contents_list(text: str) -> str:
    """
    Extracts all contents within \\boxed{...} from a given text string,
    after normalizing braces.
    """
    # Match \boxed{...} with non-greedy content
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.search(pattern, text)
    try:
        matches = matches.group(1)
    except Exception:
        matches = None
    return matches


def list_format_reward(completion, num_candidates, **kwargs):
    """
    Assigns a reward based on whether the model's output is correctly formatted.
    0 if correctly formatted, else -1
    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
    Returns:
        List[float]: A list of rewards for each completion.
    """
    pattern_dict = {
        2: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{[AB]}$",
        3: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{[ABC]}$",
        4: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{[ABCD]}$",
        5: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{[ABCDE]}$",
    }

    match = re.match(pattern_dict[num_candidates], completion, re.DOTALL)
    return 0.0 if match else -1.0


def compute_score(solution_str, ground_truth, extra_info=None) -> float:
    """
    Compute the reward score based on the provided solution string and ground truth.
    By default, it uses the correctness only reward. To change this behaviour, pass `custom_reward_fn` in the config
    Args:
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed reward score.
    """
    answer = solution_str.split("</think>")[-1].strip()
    answer = extract_boxed_contents_list(answer)
    correctness_reward = 1.0 if answer == ground_truth else 0.0
    format_reward = list_format_reward(f"<think>\n{solution_str}", extra_info["num_candidates"])
    return correctness_reward + format_reward
