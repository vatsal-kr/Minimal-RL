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
    return 1.0 if answer == ground_truth else 0.0