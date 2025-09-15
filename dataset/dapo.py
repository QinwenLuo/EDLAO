from datasets import concatenate_datasets, load_dataset


# ======================
# 全局 index 生成器
# ======================
class GlobalIndex:
    def __init__(self, start=0):
        self.counter = start

    def next(self):
        idx = self.counter
        self.counter += 1
        return idx


global_index = GlobalIndex(start=0)


# 加载数据集
ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "all", trust_remote_code=True)
df_math = load_dataset("DigitalLearningGmbH/MATH-lighteval", trust_remote_code=True)

train_dataset = ds["train"]
train_dataset_math = df_math["train"]

local_dir = "./data/dapo"


# 工具函数
def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]
    return retval


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


instruction_following = (
    "Let's think step by step and output the final answer within \\boxed{}."
)
data_source_math = "DigitalLearningGmbH/MATH-lighteval"


# map 函数
def make_map_fn(split):
    def process_fn(example, idx):
        orig_extra_info = example.pop("extra_info")
        extra_info = orig_extra_info.copy()
        extra_info["index"] = global_index.next()
        extra_info["split"] = split
        example["prompt"] = example["source_prompt"]
        example["extra_info"] = extra_info
        return example

    return process_fn


def make_map_fn_math(split):
    def process_fn_math(example, idx):
        question = example.pop("problem")
        question = question + " " + instruction_following

        answer = example.pop("solution")
        solution = extract_solution(answer)

        data = {
            "data_source": data_source_math,
            "prompt": [{"role": "user", "content": question}],
            "solution": solution,
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": global_index.next()},
        }
        return data

    return process_fn_math


def prompt_str_to_list(example):
    if isinstance(example["prompt"], str):
        return {**example, "prompt": [{"role": "user", "content": example["prompt"]}]}
    return example


# 处理两个数据集
train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
train_dataset_math = train_dataset_math.map(
    function=make_map_fn_math("train"), with_indices=True
)

train_dataset.to_parquet(local_dir + "train.parquet")
train_dataset_math.to_parquet(local_dir + "train_math.parquet")


combined_dataset = concatenate_datasets([train_dataset, train_dataset_math])
combined_dataset = combined_dataset.shuffle(seed=42).remove_columns(
    ["source_prompt", "ability", "type", "level"]
)

combined_dataset.to_parquet(local_dir + "train_combined.parquet")
