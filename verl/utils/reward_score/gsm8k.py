# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    final_answer = None

    # Step 1: Try to extract \boxed{} content first
    boxed_answers = re.findall(r"\\boxed\{([^\}]*)\}", solution_str)
    if boxed_answers:
        # Get last boxed content
        boxed_content = boxed_answers[-1]
        # Try to find number inside (e.g., remove units like 'hours')
        numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", boxed_content)
        if numbers:
            final_answer = numbers[0].replace(",", "")
            return final_answer

    # Step 2: Fallback to legacy methods
    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if len(solutions) != 0:
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        for final_answer in reversed(answer):
            if final_answer not in ["", "."]:
                final_answer = final_answer.replace(",", "")
                break

    return final_answer
    

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
