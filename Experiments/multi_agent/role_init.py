from typing import Dict, Any, List
import os
import json
from prompt import prompts as prompts_meta                # your prompt definitions from prompt.py
from prompt_baseline import prompts as prompts_baseline  # your prompt definitions from prompt_baseline.py
from openai import OpenAI      # or whatever LLM client you use


task = "以大熊猫为主题为成都2025世界运动会设计一开场节目"

# 1. Low‑level call into the LLM
def call_llm(prompt: str) -> str:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
            model="qwen-turbo-2025-04-28",
            messages=[{"role": "user", "content": prompt}]
        )
    result = completion.choices[0].message.content
    return result

# 2. Pipeline orchestration
def run_pipeline(question: str, is_baseline: bool = False) -> Dict[str, Any]:
    if is_baseline:
        prompts = prompts_baseline
    else:
        prompts = prompts_meta
    # a. Task rewrite → structured
    rewrite_prompt = prompts["task_rewrite"].replace("{question}", question)
    structured = call_llm(rewrite_prompt)
    task_json = json.loads(structured)

    if is_baseline:
        role_prompt = prompts["role_generation"].replace("{question_overall}", json.dumps(task_json, ensure_ascii=False))
        ret = call_llm(role_prompt)
        roles = json.loads(ret)
        return task_json, None, {"baseline": roles}
    # b. Split into subtasks
    split_prompt = prompts["subtask_split"].replace("{question}", json.dumps(task_json, ensure_ascii=False))
    subtasks = json.loads(call_llm(split_prompt))

    # c. For each subtask generate roles
    # 保存每个子任务对应的角色列表
    roles_by_subtask: Dict[str, List[Any]] = {}
    for sub in subtasks:
        # 使用子任务名称作为键
        task_name = sub.get("任务名称", str(sub))
        role_prompt = prompts["role_generation"] \
            .replace("{question_overall}", json.dumps(task_json, ensure_ascii=False)) \
            .replace("{question_subtask}", json.dumps(sub, ensure_ascii=False))
        ret = call_llm(role_prompt)
        roles = json.loads(ret)
        roles_by_subtask[task_name] = roles
    return task_json, subtasks, roles_by_subtask

# 3. CLI entrypoint
if __name__ == "__main__":
    roles = run_pipeline(task, is_baseline=False)
    roles = run_pipeline(task)
    print(json.dumps(roles, ensure_ascii=False, indent=2))
