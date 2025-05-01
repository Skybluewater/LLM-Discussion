import json
import os
import logging
import argparse
import copy
import time
import pickle
import re
import concurrent.futures

from typing import Dict, Any, List, Union

from openai import OpenAI
from role_init import run_pipeline
from prompt import prompts
from prompt_baseline import prompts as prompts_baseline

argparser = argparse.ArgumentParser(description="Run multi-agent discussion")
argparser.add_argument("--is_baseline", type=bool, help="Use baseline methods")
args = argparser.parse_args()

is_baseline = args.is_baseline
if is_baseline:
    prompts = prompts_baseline
    
org_task_text = "以大熊猫为主题为成都2025世界运动会设计一开场节目"
sub_task_to_dps = ""

class Agent:
    def generate_answer(self, answer_context):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_assistant_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_user_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")

class OpenAIAgent(Agent):
    def __init__(self, model_name, agent_role, agent_speciality, agent_subtask, agent_role_prompt, speaking_rate, missing_history = []):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.agent_role = agent_role
        self.agent_subtask = agent_subtask
        self.agent_speciality = agent_speciality
        self.self_prompt = prompts["agent_prompt"] \
            .replace("{agent_role}", self.agent_role) \
            .replace("{agent_speciality}", self.agent_speciality) \
            .replace("{agent_subtask}", self.agent_subtask)
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history

    def generate_answer(self, answer_context, temperature=1):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=answer_context,
                n=1)
            result = completion.choices[0].message.content
            # for pure text -> return completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(10)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}

    @staticmethod
    def extract_json(str):
        # Attempt to parse the string as JSON
        try:
            str = str.replace("'", "").replace("json", "").replace("：", ":").replace("，", ",").replace("`", "")
            pattern = re.compile(r'(\{.*\})', re.DOTALL)
            match = pattern.search(str)
            if match:
                filtered = match.group(1)
                json_data = json.loads(filtered)
            return json_data
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON: {str}")
            return None

class SubTask():
    def __init__(self, task_name, core_goal, agents, model_name="qwen-plus-2025-04-28"):
        self.task_name = task_name
        self.core_goal = core_goal
        self.subtask_prompt = "" if is_baseline else prompts['subtask_prompt'] \
            .replace("{task_name}", self.task_name) \
            .replace("{core_goal}", self.core_goal)
        self.agent_roles: List[OpenAIAgent] = []
        for agent in agents:
            self.init_agent(
                model_name=model_name,
                agent_role=agent.get("角色名称"),
                agent_speciality=agent.get("专业特点"),
                agent_subtask=agent.get("工作职责"),
                speaking_rate=agent.get("speaking_rate", 1.0),
                agent_role_prompt=prompts["agent_task"]
            )

    def __str__(self):
        return f"SubTask({self.task_name}, {self.core_goal})"

    def __repr__(self):
        return f"SubTask({self.task_name}, {self.core_goal})"
    
    def init_agent(self, model_name, agent_role, agent_speciality, agent_subtask, agent_role_prompt, speaking_rate):
        self.agent_roles.append(OpenAIAgent(
            model_name=model_name,
            agent_role=agent_role,
            agent_subtask=agent_subtask,
            agent_speciality=agent_speciality,
            agent_role_prompt=agent_role_prompt,
            speaking_rate=speaking_rate
        ))
    
    def construct_response(self, current_agent: OpenAIAgent, is_first_round, is_last_round, most_recent_responses, optimized_design: None):
        prefix_string = "你所在的子任务小组目前正在讨论任务的实施方案，你现在在和其他团队成员积极讨论。请尽己所能给出有价值的观点和创新性的想法。\n"
        if is_first_round:
            if optimized_design is not None:
                prefix_string += f"\n以下是一个较优的方案，请你在此方案基础上，展开你的优化。\n较优方案{optimized_design}"
            return prefix_string
        prefix_string += "\n以下是来自其他团队成员的设计:\n"
        recommend_list = []
        recommend_prompt = ""
        query, query_retrieved, query_prompt, current_design = "", "", "", {}
        most_recent_responses = copy.deepcopy(most_recent_responses)
        for agent_role, responses in most_recent_responses.items():
            content = OpenAIAgent.extract_json(responses[-1]["content"])
            if content is None:
                continue
            if agent_role != current_agent.agent_role:
                design = content.get("设计", {})
                recommend = content.get("建议", {})
                if len(recommend) > 0:
                    for key, value in recommend.items():
                        if key == "个人建议":
                            recommend_list.append(f"来自 {agent_role} 在设计上的建议：\n 建议: {value["意见"]}\n 原因: {value["原因"]}\n")
                        elif key.split("-")[0].strip() == current_agent.agent_role:
                            splits = key.split("-")
                            if len(splits) > 1:
                                recommend_list.append(f"来自 {agent_role} 对你在 {key.split("-")[1].strip()} 设计上的建议：\n 建议: {value["意见"]}\n 原因: {value["原因"]}\n")
                            else:
                                recommend_list.append(f"来自 {agent_role} 对你的建议：\n 建议: {value["意见"]}\n 原因: {value["原因"]}\n")
                prefix_string += f"团队成员 {agent_role} 的设计: {design}\n"
            else:
                query = content.get("检索内容", "")
                current_design = content.get("设计", {})
        if is_last_round:
            prefix_string += f"团队成员(你) {current_agent.agent_role} 的设计: {current_design}\n"
        if len(query) > 0:
            context = [current_agent.construct_user_message(f"请根据以下内容进行检索: {query}，并用简短的语言以Json格式回答。")]
            query_retrieved = current_agent.generate_answer(context)
        if len(recommend_list) > 0:
            recommend_prompt = "以及他人对你的建议"
            prefix_string += "\n以下是来自其他团队成员对你的建议，你可以选择采纳或不采纳:\n"
            prefix_string += "\n".join(recommend_list) + "\n"
        if len(query_retrieved) > 0:
            prefix_string += f"\n同时对于你刚才的提问 {query}, 以下是来自知识库的检索结果:\n{query_retrieved}\n 希望对你充分理解任务需求并设计出最优的方案有所帮助。\n"
            query_prompt = "以及你从知识库检索获得的相关内容"
        if not is_last_round:
            prefix_string += prompts["agent_debate"] \
                .replace("{recommend_prompt}", recommend_prompt) \
                .replace("{query_prompt}", query_prompt)
        if is_last_round:
            prefix_string += prompts["agent_conv"]
        return prefix_string

    def run(self, rounds: int, higher_prompt: str, response_from_other_subtasks = None):
        """
        rounds: number of debate rounds
        prompt_template supports placeholders:
          {agent_role_prompt}, {subtask_name}, {subtask_core_goal}, {round}, {is_last_round}
        """
        print(f"Running subtask: {self.task_name} with {len(self.agent_roles)} agents")
        chat_history = {agent.agent_role: [] for agent in self.agent_roles}
        most_recent_responses = {}
        for r in range(rounds + 1):
            is_last = (r == rounds)
            is_first = r == 0
            round_responses = {agent.agent_role: [] for agent in self.agent_roles}
            for agent in self.agent_roles:
                response_from_others = self.construct_response(
                    current_agent=agent,
                    is_first_round=is_first,
                    is_last_round=is_last,
                    most_recent_responses=most_recent_responses,
                    optimized_design=response_from_other_subtasks
                )
                # build per-agent prompt
                if is_baseline:
                    prompt = agent.self_prompt + higher_prompt + agent.agent_role_prompt + response_from_others
                else:
                    prompt = agent.self_prompt + higher_prompt + self.subtask_prompt + agent.agent_role_prompt + response_from_others
                user_msg = agent.construct_user_message(prompt)
                chat_history[agent.agent_role].append(user_msg)
                response = agent.generate_answer(chat_history[agent.agent_role])
                if sub_task_to_dps == self.task_name:
                    print(f"Agent {agent.agent_role} response: {response}...")
                assistant_msg = agent.construct_assistant_message(response)
                chat_history[agent.agent_role].append(assistant_msg)
                round_responses[agent.agent_role] = chat_history[agent.agent_role]
            most_recent_responses = round_responses
        # After debate rounds, aggregate votes from final responses
        vote_counts = {}
        for agent in self.agent_roles:
            final_msg = most_recent_responses[agent.agent_role][-1]["content"]
            try:
                final_vote = OpenAIAgent.extract_json(final_msg)
            except Exception as e:
                logging.error(f"Error parsing JSON from {agent.agent_role}: {e}")
                continue
            try:
                for component, result in final_vote.items():
                    try:
                        chosen = result.get("设计选择")
                    except Exception as e:
                        logging.error(f"Error parsing JSON from {agent.agent_role}: {e}")
                        continue
                    if not chosen:
                        continue
                    if isinstance(chosen, list):
                        chosen = chosen[0]
                    if not isinstance(chosen, str):
                        continue
                    vote_counts.setdefault(component, {})
                    vote_counts[component][chosen] = vote_counts[component].get(chosen, 0) + 1
            except Exception as e:
                logging.error(f"Error parsing JSON from {agent.agent_role}: {e}")
                continue

        best_answers = {}
        for component, votes in vote_counts.items():
            max_votes = max(votes.values())
            best_candidates = [candidate for candidate, count in votes.items() if count == max_votes]
            best_answers[component] = {
                "设计选择": best_candidates,
                "票数": max_votes,
            }
        best_designs: Dict[str, Union[Dict, List[Dict]]] = {}
        for component, info in best_answers.items():
            best_roles = info["设计选择"]
            design_details = []
            for best_role in best_roles:
                for agent in self.agent_roles:
                    if agent.agent_role == best_role:
                        final_content = OpenAIAgent.extract_json(most_recent_responses[agent.agent_role][-3]["content"])
                        # Try to fetch a detailed design field; fallback to the entire component design if not specified.
                        try:
                            design_detail = final_content.get("设计").get(component)
                            if isinstance(design_detail, dict):
                                design_details.append(design_detail)
                        except Exception as e:
                            logging.error(f"Error extracting detailed design from agent {agent.agent_role}: {e}")
                        break
            if len(design_details) == 1:
                design_details = design_details[0]
            elif len(design_details) > 2:
                design_details = design_details[0]
            best_designs[component] = design_details
        print(f"Best designs for {self.task_name}: {best_designs}")
        return chat_history, best_designs

class TotalTask:
    def __init__(self, task_text: str, model_name: str):
        # 1) run the pipeline to get structured task, subtasks and roles
        task_json, subtasks, roles_by_subtask = run_pipeline(task_text, is_baseline=is_baseline)
        self.agent = OpenAIAgent(
            model_name="qwen-plus-2025-04-28",
            agent_role="大统领",
            agent_speciality="狠狠压榨",
            agent_subtask="管喽喽",
            agent_role_prompt="看我干啥，瞅你咋地",
            speaking_rate=1.0,
        )
        self.task_info = task_json
        self.task_bg = task_json.get("任务背景", "")
        self.task_core_goal = task_json.get("核心需求", "")
        self.task_additional = task_json.get("附加需求", [])
        self.task_optimized = task_json.get("优化提问", "")
        self.prompt = prompts["total_task"] \
            .replace("{task_optimized}", self.task_optimized) \
            .replace("{task_core_goal}", self.task_core_goal) \
            .replace("{task_additional}", str(self.task_additional)) \
            .replace("{task_bg}", self.task_bg)
        
        # If is baseline, there are no subtasks but roles, init subtask as empty dictionary
        if is_baseline:
            subtasks = [{"--foo": "--bar"}]
        
        # 2) for each subtask, init SubTask and its agents
        self.subtasks: List[SubTask] = []
        self.sub_task_prompt = ""
        for sub in subtasks:
            name = sub.get("任务名称", "baseline")
            core_goal = sub.get("核心目标", "baseline")
            st = SubTask(name, core_goal, roles_by_subtask.get(name, []), model_name=model_name)
            self.subtasks.append(st)
        self.sub_task_prompt = ", ".join([_.get("任务名称", "") for _ in subtasks])

    def __str__(self):
        return f"TotalTask with {len(self.subtasks)} subtasks"

    def to_dict(self):
        return {
            "task_info": self.task_info,
            "subtasks": [
                {
                    "task_name": st.task_name,
                    "core_goal": st.core_goal,
                    "agents": [
                        {
                            "agent_role": ag.agent_role,
                            "agent_speciality": ag.agent_speciality,
                            "agent_subtask": ag.agent_subtask
                        }
                        for ag in st.agent_roles
                    ]
                }
                for st in self.subtasks
            ]
        }

    def save_structured(self, filepath: str, fmt: str = "json"):
        data = self.to_dict()
        if fmt == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif fmt == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
    
    def fetch_best_answer(self, generation: Dict, subtask_designs: Dict):
        best_design_ch = generation["最优设计组合"]
        combest_design: Dict[str, Dict] = {}
        for component, info in best_design_ch.items():
            subtask_choice = info["设计选择"]
            try:
                design = subtask_designs.get(subtask_choice).get(component)
                combest_design[component] = design
            except Exception as e:
                print(f"Fetch best answer error: {e}")
        return combest_design

    def run_all(self, iters: int, rounds: int):
        """
        Run debate for each subtask and return a mapping:
          { subtask_name: chat_history }
        """
        print(f"Running total task: {self.task_core_goal} with {len(self.subtasks)} subtasks")
        # 1) run the pipeline to get structured task, subtasks and roles
        final_best_design = {}
        for i in range(iters):
            response_from_other_subtasks = ""
            results, designs = {}, {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.subtasks)) as executor:
                future_to_subtask = {
                    executor.submit(
                        st.run,
                        rounds,
                        self.prompt,
                        None if i == 0 else response_from_other_subtasks
                    ): st.task_name for st in self.subtasks
                }
                for future in concurrent.futures.as_completed(future_to_subtask):
                    task_name = future_to_subtask[future]
                    history, best_design = future.result()
                    results.setdefault(task_name, []).append(history)
                    designs[task_name] = best_design
                    response_from_other_subtasks += f"\n子任务小组 {task_name} : {best_design}\n"
            
            if is_baseline:
                break
            
            # 2) Remix & Review
            head_agent_subtask_prompt = prompts["head_agent_subtask"].replace("{subtasks}", self.sub_task_prompt)
            head_agent_task = prompts["head_agent_task"].replace("{discussion_results}", response_from_other_subtasks)
            prompt = prompts["head_agent_role"] + self.prompt + head_agent_subtask_prompt + head_agent_task
            context = [self.agent.construct_user_message(prompt)]
            generation = OpenAIAgent.extract_json(self.agent.generate_answer(context))
            print(f"Iteration {i + 1} best choice reason: {generation}")
            # get the best answer from all subtasks returns
            total_best_answer = self.fetch_best_answer(generation=generation, subtask_designs=designs)
            response_from_other_subtasks = str(total_best_answer)
            final_best_design = total_best_answer
            # save the best answer to use for next turn discussion
            print(f"Best designs for {self.task_core_goal} in iter{i + 1}: {total_best_answer}")

        return results, final_best_design


if __name__ == "__main__":
    total = TotalTask(
        task_text=org_task_text,
        model_name="qwen-plus-2025-04-28"#"qwen2.5-vl-32b-instruct"#"qwen-plus-2025-04-28"#"qwen3-30b-a3b"#"qwen2.5-14b-instruct-1m"#"qwen-turbo-2025-04-28",#,#"","
    )
    total.save_structured(
        f"total_task_{org_task_text[:10]}_{is_baseline}.json",
        fmt="json"
    )
    sub_task_to_dps = total.subtasks[0].task_name
    debate_results, best_design = total.run_all(iters=3, rounds=3)
    with open(f"debate_results_{org_task_text[:10]}_{is_baseline}.pkl", "wb") as f:
        pickle.dump(debate_results, f)
    with open(f"best_design_{org_task_text[:10]}_{is_baseline}.pkl", "wb") as f:
        pickle.dump(best_design, f)