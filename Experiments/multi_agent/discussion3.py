import json
import os
import logging
import subprocess
import time
import pickle
import re

from typing import Dict, Any, List, Union

from openai import OpenAI
from role_init import run_pipeline
from prompt import prompts

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
        try:
            # Attempt to parse the string as JSON
            str = str.replace("'", "\"")
            pattern = re.compile(r'(\{.*\})', re.DOTALL)
            match = pattern.search(str)
            if match:
                filtered = match.group(1)
                json_data = json.loads(filtered)
            return json_data
        except json.JSONDecodeError:
            # If parsing fails, return None or handle the error as needed
            logging.error(f"Failed to parse JSON: {str}")
            return None

class SubTask():
    def __init__(self, task_name, core_goal):
        self.task_name = task_name
        self.core_goal = core_goal
        self.prompt = prompts['subtask_prompt'] \
            .replace("{task_name}", self.task_name) \
            .replace("{core_goal}", self.core_goal)
        self.agent_roles: List[OpenAIAgent] = []

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
            # subtask=agent_subtask,
            agent_role_prompt=agent_role_prompt,
            speaking_rate=speaking_rate
        ))
    
    def construct_response(self, is_last_round, is_first_round, current_agent: OpenAIAgent, most_recent_responses, optimized_design: None):
        prefix_string = "你所在的子任务小组目前正在讨论任务的实施方案，你现在在和其他团队成员积极讨论。请尽己所能给出有价值的观点和创新性的想法。\n"
        if is_first_round:
            if optimized_design is not None:
                prefix_string += f"\n以下是一个较优的方案，请你在此方案基础上，展开你的优化。\n较优方案{optimized_design}"
            return prefix_string
        prefix_string += "\n以下是来自其他团队成员的设计:\n"
        recommend_list = []
        recommend_prompt = ""
        query = ""
        query_retrieved = ""
        query_prompt = ""
        current_design = {}
        for agent_role, responses in most_recent_responses.items():
            content = OpenAIAgent.extract_json(responses[-1]["content"])
            if agent_role != current_agent.agent_role:
                design = content.get("设计", {})
                recommend = content.get("建议", {})
                if len(recommend) > 0:
                    for key, value in recommend.items():
                        if key.split("-")[0].strip() == current_agent.agent_role:
                            recommend_list.append(f"来自 {agent_role} 对你在 {key.split("-")[1].strip()} 设计上的建议：\n 建议: {value["意见"]}\n 原因: {value["原因"]}\n")
                prefix_string += f"团队成员 {agent_role} 的设计: {design}\n"
            else:
                query = content.get("检索内容", self.task_name)
                current_design = content.get("设计", {})
        if is_last_round:
            prefix_string += f"团队成员(你) {current_agent.agent_role} 的设计: {current_design}\n"
        if len(query) > 0:
            context = current_agent.construct_user_message(f"请根据以下内容进行检索: {query}")
            query_retrieved = current_agent.generate_answer(context)
        if len(recommend_list) > 0:
            recommend_prompt = "以及他人对你的建议"
            prefix_string += "\n以下是来自其他团队成员对你的建议:\n"
            prefix_string += "\n".join(recommend_list) + "\n"
        if len(query_retrieved) > 0:
            prefix_string += f"\n同时对于你刚才的提问 {query}, 以下是来自知识库的检索结果:\n{query_retrieved}\n 希望对你充分理解任务需求并设计出最优的方案有所帮助。\n"
            query_prompt = "以及你从知识库检索获得的相关内容"
        if not is_last_round:
            prefix_string += f"""
根据以上内容，请你结合他人设计中的亮点{recommend_prompt}{query_prompt}，优化你的观点。
同时请从你的专业角度对其他人的观点选择支持、反对或提出不同的看法，并用简短明确的语言指出其他团队成员观点中的不足之处或可改进的地方。

请注意保持讨论过程的友好和建设性，同时在针对其他人的观点时请在开头部分明确指出提出这个观点的对象，以提升讨论的针对性。
同时请记住在讨论中向其他人声明你的角色。请在讨论中保持礼貌。
"""
        if is_last_round:
            prefix_string += """
你们的讨论时间即将结束，请根据以上内容，从包括你的设计在内的所有人的设计中，请组合选择出你认为最优的一套方案，并给出你的简要选择理由。

你的输出格式应如下所示。
{
    "演员元素": {
        "设计选择": 你认为最优的团队成员名称(如舞蹈专家,当你选择自身设计时,给出你的角色名称),
        "选择理由": 你的简短选择理由
    },
    "舞台元素": {
        "设计选择": ...,
        "选择理由": ...
    },
    "灯光元素": {
        "设计选择": ...,
        "选择理由": ...
    },
    "道具元素": {
        "设计选择": ...,
        "选择理由": ...
    }
}
"""
        return prefix_string

    def run(self, rounds: int, higher_prompt: str, response_from_other_subtasks = None):
        """
        rounds: number of debate rounds
        prompt_template supports placeholders:
          {agent_role_prompt}, {subtask_name}, {subtask_core_goal}, {round}, {is_last_round}
        """
        tool_call_prompt = """
        """
        chat_history = {agent.agent_role: [] for agent in self.agent_roles}
        most_recent_responses = {}
        for r in range(rounds + 1):
            is_last = (r == rounds)
            is_first = r == 0
            round_responses = {agent.agent_role: [] for agent in self.agent_roles}
            for agent in self.agent_roles:
                response_from_others = self.construct_response(
                    is_last_round=is_last,
                    is_first=is_first,
                    current_agent=agent,
                    most_recent_responses=most_recent_responses,
                    optimized_design=response_from_other_subtasks
                )
                # build per-agent prompt
                prompt = agent.self_prompt + higher_prompt + self.prompt + agent.agent_role_prompt + response_from_others
                user_msg = agent.construct_user_message(prompt)
                chat_history[agent.agent_role].append(user_msg)
                response = agent.generate_answer(chat_history[agent.agent_role])
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
                        final_content = OpenAIAgent.extract_json(most_recent_responses[agent.agent_role][-3]["content"].replace("'", "\""))
                        try:
                            final_json = json.loads(final_content)
                            # Try to fetch a detailed design field; fallback to the entire component design if not specified.
                            design_detail = final_json.get("设计").get(component)
                            # If design_detail is a dict, further extract component details if available.
                            if isinstance(design_detail, dict): # and component in design_detail:
                                design_details.append(design_detail)
                            # Fetch the design detail succesfully
                        except Exception as e:
                            logging.error(f"Error extracting detailed design from agent {agent.agent_role}: {e}")
                        break
            if len(design_details) == 1:
                design_details = design_details[0]
            best_designs[component] = design_details
        return chat_history, best_designs

class TotalTask:
    def __init__(self, task_text: str, model_name: str):
        # 1) run the pipeline to get structured task, subtasks and roles
        task_json, subtasks, roles_by_subtask = run_pipeline(task_text)
        self.agent = OpenAIAgent(
            model_name="qwen-plus-2025-04-28",
            agent_role=None,
            agent_speciality=None,
            agent_subtask=None,
            agent_role_prompt=None,
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
        self.subtasks: List[SubTask] = []
        # 2) for each subtask, init SubTask and its agents
        self.sub_task_prompt = ""
        for sub in subtasks:
            name = sub.get("任务名称", "")
            core_goal = sub.get("核心目标", "")
            st = SubTask(name, core_goal)
            for role in roles_by_subtask.get(name, []):
                st.init_agent(
                    model_name=model_name,
                    agent_role=role.get("角色名称"),
                    agent_speciality=role.get("专业特点"),
                    agent_subtask=role.get("工作职责"),
                    speaking_rate=role.get("speaking_rate", 1.0),
                    agent_role_prompt=prompts["agent_task"]
                )
            self.subtasks.append(st)
        self.sub_task_prompt = ", ".join([_.task_name for _ in subtasks])

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
        best_design_ch = generation["最佳设计组合"]
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
        for i in range(iters):
            response_from_other_subtasks = ""
            results = {}
            designs = {}
            for st in self.subtasks:
                history, best_design = st.run(rounds, self.prompt, None if i == 0 else response_from_other_subtasks)
                
                results[st.task_name] = history
                designs[st.task_name] = best_design
                response_from_other_subtasks += f"\n子任务小组 {st.task_name} : {best_design}\n"
            head_agent_subtask_prompt = prompts["head_agent_subtask"].replace("{subtasks}", self.sub_task_prompt)
            head_agent_task = prompts["head_agent_task"].replace("{discussion_results}", response_from_other_subtasks)
            prompt = prompts["head_agent_role"] + self.prompt + head_agent_subtask_prompt + head_agent_task
            context = self.agent.construct_user_message(prompt)
            generation = OpenAIAgent.extract_json(self.agent.generate_answer(context))
            # get the best answer from all subtasks returns
            total_best_answer = self.fetch_best_answer(generation=generation, subtask_designs=designs)

            response_from_other_subtasks = str(total_best_answer)

        return results

if __name__ == "__main__":
    total = TotalTask(
        task_text="以大熊猫为主题为成都2025世界运动会设计一开场节目",
        model_name="qwen-plus-2025-04-28"#"qwen3-30b-a3b"#"qwen2.5-14b-instruct-1m"#"qwen-turbo-2025-04-28",#"qwen2.5-14b-instruct-1m",#"",
    )
    total.save_structured(
        "total_task.json",
        fmt="json"
    )
    total.save_structured(
        "total_task.pkl",
        fmt="pickle"
    )
    # 运行辩论
    debate_results = total.run_all(iters=1, rounds=2)
    print(debate_results)
    # 保存结构化初始化结果