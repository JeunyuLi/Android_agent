import re
from abc import abstractmethod
from typing import List
from http import HTTPStatus
import requests
from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import HumanMessage
from pydantic import BaseModel, Field

from agents import prompts
from utils import print_with_color, encode_image

class LLMBaseModel:
    def __init__(self):
        pass

    @abstractmethod
    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        pass

class Explore_rsp(BaseModel):
    Observation: str = Field(..., description="The result of the observation")
    Thought: str = Field(..., description="The thought process that led to the observation")
    Action: str = Field(..., description="The action to take")
    Summary: str = Field(..., description="The summary of the thought process")

class Reflect_rsp(BaseModel):
    Decision: str = Field(..., description="The decision to take")
    Thought: str = Field(..., description="The thought process that led to the decision")
    Documentation: str = Field(..., description="The documentation of the decision")

class AppLaunch_rsp(BaseModel):
    app_name: str = Field(description="Name of the application to launch")
    action: str = Field(description="Explanation of the action taken")


class Lang_Azure(LLMBaseModel):
    def __init__(self, base_url: str, api_key: str, api_version: str, model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 实例化模型
        self.mllm = AzureChatOpenAI(
            azure_endpoint=self.base_url,
            api_key=self.api_key,
            api_version=api_version,  # 假设使用固定版本，可根据需要修改
            model_name=self.model_name,
            deployment_name=self.model_name,
            request_timeout=500,
            max_retries=3,
            max_tokens=self.max_tokens,
        )

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        try:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                base64_img = encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                })
            message = HumanMessage(
                content=content
            )
            res = self.mllm.invoke([message])
            return True, res.content
        except Exception as e:
            return False, str(e)

    def get_app_launch_rsp(self, task_desc, app_list) -> AppLaunch_rsp:
        try:
            prompt_temp = prompts.launch_app_template
            app_launch_model = self.mllm.with_structured_output(AppLaunch_rsp)
            chain = prompt_temp | app_launch_model
            res: AppLaunch_rsp = chain.invoke({"task_description": task_desc,
                                               "app_list": app_list})
            return res
        except Exception as e:
            print_with_color(f"ERROR: {e}", "red")
            return AppLaunch_rsp(app_name="No application opened", action="ERROR")

    def get_explor_rsp(self, task_desc, last_act, images: List[str]) -> (list):
        try:
            prompt = prompts.self_explore_task_template_str.format(task_description=task_desc,
                                                                   last_act=last_act)
            content = [{"type": "text", "text": prompt}]
            for img in images:
                base64_img = encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                })
            message = HumanMessage(
                content=content
            )
            # 基于langchain的结构化输出
            exp_model = self.mllm.with_structured_output(Explore_rsp)
            # parser = JsonOutputParser(pydantic_object=Explore_rsp)
            res: Explore_rsp = exp_model.invoke([message])
            # res_dict = parser.parse_result(res)

            observation = res.Observation
            think = res.Thought
            act = res.Action
            last_act = res.Summary

            print_with_color("Action:", "yellow")
            print_with_color(act, "magenta")

            # 准备该函数的输出
            if "FINISH" in act:
                return ["FINISH"]
            act_name = act.split("(")[0]
            if act_name == "tap":
                area = int(re.findall(r"tap\((.*?)\)", act)[0])
                return [act_name, area, last_act]
            elif act_name == "text":
                input_str = re.findall(r"text\((.*?)\)", act)[0][1:-1]
                return [act_name, input_str, last_act]
            elif act_name == "long_press":
                area = int(re.findall(r"long_press\((.*?)\)", act)[0])
                return [act_name, area, last_act]
            elif act_name == "swipe":
                params = re.findall(r"swipe\((.*?)\)", act)[0]
                area, swipe_dir, dist = params.split(",")
                area = int(area)
                swipe_dir = swipe_dir.strip()[1:-1]
                dist = dist.strip()[1:-1]
                return [act_name, area, swipe_dir, dist, last_act]
            elif act_name == "grid":
                return [act_name]
            else:
                print_with_color(f"ERROR: Undefined act {act_name}!", "red")
                return ["ERROR"]
        except  Exception as e:

            return ["ERROR"]

    def get_reflect_rsp(self, last_res, task_desc, last_act, images: List[str]):
        act_name = last_res[0]
        area = last_res[1]

        if act_name == "tap":
            # prompt = re.sub(r"<action>", "tapping", prompts.self_explore_reflect_template)
            prompt = prompts.self_explore_reflect_template_str.format(action="tapping",
                                                                      task_desc=task_desc,
                                                                      last_act=last_act,
                                                                      ui_element=str(area))
        elif act_name == "text":
            prompt = prompts.self_explore_reflect_template_str.format(action="typing",
                                                                      task_desc=task_desc,
                                                                      last_act=last_act,
                                                                      ui_element=str(area))
        elif act_name == "long_press":
            # prompt = re.sub(r"<action>", "long pressing", prompts.self_explore_reflect_template)
            prompt = prompts.self_explore_reflect_template_str.format(action="long pressing",
                                                                      task_desc=task_desc,
                                                                      last_act=last_act,
                                                                      ui_element=str(area))
        elif act_name == "swipe":
            swipe_dir = last_res[2]
            if swipe_dir == "up" or swipe_dir == "down":
                act_name = "v_swipe"
            elif swipe_dir == "left" or swipe_dir == "right":
                act_name = "h_swipe"
            # prompt = re.sub(r"<action>", "swiping", prompts.self_explore_reflect_template)
            prompt = prompts.self_explore_reflect_template_str.format(action=act_name,
                                                                      task_desc=task_desc,
                                                                      last_act=last_act,
                                                                      ui_element=str(area))
        else:
            print_with_color("ERROR: Undefined act!", "red")
            raise ValueError("Undefined action encountered during fallback processing.")

        try:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                base64_img = encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                })
            message = HumanMessage(
                content=content
            )

            print_with_color("Reflecting on my previous action...", "yellow")
            ref_model = self.mllm.with_structured_output(Reflect_rsp)
            res: Reflect_rsp = ref_model.invoke([message])

            decision = res.Decision
            think = res.Thought
            doc = res.Documentation
        except Exception as e:
            print_with_color(f"ERROR: {e}", "red")
            raise e

        if decision == "INEFFECTIVE":
            return [decision, think]
        elif decision == "BACK" or decision == "CONTINUE" or decision == "SUCCESS":
            print_with_color("Documentation:", "yellow")
            print_with_color(doc, "magenta")
            return [decision, think, doc]
        else:
            print(f"decision = {decision}")
            print_with_color(f"ERROR: Undefined decision {decision}!", "red")
            return ["ERROR"]
