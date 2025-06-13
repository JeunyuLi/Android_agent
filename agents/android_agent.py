import sys
import os
import yaml
import json
import time
import ast
from langchain_core.tools import tool
from langchain.schema.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory

from agents.and_controller import AndroidController, execute_adb, traverse_tree
from utils import print_with_color, draw_bbox_multi, encode_image
from agents.state import ControlState
from utils import parse_explore_rsp, parse_reflect_rsp, AppLaunchOutputParser
from agents import prompts
from configs import load_config
configs = load_config()
mllm = AzureChatOpenAI(
    azure_endpoint=configs["OPENAI_API_BASE"],
    api_key=configs["OPENAI_API_KEY"],
    api_version=configs["OPENAI_API_VERSION"],
    model_name=configs["MODEL"],
    deployment_name=configs["MODEL"],
    request_timeout=500,
    max_tokens=configs["MAX_TOKENS"],
)
operation_history = ChatMessageHistory()

controller = AndroidController(configs["DEVICE_IP"])

def init_node(state: ControlState):
    # 获取wh
    width, height = controller.get_device_size()
    if not width and not height:
        print_with_color("ERROR: Invalid device size!", "red")
        sys.exit()
    print_with_color(f"Screen resolution of {state['device_ip']}: {width}x{height}", "yellow")
    state["screen_width"] = width
    state["screen_height"] = height

    # 将用户的操作需求添加进历史记录
    operation_history.add_user_message(state["task_desc"])

    return state

def launch_app_node(state: ControlState):
    print("🚀 Launching application...")

    # controller.home() # 回桌面
    # 获取所有已经安装的应用
    adb_command = f"adb -s {state['device_ip']} shell pm list packages"
    ret = execute_adb(adb_command)
    installed_app = ret.split('\n')
    # 应用名与启动包对应的列表
    app2package = {p.split(":")[-1].replace("com.", ""): p.split(":")[-1] for p in installed_app}
    # 读取映射表
    with open(configs["APP_MAPPING_FILE"], "r", encoding="utf-8") as f:
        app_mapping = yaml.safe_load(f)
    # 删掉当前没有安装的app
    for app, activity in app_mapping.items():
        if activity in app2package.values():
            app2package[app] = activity

    prompt = prompts.launch_app_template
    chain = prompt | mllm | AppLaunchOutputParser()
    response = chain.invoke({"task_description": state["task_desc"],
                             "app_list": str(app2package.keys())})

    if response.app_name in app2package:
        print_with_color(f"Launching {response.app_name}...", "yellow")
        controller.launch_app(app2package[response.app_name])
        operation_history.add_ai_message(response.action)
        state["app_launched"] = True
        state["last_act"] = response.action
    elif "No application opened" in response.app_name:
        print_with_color(f"ERROR: {response.app_name} is not installed!", "red")
        operation_history.add_ai_message(response.action)
        state["app_launched"] = False
        state["last_act"] = response.action
    else:
        print_with_color(f"ERROR: {response.app_name} is not installed!", "red")
        operation_history.add_ai_message(response.action)
        state["app_launched"] = False
        state["last_act"] = response.action

    return state

def capture_screen_node(state: ControlState):
    output_state = dict()
    if state["current_page_screenshot"]:
        output_state["last_page_screenshot"] = state["current_page_screenshot"]
    output_state["current_page_screenshot"] = controller.get_screenshot(f"{state['round_count']}_before", state["task_dir"])
    return output_state

def element_extract_node(state: ControlState):

    state["xml_path"] = controller.get_xml(f"{state['round_count']}", state["task_dir"])
    if state["current_page_screenshot"] == "ERROR" or state["xml_path"] == "ERROR":
        raise Exception("截图或XML获取失败")
    clickable_list = []
    focusable_list = []
    traverse_tree(state["xml_path"], clickable_list, "clickable", True)
    traverse_tree(state["xml_path"], focusable_list, "focusable", True)
    elem_list = []
    for elem in clickable_list:
        if elem.uid in state["useless_list"]:
            continue
        elem_list.append(elem)
    for elem in focusable_list:
        if elem.uid in state["useless_list"]:
            continue
        bbox = elem.bbox
        center = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
        close = False
        for e in clickable_list:
            bbox = e.bbox
            center_ = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
            dist = (abs(center[0] - center_[0]) ** 2 + abs(center[1] - center_[1]) ** 2) ** 0.5
            if dist <= configs["MIN_DIST"]:
                close = True
                break
        if not close:
            elem_list.append(elem)

    # 历史处理
    # 上一次的元素图放到
    # TODO: 优化保存文件的文件名，什么是before，什么是after？
    state["last_elem_list"] = state["current_elem_list"]
    state["current_elem_list"] = elem_list
    state["last_page_screenshot_before_draw"] = state["current_page_screenshot_draw"]
    state["current_page_screenshot_draw"] = os.path.join(state["task_dir"],
                                                         f"{state['round_count']}_before_labeled.png")
    # 在当前截图上绘制当前element标注
    draw_bbox_multi(state["current_page_screenshot"], state["current_page_screenshot_draw"], state["current_elem_list"],
                    dark_mode=configs["DARK_MODE"])
    # 在当前截图上绘制过往element标注
    if state["round_count"] != 1:
        state["last_page_screenshot_after_draw"] = os.path.join(state["task_dir"],
                                                            f"{state['round_count'] - 1}_after_labeled.png")
        draw_bbox_multi(state["current_page_screenshot"], state["last_page_screenshot_after_draw"],
                        state["last_elem_list"],
                        dark_mode=configs["DARK_MODE"])

    return state

def back_to_human_node(state: ControlState):
    output_state = dict()
    prompt = prompts.human_in_the_loop_str
    base64_img_before = state["current_page_screenshot_draw"]
    print_with_color("Thinking about what to do in the next step...", "yellow")
    start_time = time.time()
    # TODO: 待优化成langchain结构化输出
    # res = mllm.get_explor_rsp(task_desc=state["task_desc"], last_act=state["last_act"], images=[base64_img_before])
    base64_img = encode_image(base64_img_before)

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
         }
    ]
    message = HumanMessage(
        content=content
    )

    try:
        rsp = mllm.invoke([message]).content
        # exp_mllm = mllm.with_structured_output(Explore_rsp)
        # a = exp_mllm.invoke([message])
        if "YES" in rsp:
            output_state["human_in_the_loop_action"] = True
        elif "NO" in rsp:
            output_state["human_in_the_loop_action"] = False
        print_with_color(f"大模型调用成功: {rsp}", "green")

    except Exception as e:
        print_with_color(f"大模型调用错误: {e}", "red")
        output_state["next_action"] = ["ERROR", "", f"ERROR: {e}"]
        return output_state

    end_time = time.time()
    print_with_color(f"判断是否需要回传人类时的模型推理耗时: {end_time - start_time:.2f}秒", "yellow")

    return output_state

def think_next_step_node(state: ControlState):
    output_state = dict()
    prompt = prompts.self_explore_task_template.format(task_description=state["task_desc"],
                                                           last_act=state["last_act"])
    base64_img_before = state["current_page_screenshot_draw"]
    print_with_color("Thinking about what to do in the next step...", "yellow")
    start_time = time.time()
    # TODO: 待优化成langchain结构化输出
    # res = mllm.get_explor_rsp(task_desc=state["task_desc"], last_act=state["last_act"], images=[base64_img_before])
    content = [{"type": "text", "text": prompt}]
    base64_img = encode_image(base64_img_before)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_img}"
        }
    })
    message = HumanMessage(
        content=content
    )

    try:
        rsp = mllm.invoke([message]).content
        # exp_mllm = mllm.with_structured_output(Explore_rsp)
        # a = exp_mllm.invoke([message])
    except Exception as e:
        print_with_color(f"大模型调用错误: {e}", "red")
        output_state["next_action"] = ["ERROR", "", f"ERROR: {e}"]
        return output_state

    end_time = time.time()
    print_with_color(f"模型第一次推理耗时: {end_time - start_time:.2f}秒", "yellow")


    # status, rsp = mllm.get_model_response(prompt, [base64_img_before], )
    # if not status:
    #     raise Exception(f"大模型调用错误: {rsp}")
    with open(state["explore_log_path"], "a") as logfile:
        log_item = {"step": state["round_count"], "prompt": prompt,
                    "image": f"{state['round_count']}_before_labeled.png",
                    "response": rsp}
        logfile.write(json.dumps(log_item) + "\n")

    res = parse_explore_rsp(rsp)  # res的结构为[act_name, *act_params, last_act]
    # 加入操作历史中
    operation_history.add_ai_message(res[-1])
    # 如果大模型的输出超出当前屏幕上的UI元素的范围，则返回ERROR
    if res[0] != "text" and res[0] != "FINISH" and res[1] > len(state["current_elem_list"]):
        output_state["next_action"] = ["ERROR", "", f"ERROR: {res[0]} {res[1]} is out of the range of the current UI elements!"]
        print_with_color("ERROR: The model output is out of the range of the current UI elements!", "red")
    else:
        output_state["next_action"] = res
    # output_state["action_history"].append(res)

    return output_state

def fallback_node(state: ControlState):
    """
    Used to check whether the last app operation was successful.
    :param state:
    :return:
    """
    output_state = dict()
    output_state["fallback_decision"] = state["fallback_decision"]
    output_state["useless_list"] = state["useless_list"]
    output_state["doc_count"] = state["doc_count"]

    # 如果上一次没有操作或者，则跳过
    if state["step_acted"] is False:
        output_state["fallback_decision"] = "PASS"
        return output_state

    # 获取上一次的action
    last_res = state["action_history"][-1]
    act_name = last_res[0]
    area = last_res[1]

    # 如果上一次操作为输入文字，则跳过
    if act_name == "text":
        output_state["fallback_decision"] = "PASS"
        return output_state

    # 获取上一次的元素图
    img_before_path = state["last_page_screenshot_before_draw"]
    img_after_path = state["last_page_screenshot_after_draw"]

    if act_name == "tap":
        prompt = prompts.self_explore_reflect_template.format(action="tapping",
                                                                  task_desc=state["task_desc"],
                                                                  last_act=state["last_act"],
                                                                  ui_element=str(area))
    elif act_name == "text":
        prompt = prompts.self_explore_reflect_template.format(action="typing",
                                                                  task_desc=state["task_desc"],
                                                                  last_act=state["last_act"],
                                                                  ui_element=str(area))
    elif act_name == "long_press":
        prompt = prompts.self_explore_reflect_template.format(action="long pressing",
                                                                  task_desc=state["task_desc"],
                                                                  last_act=state["last_act"],
                                                                  ui_element=str(area))
    elif act_name == "swipe":
        swipe_dir = last_res[2]
        if swipe_dir == "up" or swipe_dir == "down":
            act_name = "v_swipe"
        elif swipe_dir == "left" or swipe_dir == "right":
            act_name = "h_swipe"
        prompt = prompts.self_explore_reflect_template.format(action=act_name,
                                                                  task_desc=state["task_desc"],
                                                                  last_act=state["last_act"],
                                                                  ui_element=str(area))
    else:
        print_with_color("ERROR: Undefined act!", "red")
        raise ValueError("Undefined action encountered during fallback processing.")

    print_with_color("Reflecting on my previous action...", "yellow")
    start_time = time.time()
    content = [{"type": "text", "text": prompt}]
    base64_img_before = encode_image(img_before_path)
    base64_img_after = encode_image(img_after_path)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_img_before}"
        }
    })
    content = [{"type": "text", "text": prompt},
               {
                   "type": "image_url",
                   "image_url": {
                       "url": f"data:image/jpeg;base64,{base64_img_before}"
                   }
               },
               {
                   "type": "image_url",
                   "image_url": {
                       "url": f"data:image/jpeg;base64,{base64_img_after}"
                   }
               }
               ]
    message = HumanMessage(
        content=content
    )
    try:
        rsp = mllm.invoke([message]).content
        # ref_mllm = mllm.with_structured_output(Reflect_rsp)
        # a = ref_mllm.invoke([message])
        status = True
    except Exception as e:
        print_with_color(f"大模型调用错误: {e}", "red")
        status = False
    # status, rsp = mllm.get_model_response(prompt, [img_before_path, img_after_path])
    end_time = time.time()
    print_with_color(f"模型第二次推理耗时: {end_time - start_time:.2f}秒", "yellow")
    if status:
        resource_id = state["last_elem_list"][int(area) - 1].uid
        with open(state["reflect_log_path"], "a") as logfile:
            log_item = {"step": state["round_count"] - 1, "prompt": prompt,
                        "image_before": f"{state['round_count'] - 1}_before_labeled.png",
                        "image_after": f"{state['round_count'] - 1}_after.png", "response": rsp}
            logfile.write(json.dumps(log_item) + "\n")
        res = parse_reflect_rsp(rsp)
        decision = res[0]
        output_state["fallback_decision"] = decision
        if decision == "ERROR":
            return output_state
        if decision == "INEFFECTIVE":
            output_state["useless_list"].add(resource_id)
            return output_state
        elif decision == "BACK" or decision == "CONTINUE" or decision == "SUCCESS":
            if decision == "BACK" or decision == "CONTINUE":
                output_state["useless_list"].add(resource_id)
            doc = res[-1]
            doc_name = resource_id + ".txt"
            doc_path = os.path.join(state["docs_dir"], doc_name)
            if os.path.exists(doc_path):
                doc_content = ast.literal_eval(open(doc_path).read())
                if doc_content[act_name]:
                    print_with_color(f"Documentation for the element {resource_id} already exists.", "yellow")
            else:
                doc_content = {
                    "tap": "",
                    "text": "",
                    "v_swipe": "",
                    "h_swipe": "",
                    "long_press": ""
                }
            doc_content[act_name] = doc
            with open(doc_path, "w") as outfile:
                outfile.write(str(doc_content))
            output_state["doc_count"] += 1
            print_with_color(f"Documentation generated and saved to {doc_path}", "yellow")
            return output_state
        else:
            print_with_color(f"ERROR: Undefined decision! {decision}", "red")
            return output_state

    else:
        print_with_color(rsp["error"]["message"], "red")
        output_state["fallback_decision"] = "ERROR"
        return output_state

def check_task_completion_node(state: ControlState):
    # build message
    messages = []
    messages.extend(operation_history.messages)
    messages.append(SystemMessage(
            content=prompts.check_task_finished_template_str
        ),)
    res = mllm.invoke(messages).content
    if "FINISHED" in res:
        state["completed"] = True
    elif "CONTINUE" in res:
        state["completed"] = False
        time.sleep(configs["REQUEST_INTERVAL"])
    else:
        print_with_color(f"ERROR: Undefined task completion status! {res}", "red")
    return state

def action_next_step_node(state: ControlState):
    state["round_count"] += 1

    # 执行下一步操作
    res = state["next_action"]
    act_name = res[0]
    state["last_act"] = res[-1]
    res = res[:-1]
    state["action_history"].append(res)

    if state["fallback_decision"] == "BACK":
        state["step_acted"] = False
        ret = controller.back()
        state["last_act"] = "None"
        if ret == "ERROR":
            print_with_color("ERROR: back execution failed", "red")
        return state

    # TODO: 增加对于INEFFECTIVE状态的处理，比对要按的按钮的id
    if state["fallback_decision"] == "INEFFECTIVE" and act_name != "text":
        act_uid = state["current_elem_list"][res[1]-1].uid
        # state["step_acted"] = False
        # return state
        if act_uid in state["useless_list"]:
            print_with_color("INFO: Skipping the current element.", "yellow")
            state["step_acted"] = False
            return state

    if act_name == "FINISH":
        state["completed"] = True
        return state
    if act_name == "ERROR":
        state["step_acted"] = False
        return state

    if act_name == "tap":
        _, area = res
        tl, br = state["current_elem_list"][area - 1].bbox
        x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        ret = controller.tap(x, y)
        state["step_acted"] = True
        if ret == "ERROR":
            print_with_color("ERROR: tap execution failed", "red")
    elif act_name == "text":
        _, input_str = res
        ret = controller.text(input_str)
        state["step_acted"] = True
        if ret == "ERROR":
            print_with_color("ERROR: text execution failed", "red")
    elif act_name == "long_press":
        _, area = res
        tl, br = state["current_elem_list"][area - 1].bbox
        x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        ret = controller.long_press(x, y)
        state["step_acted"] = True
        if ret == "ERROR":
            print_with_color("ERROR: long press execution failed", "red")
    elif act_name == "swipe":
        _, area, swipe_dir, dist = res
        tl, br = state["current_elem_list"][area - 1].bbox
        x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        ret = controller.swipe(x, y, swipe_dir, dist)
        state["step_acted"] = True
        if ret == "ERROR":
            print_with_color("ERROR: swipe execution failed", "red")
    else:
        print_with_color("ERROR: Cann't run this action", "read")

    return state



def is_task_completed(state: ControlState) -> str:
    """
    Check if task is completed
    """
    if state["completed"]:
        return "end"
    return "continue"

def should_fallback(state: ControlState):
    if state["fallback_decision"] in ["INEFFECTIVE", "BACK"] and state["step_acted"] is False:
        return "redone"
    elif state["fallback_decision"] == "ERROR":
        return "end"
    elif state["completed"]:
        return "end"
    else:
        return "continue"

def build_workflow() -> StateGraph:
    """
    builds the workflow
    :return:
    """
    workflow = StateGraph(ControlState)

    # add node
    workflow.add_node("init", init_node)
    workflow.add_node("launch_app", launch_app_node)
    workflow.add_node("capture_screen", capture_screen_node)
    workflow.add_node("element_extract", element_extract_node)
    workflow.add_node("think_next_step", think_next_step_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("action", action_next_step_node)
    workflow.add_node("complete", check_task_completion_node)

    # add edge
    workflow.set_entry_point("init")
    workflow.add_edge("init", "launch_app")
    workflow.add_edge("launch_app", "capture_screen")
    workflow.add_edge("capture_screen", "element_extract")
    workflow.add_edge("element_extract", "think_next_step")
    workflow.add_edge("element_extract", "fallback")
    workflow.add_edge("think_next_step", "action")
    workflow.add_edge("fallback", "action")

    # routing
    workflow.add_conditional_edges("action", should_fallback,
                                   {"redone": "capture_screen", "continue": "complete", "end": END})
    workflow.add_conditional_edges("complete", is_task_completed,
                                   {"continue": "capture_screen", "end": END})



    return workflow