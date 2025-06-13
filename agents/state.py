from typing_extensions import TypedDict
from typing import Set, List, Dict, Optional, Annotated, Callable, Literal, Any
from langgraph.graph import add_messages
import os
import datetime
import time

class ControlState(TypedDict):

    task_desc: str
    step: int
    history_steps: List[Dict]

    # device related
    device_ip: str
    device_resolution: str
    screen_width: int
    screen_height: int

    # page related
    current_page_screenshot: str
    last_page_screenshot: str
    current_page_screenshot_draw: str
    last_page_screenshot_before_draw: str
    last_page_screenshot_after_draw: str


    # element related
    xml_path: str
    current_elem_list: List
    last_elem_list: List
    useless_list: Set

    # action related
    next_action: List
    reflect_action: str
    human_in_the_loop_action: bool
    action_history: List[str]
    reflect_history: List[str]
    last_act: str
    step_acted: bool

    # decision_related
    fallback_decision: Literal["ERROR", "INEFFECTIVE", "BACK", "CONTINUE", "SUCCESS", "PASS"]

    # dir
    work_dir: str
    demo_dir: str
    task_dir: str
    docs_dir: str

    # log
    explore_log_path: str
    reflect_log_path: str

    # 状态标志位
    app_launched: bool
    completed: bool
    round_count: int
    doc_count: int


def create_controlstate(device_ip:str, task:str):

    # 先初始化状态字典
    state: Dict[str, Any] = {"task_desc": task, "step": 0, "history_steps": [], "device_ip": device_ip,
                             "device_resolution": "", "screen_width": 0, "screen_height": 0,
                             "current_page_screenshot": "", "last_page_screenshot": "",
                             "current_page_screenshot_draw": "", "last_page_screenshot_before_draw": "",
                             "last_page_screenshot_after_draw": "", "xml_path": "", "current_elem_list": [],
                             "last_elem_list": [], "useless_list": set(),
                             "next_action": [], "reflect_action": "", "human_in_the_loop_action": False,
                             "action_history": [], "reflect_history": [], "last_act": "",
                             "step_acted": False, "fallback_decision": "PASS", "work_dir": "", "demo_dir": "",
                             "task_dir": "", "docs_dir": "", "explore_log_path": "", "reflect_log_path": "",
                             "app_launched": False, "completed": False, "round_count": 1, "doc_count": 0}

    # 初始化一些目录
    work_dir = "./apps/robot"
    os.makedirs(work_dir, exist_ok=True)
    demo_dir = os.path.join(work_dir, "demos")
    os.makedirs(demo_dir, exist_ok=True)
    demo_timestamp = int(time.time())
    task_name = datetime.datetime.fromtimestamp(demo_timestamp).strftime("self_explore_%Y-%m-%d_%H-%M-%S")
    task_dir = os.path.join(demo_dir, task_name)
    os.mkdir(task_dir)
    docs_dir = os.path.join(work_dir, "auto_docs")
    os.makedirs(docs_dir, exist_ok=True)
    explore_log_path = os.path.join(task_dir, f"log_explore_{task_name}.txt")
    reflect_log_path = os.path.join(task_dir, f"log_reflect_{task_name}.txt")

    # 初始化一些值
    state["work_dir"] = work_dir
    state["demo_dir"] = demo_dir
    state["task_dir"] = task_dir
    state["docs_dir"] = docs_dir
    state["explore_log_path"] = explore_log_path
    state["reflect_log_path"] = reflect_log_path

    # Check if all fields are initialized
    for key in ControlState.__annotations__:
        if key not in state:
            raise ValueError(f"State initialization failed: missing field '{key}'")

    return state


