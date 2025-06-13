import pytest
from agents.android_agent import launch_app_node, back_to_human_node

@pytest.mark.parametrize("task, expected", [("帮我打开12306买张高铁票", True),
                                            ("打开小红书", True),
                                            ("打开百度贴吧", False),
                                            ("帮我打开百度地图", True),
                                            ("帮我打开唯品会APP", False)])
@pytest.mark.repeat(10)
def test_launch_app_node(task, expected):
    # assert add(task, b) == expected
    state = {
        "device_ip": "10.39.52.43:5555",
        "task_desc": task,
        "work_dir": "./",
        "app_name": "robot",
        "app_launched": False,
        "step": 0,
        "step_acted": False,
        "fallback_decision": "CONTINUE",
        "action_history": [],
        "reflect_history": [],
        "reflect_action": "",
    }
    result_state = launch_app_node(state)
    assert result_state["app_launched"] == expected

# @pytest.mark.parametrize("screenshot_draw_path, expected", [("tests/screenshot/1.png", True),
#                                                             ("tests/screenshot/2.png", True),
#                                                             ("tests/screenshot/3.png", True),
#                                                             ("tests/screenshot/4.png", False),
#                                                             ("tests/screenshot/5.png", False),
#                                                             ("tests/screenshot/6.png", False),])
# # @pytest.mark.parametrize("screenshot_draw_path, expected", [("tests/screenshot/4.png", False),])
# @pytest.mark.repeat(2)
# def test_human_in_the_loop(screenshot_draw_path, expected):
#     state = {
#         "current_page_screenshot_draw": screenshot_draw_path,
#     }
#     result_state = back_to_human_node(state)
#     assert result_state["human_in_the_loop_action"] == expected