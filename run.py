
from agents.android_agent import build_workflow
from configs.config import load_config
from utils import show_graph
configs = load_config()

def run_task(task: str, device: str) -> bool:
    print(f"🚀 Starting task execution: {task}")

    try:
        from agents.state import create_controlstate

        state = create_controlstate(device, task)
        workflow = build_workflow()
        app = workflow.compile()

        # 显示langgraph 的workflow的图
        # show_graph(app)

        result = app.invoke(state, {"recursion_limit": 1000})
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    task = "帮我给john发条手机短信，和他说hello" # 输入任务描述
    # device = "10.39.52.148:5555"
    result = run_task(task, configs["DEVICE_IP"])
    print(f"Task execution result: {result}")