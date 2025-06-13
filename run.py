
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
    # task = "帮我浏览抖音短视频，不断向上滑动以切换到下一个视频"
    task = "帮我拍两张照片"
    # device = "10.39.52.148:5555"
    result = run_task(task, configs["DEVICE_IP"])
    print(f"Task execution result: {result}")