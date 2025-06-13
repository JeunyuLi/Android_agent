from langchain_core.prompts import PromptTemplate


self_explore_task_template_str = """You are an agent that is trained to complete certain tasks on a smartphone. You will be 
given a screenshot of a smartphone app. The interactive UI elements on the screenshot are labeled with numeric tags 
starting from 1. 

You can call the following functions to interact with those labeled elements to control the smartphone:

1. tap(element: int)
This function is used to tap an UI element shown on the smartphone screen.
"element" is a numeric tag assigned to an UI element shown on the smartphone screen.
A simple use case can be tap(5), which taps the UI element labeled with the number 5.

2. text(text_input: str)
This function is used to insert text input in an input field/box. text_input is the string you want to insert and must 
be wrapped with double quotation marks. A simple use case can be text("Hello, world!"), which inserts the string 
"Hello, world!" into the input area on the smartphone screen. This function is only callable when you see a keyboard 
showing in the lower half of the screen.

3. long_press(element: int)
This function is used to long press an UI element shown on the smartphone screen.
"element" is a numeric tag assigned to an UI element shown on the smartphone screen.
A simple use case can be long_press(5), which long presses the UI element labeled with the number 5.

4. swipe(element: int, direction: str, dist: str)
This function is used to swipe an UI element shown on the smartphone screen, usually a scroll view or a slide bar.
"element" is a numeric tag assigned to an UI element shown on the smartphone screen. "direction" is a string that 
represents one of the four directions: up, down, left, right. "direction" must be wrapped with double quotation 
marks. "dist" determines the distance of the swipe and can be one of the three options: short, medium, long. You should 
choose the appropriate distance option according to your need.
A simple use case can be swipe(21, "up", "medium"), which swipes up the UI element labeled with the number 21 for a 
medium distance.

The task you need to complete is to {task_description}. Your past actions to proceed with this task are summarized as 
follows: {last_act}
Now, given the following labeled screenshot, you need to think and call the function needed to proceed with the task. 
Your output should include three parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the next step I should do>
Action: <The function call with the correct parameters to proceed with the task. If you believe the task is completed or 
there is nothing to be done, you should output FINISH. You cannot output anything else except a function call or FINISH 
in this field.>
Summary: <Summarize your past actions along with your latest action in one or two sentences. Do not include the numeric 
tag in your summary>
You can only take one action at a time, so please directly call the function."""

self_explore_task_template = PromptTemplate(input_variables=["task_description", "last_act"],
                                            template=self_explore_task_template_str)

self_explore_reflect_template_str = """I will give you screenshots of a mobile app before and after {action} the UI 
element labeled with the number 'ui_element' on the first screenshot. The numeric tag of each element is located at 
the center of the element. The action of {action} this UI element was described as follows:
{last_act}
The action was also an attempt to proceed with a larger task, which is to {task_desc}. Your job is to carefully analyze 
the difference between the two screenshots to determine if the action is in accord with the description above and at 
the same time effectively moved the task forward. Your output should be determined based on the following situations:
1. BACK
If you think the action navigated you to a page where you cannot proceed with the given task, you should go back to the 
previous interface. At the same time, describe the functionality of the UI element concisely in one or two sentences by 
observing the difference between the two screenshots. Notice that your description of the UI element should focus on 
the general function. Never include the numeric tag of the UI element in your description. You can use pronouns such as 
"the UI element" to refer to the element. Your output should be in the following format:
Decision: BACK
Thought: <explain why you think the last action is wrong and you should go back to the previous interface>
Documentation: <describe the function of the UI element>
2. INEFFECTIVE
If you find the action changed nothing on the screen (screenshots before and after the action are identical), you 
should continue to interact with other elements on the screen. Notice that if you find the location of the cursor 
changed between the two screenshots, then they are not identical. Your output should be in the following format:
Decision: INEFFECTIVE
Thought: <explain why you made this decision>
3. CONTINUE
If you find the action changed something on the screen but does not reflect the action description above and did not 
move the given task forward, you should continue to interact with other elements on the screen. At the same time, 
describe the functionality of the UI element concisely in one or two sentences by observing the difference between the 
two screenshots. Notice that your description of the UI element should focus on the general function. Never include the 
numeric tag of the UI element in your description. You can use pronouns such as "the UI element" to refer to the 
element. Your output should be in the following format:
Decision: CONTINUE
Thought: <explain why you think the action does not reflect the action description above and did not move the given 
task forward>
Documentation: <describe the function of the UI element>
4. SUCCESS
If you think the action successfully moved the task forward (even though it did not completed the task), you should 
describe the functionality of the UI element concisely in one or two sentences. Notice that your description of the UI 
element should focus on the general function. Never include the numeric tag of the UI element in your description. You 
can use pronouns such as "the UI element" to refer to the element. Your output should be in the following format:
Decision: SUCCESS
Thought: <explain why you think the action successfully moved the task forward>
Documentation: <describe the function of the UI element>
"""

self_explore_reflect_template = PromptTemplate(input_variables=["action", "task_desc", "last_act", "ui_element"],
                                               template=self_explore_reflect_template_str)

check_task_finished_template_str = """You are an agent responsible for determining whether a given task on a smartphone has been completed.

Carefully review the operation history and determine whether the task described above has been successfully completed.

Your output must be one of the following:
- FINISHED (if the task was completed)
- CONTINUE (if the task is still in progress or incomplete)

Do not output any additional explanation, reasoning, or text. Only return either "FINISHED" or "CONTINUE".
"""

launch_app_template_str = """The task you need to complete is to {task_description}. 

Below is a list of applications already installed on the device:
{app_list}

Please conduct a comprehensive analysis and recommend one application that best fulfills the user's intent. 
Your output must strictly adhere to the following rules:

1. Only select an app if it directly matches the task description.
2. If none of the apps can fulfill the task, return: `APP: No application opened`.
3. You must only launch one application at a time.
4. Your decision should be based solely on the application names and their relevance to the task.
5. Do not make up any application name or infer packages from the name.

Format your response exactly as follows:
APP: <Application package name or "No application opened">
Action: <Brief explanation of why this app was selected or why no app could be launched>
"""
launch_app_template = PromptTemplate(input_variables=["task_description", "app_list"],
                                     template=launch_app_template_str)
