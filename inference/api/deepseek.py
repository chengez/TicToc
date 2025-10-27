import os
from inference.model_handler import Base_Handler
import openai
from inference.sys_pmts import *

class DeepSeek_Handler(Base_Handler):
    def __init__(self, model_name="deepseek-chat"):
        super().__init__(model_name)
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.model = model_name

    def format_input(self, history, tools=None, tools_in_user_message=True, date_string=None, add_generation_prompt=False, time_elapsed_level=0, use_time_stamp=False, use_special_sys_prompt_naive=False, use_special_sys_prompt_rule=False):
        """
        Format the input for OpenAI chat models. Converts history to OpenAI's message format and attaches tools if provided.
        Args:
            history: List of message dicts (role/content/tool_calls/etc)
            tools: List of tool/function definitions (from the 'function' field in data)
        Returns:
            Dict with 'messages' and optionally 'tools' for OpenAI API
        """
        messages = []
        for msg in history:
            m = {"role": msg["role"]}
            time_string = msg['time'] if type(msg['time']) is str else msg['time'][time_elapsed_level]
            if msg.get("content") is not None and "tool_calls" not in msg:
                m["content"] = f"[{time_string}] " + msg["content"] if use_time_stamp else msg["content"]
                if m["role"] == "system" and use_special_sys_prompt_naive:
                    m["content"] = m["content"] + NAIVE
                elif m["role"] == "system" and use_special_sys_prompt_rule:
                    m["content"] = m["content"] + RULE
            if "tool_calls" in msg:
                m["tool_calls"] = msg["tool_calls"]
            # Ensure tool messages have tool_call_id (required by OpenAI API)
            if msg["role"] == "tool" and "tool_call_id" in msg:
                m["tool_call_id"] = msg["tool_call_id"]
            messages.append(m)
        # OpenAI expects tools as a list of function dicts
        openai_tools = None
        if tools is not None:
            # OpenAI expects a list of dicts with 'type' and 'function' keys
            openai_tools = tools
        return {"messages": messages, "tools": openai_tools} if openai_tools else {"messages": messages}

    def run_inference(self, formatted_inputs):
        """
        Run batch inference using OpenAI chat completions API in parallel.
        Args:
            formatted_inputs: List of dicts as returned by format_input
        Returns:
            List of output strings (assistant responses)
        """
        import concurrent.futures
        print(f"Running with model: {self.model}")
        def infer_one(formatted):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted["messages"],
                    tools=formatted.get("tools", None),
                    tool_choice="auto"
                )
                choice = response.choices[0].message
                # breakpoint()
                # if "Please reserve a Luxury Suite for Sherlock Holmes there." in formatted["messages"][-1]["content"]:
                #     breakpoint()
                if hasattr(choice, "content") and choice.content is not None and choice.content != "":
                    return choice.content
                elif hasattr(choice, "tool_calls") and choice.tool_calls is not None:
                    return str(choice.tool_calls)
                else:
                    return "[ERROR]"
            except Exception as e:
                print(f"Error during inference for sample: {formatted}\nException: {e}")
                return f"[ERROR]: {e}"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(infer_one, formatted_inputs))
        return results
