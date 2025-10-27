import os
from inference.model_handler import Base_Handler
import cohere
from inference.sys_pmts import *

class Cohere_Handler(Base_Handler):
    def __init__(self, model_name="command-r"):  # Default to Cohere's Command R model
        super().__init__(model_name)
        self.api_key = os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY environment variable not set.")
        self.client = cohere.ClientV2(self.api_key)
        self.model = model_name

    def format_input(self, history, tools=None, tools_in_user_message=True, date_string=None, add_generation_prompt=False, time_elapsed_level=0, use_time_stamp=False, use_special_sys_prompt_naive=False, use_special_sys_prompt_rule=False):
        """
        Format the input for Cohere chat models. Converts history to Cohere's message format and attaches tools if provided.
        Args:
            history: List of message dicts (role/content/tool_calls/etc)
            tools: List of tool/function definitions (if supported)
        Returns:
            Dict with 'chat_history' and optionally 'tools' for Cohere API
        """
        chat_history = []
        for msg in history:
            entry = {"role": msg["role"]}
            time_string = msg['time'] if type(msg['time']) is str else msg['time'][time_elapsed_level]
            if msg.get("content") is not None and "tool_calls" not in msg:
                entry["content"] = f"[{time_string}] " + msg["content"] if use_time_stamp else msg["content"]
                if entry["role"] == "system" and use_special_sys_prompt_naive:
                    entry["content"] = entry["content"] + NAIVE
                elif entry["role"] == "system" and use_special_sys_prompt_rule:
                    entry["content"] = entry["content"] + RULE
            if "tool_calls" in msg:
                entry["tool_calls"] = msg["tool_calls"]
            if msg["role"] == "tool" and "tool_call_id" in msg:
                entry["tool_call_id"] = msg["tool_call_id"]
            chat_history.append(entry)
        cohere_tools = tools if tools is not None else None
        return {"chat_history": chat_history, "tools": cohere_tools} if cohere_tools else {"chat_history": chat_history}

    def run_inference(self, formatted_inputs):
        """
        Run batch inference using Cohere chat completions API in parallel (multithreaded).
        Args:
            formatted_inputs: List of dicts as returned by format_input
        Returns:
            List of output strings (assistant responses)
        """
        import concurrent.futures

        def infer_one(formatted):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=formatted["chat_history"],
                    tools=formatted.get("tools", None),
                    tool_choice="auto"
                )
                response_msg = response.message if hasattr(response, "message") else None
                if hasattr(response_msg, "content") and response_msg.content is not None:
                    return response_msg.content[0].text
                elif hasattr(response_msg, "tool_calls") and response_msg.tool_calls is not None:
                    return str(response_msg.tool_calls)
                else:
                    return "[ERROR]"
            except Exception as e:
                print(f"Error during inference for sample: {formatted}\nException: {e}")
                return f"[ERROR]: {e}"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(infer_one, formatted_inputs))
        return results
