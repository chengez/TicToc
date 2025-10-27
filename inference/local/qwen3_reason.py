from inference.model_handler import Base_Handler
from jinja2 import Environment, FileSystemLoader
from vllm import LLM, SamplingParams
import os
import torch
from inference.sys_pmts import *

class Qwen3_Handler(Base_Handler):
    def __init__(self, model_path="qwen/Qwen3-14B"):
        super().__init__("qwen3")
        self.model_path = model_path
        self.llm = LLM(model=self.model_path, tensor_parallel_size=torch.cuda.device_count())
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)

    def format_input(self, history, tools=None, add_generation_prompt=True, time_elapsed_level=0, use_time_stamp=False, use_special_sys_prompt_naive=False, use_special_sys_prompt_rule=False):
        """
        Format the input history for Qwen3 using the Jinja template.
        Args:
            history: List of message dicts (role/content/tool_calls/etc)
            tools: List of tool/function definitions (from the 'function' field in data)
            add_generation_prompt: Whether to add assistant generation prompt (default False)
        Returns:
            Rendered prompt string
        """
        env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), '../../templates')),
                          trim_blocks=True, lstrip_blocks=True)
        template = env.get_template('qwen3.jinja')
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
                m["content"] = None
            if msg["role"] == "tool" and "tool_call_id" in msg:
                m["tool_call_id"] = msg["tool_call_id"]
            messages.append(m)
        rendered = template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=True
        )
        return rendered

    def run_inference(self, formatted_inputs):
        """
        Run batch inference using vllm.
        Args:
            formatted_inputs: List of formatted prompt strings
        Returns:
            List of output strings
        """
        # breakpoint()
        outputs = self.llm.generate(formatted_inputs, self.sampling_params)
        return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]
