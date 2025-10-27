# Maps model code to handler module name (without .py)
MODEL_TO_HANDLER = {
    # API models
    "gpt-4.1-mini-2025-04-14-FC": "openai",
    "gpt-4.1-nano-2025-04-14-FC": "openai",
    "gpt-4.1-2025-04-14-FC": "openai",
    "gpt-4o-mini-2024-07-18-FC": "openai",
    "gpt-4o-2024-11-20-FC": "openai",
    "o3-2025-04-16-FC": "openai",
    "o4-mini-2025-04-16-FC": "openai",
    "command-r": "cohere",
    "command-r-plus": "cohere",
    "command-a": "cohere",
    "deepseek-chat": "deepseek",

    # Local models
    "meta-llama/Llama-3.1-8B-Instruct": "llama3_1",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3_2",
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen3-8B-reason": "qwen3_reason",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2_5",
    "mistralai/Ministral-8B-Instruct-2410": "ministral",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek_distill_qwen",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek_distill_llama",

}

MODEL_TO_TOOLCALL_SIGNATURE = {
    "gpt-4.1-mini-2025-04-14-FC": "ChatCompletionMessageToolCall",
    "gpt-4.1-nano-2025-04-14-FC": "ChatCompletionMessageToolCall",
    "gpt-4.1-2025-04-14-FC": "ChatCompletionMessageToolCall",
    "gpt-4o-mini-2024-07-18-FC": "ChatCompletionMessageToolCall",
    "gpt-4o-2024-11-20-FC": "ChatCompletionMessageToolCall",
    "o3-2025-04-16-FC": "ChatCompletionMessageToolCall",
    "o4-mini-2025-04-16-FC": "ChatCompletionMessageToolCall",
    "command-r": "ToolCallV2",
    "command-r-plus": "ToolCallV2",
    "command-a": "ToolCallV2",
    "deepseek-chat": "ChatCompletionMessageToolCall",


    "meta-llama/Llama-3.1-8B-Instruct": "\"name\"<AND>\"parameters\"",
    "meta-llama/Llama-3.2-3B-Instruct": "\"name\"<AND>\"parameters\"",
    "Qwen/Qwen3-8B": "<tool_call><AND></tool_call>",
    "Qwen/Qwen2.5-7B-Instruct": "<tool_call><AND></tool_call>",
    "mistralai/Ministral-8B-Instruct-2410": "\"arguments\"<AND>\"name\"",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "<｜tool▁call▁begin｜><AND><｜tool▁call▁end｜>",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "<｜tool▁call▁begin｜><AND><｜tool▁call▁end｜>",
}


MODEL_TO_SHORT_NAME = {
    # API models
    "gpt-4.1-mini-2025-04-14-FC": "gpt-4.1-mini",
    "gpt-4.1-nano-2025-04-14-FC": "gpt-4.1-nano",
    "gpt-4.1-2025-04-14-FC": "gpt-4.1",
    "gpt-4o-mini-2024-07-18-FC": "gpt-4o-mini",
    "gpt-4o-2024-11-20-FC": "gpt-4o",
    "o3-2025-04-16-FC": "o3",
    "o4-mini-2025-04-16-FC": "o4-mini",
    "command-r": "command-r",
    "command-r-plus": "command-r-plus",
    "command-a": "command-a",
    "deepseek-chat": "deepseek-chat",

    # Local models
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Qwen/Qwen3-8B": "Qwen-3-8B\n(no reasoning)",
    "Qwen/Qwen3-8B-reason": "Qwen-3-8B\n(with reasoning)",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen-2.5-7B",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",

}
