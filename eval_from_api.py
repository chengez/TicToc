import os
import json
from inference.model_handler import Base_Handler
import importlib
from utils import load_data
import argparse
from inference.model_map import MODEL_TO_HANDLER

def get_handler(handler_name: str, **kwargs) -> Base_Handler:
    module = importlib.import_module(f"inference.api.{handler_name}")
    handler_class = None
    from inference.model_handler import Base_Handler
    for attr in dir(module):
        obj = getattr(module, attr)
        if (
            isinstance(obj, type)
            and issubclass(obj, Base_Handler)
            and obj is not Base_Handler
            and attr.lower().endswith("_handler")
        ):
            handler_class = obj
            break
    if handler_class is None:
        raise ValueError(f"No handler class found in {handler_name}")
    return handler_class(**kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate API-based LLM function calling.")
    parser.add_argument("--data", type=str, default="data/example.json", help="Path to data JSON file.")
    parser.add_argument("--model", type=str, default="o3", help="Model code (will be mapped to handler).")
    parser.add_argument("--time_elapsed_level", type=int, default=0, choices=[0, 1, 2], help="Time elapsed level for adding time stamp to each message. 0: small change, 1: medium change, 2: large change.")
    parser.add_argument("--use_time_stamp", action="store_true", help="Whether to use time stamp in the prompt.")
    parser.add_argument("--use_special_sys_prompt_naive", action="store_true", help="Whether to use special system prompt (naive).")
    parser.add_argument("--use_special_sys_prompt_rule", action="store_true", help="Whether to use special system prompt (emperical rule).")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output JSON files.")
    args = parser.parse_args()
    assert not (args.use_special_sys_prompt_naive and args.use_special_sys_prompt_rule), "Cannot use both special sys prompts."
    use_time_stamp = True if args.use_time_stamp else False
    handler_name = MODEL_TO_HANDLER.get(args.model)
    if handler_name is None:
        raise ValueError(f"Unknown model code: {args.model}")
    data = load_data(args.data)
    model_name = args.model
    if "-FC" in model_name:
        model_name = model_name.replace("-FC", "")
    handler = get_handler(handler_name, model_name=model_name)
    formatted_prompts = []
    sample_ids = []
    for sample in data:
        history = sample["history"]
        tools = sample.get("function", None)
        formatted = handler.format_input(history, tools=tools, time_elapsed_level=args.time_elapsed_level, use_time_stamp=use_time_stamp, use_special_sys_prompt_naive=args.use_special_sys_prompt_naive, use_special_sys_prompt_rule=args.use_special_sys_prompt_rule)
        formatted_prompts.append(formatted)
        sample_ids.append(sample.get('id', 'N/A'))
    outputs = handler.run_inference(formatted_prompts)
    # for idx, text in enumerate(outputs):
    #     print(f"Sample ID: {sample_ids[idx]}")
    #     print(f"Output: {text}\n")

    # save outputs in json format
    output_data = [{"id": sample_ids[i], "output": outputs[i]} for i in range(len(outputs))]
    if use_time_stamp:
        output_file = f"{args.model.replace('/', '_')}-{args.data.split('/')[-1][:-5]}-{args.time_elapsed_level}.json"
    else:
        output_file = f"{args.model.replace('/', '_')}-{args.data.split('/')[-1][:-5]}-notime.json"

    if not os.path.exists(os.path.join(args.output_dir, args.model.replace('/', '_'))):
        os.makedirs(os.path.join(args.output_dir, args.model.replace('/', '_')))
    with open(os.path.join(args.output_dir, args.model.replace('/', '_'), output_file), 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Outputs saved to {output_file}")
