import argparse

import jsonschema

from inference.model_map import MODEL_TO_TOOLCALL_SIGNATURE
import os, json
from openai import OpenAI

client = OpenAI()
def LLM_as_judge(history, name, parameters, functions):
    """
    Evaluates if the assistant's last tool call is logically correct
    based on the conversation history and available functions.

    Args:
        history (list): A list of dictionaries representing the conversation
                        history (e.g., [{'role': 'user', 'content': '...'}, ...]).
        name (str): The last assistant message tool name.
        parameters (dict): The last assistant message tool parameters.
        functions (list): A list of dictionaries describing the available tools
                          (in OpenAI's function/tool format).

    Returns:
        bool: True if the tool call is judged as correct, False otherwise.
              Returns False if an error occurs during the API call or parsing.
    """
    history_str = json.dumps(history, indent=4)
    functions_str = json.dumps(functions, indent=4)
    # 2. Define the system prompt for the LLM judge.
    # This prompt instructs the LLM on its role, the context it will receive,
    # and how it should format its response.
    prompt = f"""
        You are an expert evaluator of AI assistant responses. Your task is to determine if an AI assistant has chosen the correct tool and parameters to respond to a user's request based on the provided conversation history and the available tools.

        You must focus solely on the **logical correctness** and **appropriateness** of the tool call. You should assume that the tool call's syntax (function name, parameter types, etc.) has already been validated and is correct. Your evaluation should be based on whether the call makes sense in the context of the conversation.

        **Context:**
        ***
        **Available Tools:**
        ```json
        {functions_str}
        ```
        ***
        **Conversation History:**
        ```
        {history_str}
        ```
        ***
        **Assistant's Tool Call to Evaluate:**
        tool name: {name}
        tool parameters: {json.dumps(parameters, indent=4)}
        ***

        **Evaluation Task:**

        1.  **Analyze the Conversation History:** Understand the user's most recent request and the overall goal of the conversation.
        2.  **Examine the Assistant's Tool Call:** Look at the specific tool chosen and the values of the parameters provided.
        3.  **Assess Logical Correctness:**
            { "- Is the chosen tool the most logical one to address the user's last message?" if len(functions)>1  else ""}
            { "- Would another tool have been more appropriate?" if len(functions)>1  else ""}
            - Are the parameter values correctly extracted or inferred from the conversation?

        **Final Decision:**

        Based on your analysis, is the assistant's tool call the correct and logical choice in this situation?

        Respond with a JSON object containing two keys, first reason key then is_correct key.
        - "reason": a brief, one-sentence explanation for your decision.
        - "is_correct": a boolean value (true or false).
        
        example output:
        ```json
        {{"reason": "The assistant chose the correct tool and parameters.", "is_correct": true}}
        ```
        
        start outputting now.
        """

    messages = [
        {"role": "system",
         "content": "You are an expert evaluator of AI assistant tool calls. Respond only in JSON format."},
        {"role": "user", "content": prompt}
    ],
    # 3. Call the LLM with the constructed prompt.
    # We use JSON mode to ensure the output is a valid JSON object.
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.0, # Use a low temperature for objective evaluation
    )

    # 4. Parse the response and return the decision.
    judge_response_str = response.output_text.strip().strip("`")
    if judge_response_str.startswith("json"):
        judge_response_str = judge_response_str[4:].strip().strip("`").strip()
    judge_response_json = json.loads(judge_response_str)
    if args.print_logs:
        print(f"Judge's Reason: {judge_response_json.get('reason', 'No reason provided.')}")
        print(f"Judge's Decision: {judge_response_json.get('is_correct', 'No decision provided.')}")
    return  judge_response_json.get("is_correct", True)



def check_tool_call_structer(out, name_to_param, args):
    correct_params = None
    name = None
    parameters = None
    try:
        if "meta-llam" in args.model:
            function_call = json.loads(out)
            name = function_call['name']
            parameters = json.loads(function_call['parameters'])

        if "Qwen" in args.model:
            out = out.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            function_call = json.loads(out)
            name = function_call['name']
            parameters = function_call['arguments']

        if "Ministral" in args.model:
            function_call = json.loads(out)[0]
            name = function_call['name']
            parameters = json.loads(function_call['arguments'])

        if "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" in args.model:
            out = out.split("<｜tool▁call▁begin｜>function<｜tool▁sep｜>")[1].split("<｜tool▁call▁end｜>")[0]
            name = out.split("```json")[0].strip()
            parameters = json.loads(out.split("```json")[1].strip().rstrip("`"))

        if "o4" in args.model or "o3" in args.model or "gpt" in args.model or "deepseek-chat" in args.model:
            class ChatCompletionMessageToolCall:
                def __init__(self, **kwargs):
                    self.id = kwargs.get('id')
                    self.function = kwargs.get('function')
                    self.type = kwargs.get('type')

            class Function:
                def __init__(self, **kwargs):
                    self.arguments = kwargs.get('arguments')
                    self.name = kwargs.get('name')

            safe_scope = {
                'ChatCompletionMessageToolCall': ChatCompletionMessageToolCall,
                'Function': Function
            }

            # Evaluate the string within the safe scope
            parsed_data = eval(out, {"__builtins__": None}, safe_scope)
            tool_call = parsed_data[0]

            function_obj = tool_call.function
            name = function_obj.name
            parameters = json.loads(function_obj.arguments)

        if name not in name_to_param:
            raise Exception(f"Unexpected tool name {name} found in output.")

        correct_params = name_to_param[name]
        jsonschema.validate(instance=parameters, schema=correct_params)
        for parameter in parameters:
            if parameter not in correct_params['properties']:
                raise Exception(f"Unexpected parameter {parameter} found in output.")
        for parameter in correct_params['required']:
            if parameter not in parameters:
                raise Exception(f"Required parameter {parameter} not found in output.")

        correct_used = True

    except Exception as e:
        if args.print_logs:
            print(f"Error processing output: {e}")
            try:
                print("error:", e)
                print(f"Output: {out}")
                print(f"name: {name}")
                print(f"parameters: {parameters}")
                print(f"Correct correct_params: {json.dumps(correct_params, indent=4)}")
                print("-" * 40)
            except:
                print("-" * 40)

        correct_used = False
    return correct_used, name, parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get tool call rate.")
    parser.add_argument("--data", type=str, default="final_2_elapse_2", help="data name")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model code (will be mapped to handler).")
    parser.add_argument("--time_elapsed_level", type=int, default=0, choices=[0, 1, 2], help="Time elapsed level for adding time stamp to each message. 0: small change, 1: medium change, 2: large change.")
    parser.add_argument("--use_time_stamp", action="store_true", help="Whether to use time stamp in the prompt.")
    parser.add_argument("--print_logs", action="store_true", help="Print logs for debugging/seeing the problem with the model.")
    parser.add_argument("--output_dir", type=str, default="outputs3", help="Directory containing the output files.")
    args = parser.parse_args()
    tool_call_signatures = MODEL_TO_TOOLCALL_SIGNATURE.get(args.model)
    if tool_call_signatures is None:
        raise ValueError(f"Unknown model code: {args.model}")
    print()
    with open(f"{args.data}.json", "r") as f:
        input_data = json.load(f)
    id_to_location = {input_d["id"]:i for i, input_d in enumerate(input_data)}
    if args.use_time_stamp:
        output_file = f"{args.model.split('/')[-1]}-{args.data.split('/')[-1]}-{args.time_elapsed_level}.json"
    else:
        output_file = f"{args.model.split('/')[-1]}-{args.data.split('/')[-1]}-notime.json"
    
    # load the json file
    with open(os.path.join(args.output_dir, args.model.split('/')[-1], output_file), "r") as f:
        output_data = json.load(f)
    if args.use_time_stamp:
        print(f"Processing {args.model} on {args.data} with time elapsed level {args.time_elapsed_level}")
    else:
        print(f"Processing {args.model} on {args.data} without time elapsed")
    attempted = 0
    correct_attempted = 0
    llm_as_judge_corrects = 0
    failed = 0
    for output in output_data:
        out = output['output']
        inp = input_data[id_to_location[output['id']]]
        functions_sig = inp['function']
        name_to_param = {function["function"]['name']: function["function"]["parameters"] for function in functions_sig }

        if out == "" or "[ERROR]" in out:
            failed += 1
            continue
        used = all(sig in out for sig in tool_call_signatures.split('<AND>'))

        if used:
            correct_used, name, parameters = check_tool_call_structer(out, name_to_param, args)
            # if correct_used:
            #     correct_params = LLM_as_judge(inp['history'], name, parameters, inp['function'])
            # else:
            #     correct_params = False
        else:
            correct_used = False
            correct_params = False



        if used:
            attempted += 1
        if correct_used:
            correct_attempted += 1
        # if correct_params:
        #     llm_as_judge_corrects +=1
        # if correct_used and correct_params:
        #     correct_attempted += 1
    
    total = len(output_data) - failed
    attempt_rate = attempted / total if total > 0 else 0
    correct_attempt_rate = correct_attempted / total if total > 0 else 0
    llm_as_judge_corrects_rate = llm_as_judge_corrects / total if total > 0 else 0
    print(f"Total samples (excluding failed): {total}")
    print(f"Attempted tool calls: {attempted} out of {total}")
    print(f"Attempt rate: {attempt_rate:.4%}")
    print("------------")
    print(f"Correct Attempted tool calls: {correct_attempted} out of {total}")
    print(f"Correct Attempt rate: {correct_attempt_rate:.4%}")
    # print("------------")
    # print(f"Correct LLM as Judge tool calls: {llm_as_judge_corrects} out of {total}")
    # print(f"Correct LLM as Judge rate: {llm_as_judge_corrects_rate:.4%}")