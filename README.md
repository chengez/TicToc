# Temporal Misalignment of LLM Tool Calls

This is a preliminary work under development, and is the code repo for the preprint [Temporal Blindness in Multi-Turn LLM Agents:
Misaligned Tool Use vs. Human Time Perception]().

We are working on expanding the dataset with more scenarios and trajectories, collecting human preferences from a larger population, and balancing the number of samples for ***preferTool*** and ***preferNoTool*** cases. Stay tuned for our updates!

- First version of the TicToc dataset is stored under `data`. It covers 725 human inspected trajectories. Human preferences were collected from 6 graduate student volunteers.
- Run the following to obtain the inference results:
  ```bash
  python $SCRIPT --model "$MODEL" --data "$DATA" --use_time_stamp --time_elapsed_level $ELAPSED
  ```
  - `$SCRIPT` is `eval_from_api.py` for api hosted models, e.g. OpenAI models; and is `eval_from_local.py` for local hosted models, e.g. Llama models
  - `$DATA` is one of the files under `data`, e.g. `data/preferTool_elapse_2.json`.
  - `$ELAPSED` is either '0', '1', or '2', where they refer to "small", "medium", and "large" elapse respectively. It should match the time elapsed level in `$DATA`. So if you chose `data/preferTool_elapse_2.json` for data, `$ELAPSED` should be '2'.

- Run `get_metric.py` to obtain the the number of tool call attempts. 