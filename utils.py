import json

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    # If the file is a dict with a single sample, wrap in a list
    if isinstance(data, dict):
        data = [data]
    return data
