import json


# Set parameters
data = {"scratch": "/datain/",
        "syspath": "/workspace/acoustic_private/",
        "path_to_echograms": "/datain/"}

# Write parameters
data_set = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
# Write to file
with open('setpyenv.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


# Run the predictions
