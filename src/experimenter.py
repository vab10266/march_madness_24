from time import localtime, strftime
import os
import json

def dict2dicts(param_ranges):
    param_options = {}
    keys = []
    num_choices = []
    out = []
    for key in param_ranges.keys():
        if type(param_ranges[key]) == list:
            param_options[key] = param_ranges[key]
            keys.append(key)
            num_choices.append(len(param_ranges[key]))
    print(param_options)
    print(num_choices)
    tmp = param_ranges.copy()
    for i in range(len(keys)):
        for j in range(num_choices[i]):
            key = keys[i]
            tmp[key] = param_options[key][j]
            out.append(tmp)

    for d in out:
        print(d)
    # print(out)


class Experimenter:
    def __init__(self, path):
        self.path = path
        self.id = f"{path.split('/')[-1].split('.')[0]}_{strftime('%Y_%m_%d_%H:%M:%S', localtime())}"
        print(self.id)

        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), path))) as json_file:
            experiment_params = json.load(json_file)
            print(experiment_params)
        dict2dicts(experiment_params)

if __name__ == "__main__":
    e = Experimenter("../experiments/test.json")
