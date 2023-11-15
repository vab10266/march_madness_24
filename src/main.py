import os
import json

from pipeline_store import *
from evaluation import *

if __name__ == "__main__":
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiment_params.json'))) as json_file:
        experiment_params = json.load(json_file)
        print(experiment_params)

    # Create an instance of MadnessPipeline
    madness_pipeline = MadnessPipeline(
        train_years=experiment_params["train_years"],
        test_year=experiment_params["test_year"],
        cols=experiment_params["cols"], 
        model_key=experiment_params["model_key"], 
        param_dict=experiment_params["param_dict"], 
        rand=experiment_params["rand"], 
        num_groups=experiment_params["num_groups"], 
        brackets_per_group=experiment_params["brackets_per_group"]
    )

    # Run the MadnessPipeline
    result = madness_pipeline.run()

    print(f'Result: {result}')
    evaluator = BracketEvaluator(year=23)
    score = evaluator.evaluate(result)
    print(f'Score: {score}')