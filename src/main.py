import os
import json

from pipeline_store import *
from evaluation import *

if __name__ == "__main__":
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiment_params.json'))) as json_file:
        experiment_params = json.load(json_file)
        print(experiment_params)

    train_years = experiment_params["train_years"]
    test_years = experiment_params["test_year"]
    del(experiment_params["train_years"])
    del(experiment_params["test_year"])
    # Create an instance of MadnessPipeline
    madness_pipeline = MadnessPipeline(
        **experiment_params
    )
    
    # Run the MadnessPipeline
    result = madness_pipeline.fit(data_dict={'years': train_years})
    result = madness_pipeline.predict(data_dict={'years': test_years})

    print(f'Result: {result}')
    # evaluator = BracketEvaluator(year=23)
    # score = evaluator.evaluate(result['brackets'])
    # print(f'Score: {score}')
    
    # print(evaluator.confidence_interval(confidence=0.9))
    # print(evaluator.confidence_interval(confidence=0.95))
    # print(evaluator.confidence_interval(confidence=0.99))