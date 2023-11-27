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
        **experiment_params
    )
    
    # Run the MadnessPipeline
    result = madness_pipeline.run()

    print(f'Result: {result}')
    evaluator = BracketEvaluator(year=23)
    score = evaluator.evaluate(result)
    print(f'Score: {score}')
    
    print(evaluator.confidence_interval(confidence=0.9))
    print(evaluator.confidence_interval(confidence=0.95))
    print(evaluator.confidence_interval(confidence=0.99))