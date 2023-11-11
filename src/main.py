from pipeline_store import *

if __name__ == "__main__":
    # Create an instance of MadnessPipeline
    madness_pipeline = MadnessPipeline(
        train_years=[2018, 2019, 2020, 2021],
        test_year=23,
        cols=[1, 2, 3, 4, 5, 18, 19],
    )

    # Run the MadnessPipeline
    result = madness_pipeline.run()

    print(f'Result: {result}')