import pandas as pd
from utils import *
from sklearn.ensemble import RandomForestClassifier
from tqdm.notebook import trange, tqdm
from time import sleep
from component_store import GeneticAlgorithmComponent
import numpy as np

def main():
    features = best_features

    results = run_pipe(2013, 2018, 2019, features)
    train_acc, test_acc, clf = results
    results
    features

    test_df = get_training_data([2019])
    test_X = column_selector(test_df, features)
    test_y = test_df['result']
    test_X
    test_year = 2019
    df = pd.read_csv(f"{DATA_DIR}/cleaned_bracket_data.csv", index_col=0)
    df = df[(df["year"] == test_year) & (df["round"] == 1)]
    start_team_names = df[["team1", "team2"]].reset_index().melt(id_vars=['index'], value_vars=['team1', 'team2']).sort_values(["index", "variable"]).reset_index(drop=True)["value"]
    start_team_names = pd.merge(start_team_names, df[["team1", "team1seed"]], how="left", left_on="value", right_on="team1").rename({"team1seed":"SEED"}, axis=1).drop("team1", axis=1)
    start_team_names = pd.merge(start_team_names, df[["team2", "team2seed"]], how="left", left_on="value", right_on="team2").drop("team2", axis=1)
    start_team_names.loc[start_team_names["SEED"].isna(), "SEED"] = start_team_names["team2seed"]
    start_team_names = start_team_names.drop("team2seed", axis=1)

    brackets = generate_brackets_np(clf, features, start_team_names, test_year, 100)
    print(brackets)

if __name__ == "__main__":
    main()
