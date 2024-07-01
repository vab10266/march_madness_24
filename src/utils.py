import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import trange

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

best_features = [5, 6, 8, 10, 12, 14, 16, 18, 20]

def get_team_data(year):
    df = pd.read_csv(f'{DATA_DIR}/cbb{year}.csv')
    return df[['TEAM','CONF','G','W','ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O','3P_D','ADJ_T','WAB']]

def get_training_data(years):
    data = pd.read_csv(f'{DATA_DIR}/training_data.csv', index_col=0)
    data = data[data.year.isin(years)]
    return data

def run_pipe(train_start, train_end, test_year, features, model=RandomForestClassifier, params=None):
    train_df = get_training_data([train_start + x for x in range(train_end-train_start+1)])
    train_X = column_selector(train_df, features)
    train_y = train_df['result']

    if train_df.size == 0:
        return

    if params:
        clf = model(**params)
    else:
        clf = model()
    clf.fit(train_X, train_y)

    test_df = get_training_data([test_year])
    test_X = column_selector(test_df, features)
    test_y = test_df['result']

    if test_df.size == 0:
        return

    train_score = clf.score(train_X, train_y)
    test_score = clf.score(test_X, test_y)

    return train_score, test_score, clf

def make_X_from_teams(team_names, year, features):
    features_df = pd.read_csv(f"{DATA_DIR}/kenpom.csv", index_col=0)
    features_df = features_df[features_df["YEAR"] == year]

    teams_df = pd.merge(team_names, features_df, how="left", left_on="value", right_on="TEAM")
    teams_df = pd.concat((teams_df.iloc[:, -3:], teams_df.iloc[:, 1:2], teams_df.iloc[:, 2:3], teams_df.iloc[:, 6:-4]), axis=1)

    teams_df_1 = teams_df.iloc[0::2].add_prefix("A_").reset_index(drop=True)
    teams_df_2 = teams_df.iloc[1::2].add_prefix("B_").reset_index(drop=True)
    teams_df = pd.concat((teams_df_1, teams_df_2), axis=1)

    X_df = column_selector(teams_df, features)

    return X_df

def run_round_np(clf, features, start_team_names, year):
    X_df = make_X_from_teams(start_team_names, year, features)

    pred_probs = clf.predict_proba(X_df)
    r = np.random.rand(pred_probs.shape[0])
    r = (pred_probs[:, 0] < r).astype(int)
    pred_inds = np.arange(pred_probs.shape[0]) * 2 + r
    next_team_names = start_team_names.iloc[pred_inds]
    return next_team_names, r

def generate_brackets_np(clf, features, start_team_names, year, num_brackets):
    all_brackets = np.zeros((0,63))

    for i in trange(num_brackets):
        bracket = np.zeros((0,))
        team_names = start_team_names
        while team_names.shape[0] > 1:
            team_names, results = run_round_np(clf, features, team_names, year)
            bracket = np.concatenate([bracket, results])
        all_brackets = np.concatenate([all_brackets, bracket.reshape((1, -1))], axis=0)
    return all_brackets

def get_bracket(year):
    df = pd.read_csv(f'{DATA_DIR}/data_cleaned.csv')
    df = df[df['YEAR'] == int(f'20{year}')]
    return df

def make_dataset(teams, bracket):
    bracket_a = bracket.copy()
    bracket_b = bracket.copy()
    bracket_a = bracket_a.rename({'WSEED':'A_SEED', 'WTEAM':'A_TEAM', 'LSEED':'B_SEED', 'LTEAM':'B_TEAM'}, axis=1)
    bracket_b = bracket_b.rename({'WSEED':'B_SEED', 'WTEAM':'B_TEAM', 'LSEED':'A_SEED', 'LTEAM':'A_TEAM'}, axis=1)
    bracket_a['RESULT'] = 0
    bracket_b['RESULT'] = 1
    
    bracket_df = pd.concat((bracket_a, bracket_b), axis=0)
    bracket_df = bracket_df[['YEAR', 'ROUND', 'A_SEED', 'A_TEAM', 'B_SEED', 'B_TEAM', 'RESULT']]
    
    temp_df = teams.copy()
    temp_df.columns = ['A_' + str(col) for col in temp_df.columns]
    a_df = pd.merge(bracket_df[['A_SEED' ,'A_TEAM']], temp_df, 'left', on='A_TEAM').drop(['A_TEAM', 'A_CONF'], axis=1)

    temp_df = teams.copy()
    temp_df.columns = ['B_' + str(col) for col in temp_df.columns]
    b_df = pd.merge(bracket_df[['B_SEED' ,'B_TEAM']], temp_df, 'left', on='B_TEAM').drop(['B_TEAM', 'B_CONF'], axis=1)
    
    df = pd.concat((a_df, b_df, bracket_df.RESULT.reset_index(drop=True)), axis=1)
    # y = bracket_df.RESULT
    return df

def column_selector(df, cols):
    x_cols = []

    for col in cols:
        x_cols.append(df.columns[col])
    for col in cols:
        x_cols.append(df.columns[22+col])

    X = df[x_cols]
    return X

def generate_round(teams):
    a_teams = teams.copy()
    b_teams = teams.copy()
    a_teams = a_teams.iloc[::2].reset_index(drop=True)
    b_teams = b_teams.iloc[1::2].reset_index(drop=True)
    a_teams = a_teams.drop('TEAM', axis=1)
    b_teams = b_teams.drop('TEAM', axis=1)
    a_teams = a_teams.drop('CONF', axis=1)
    b_teams = b_teams.drop('CONF', axis=1)
    a_teams = a_teams.add_prefix('A_')
    b_teams = b_teams.add_prefix('B_')
    return pd.concat((a_teams, b_teams), axis=1)

def predict_round(teams, model, cols, prob=False):
    # print(teams)
    X = generate_round(teams)
    # print(X)
    
    x_cols = []
    for col in cols:
        x_cols.append(X.columns[col])
    for col in cols:
        x_cols.append(X.columns[20+col])

    X = X[x_cols]
    # print(X)
    # print(model.predict_proba(X))
    if prob == True:
        probs = model.predict_proba(X)
        r = np.random.random_sample(probs.shape[0])
        preds = probs[:, 0] > r
        return preds.astype(int)
    return model.predict(X)

def get_winning_teams(teams, preds):
    # print(teams)
    # print(preds)
    winners = pd.DataFrame()
    ind = 0
    # print(teams.shape)
    for i in range(preds.shape[0]):
        # print(ind, preds[i])
        winners = pd.concat((winners, teams.iloc[ind + preds[i]]), axis=1)
        ind += 2
    return winners.T

class Bracket:
    def __init__(self, teams, wins) -> None:
        self.teams = teams
        self.wins = wins