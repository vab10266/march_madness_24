import os
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\data'))

def get_team_data(year):
    df = pd.read_csv(f'{DATA_DIR}\\cbb{year}.csv')
    return df[['TEAM','CONF','G','W','ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O','3P_D','ADJ_T','WAB']]

def get_bracket(year):
    df = pd.read_csv(f'{DATA_DIR}\\data_cleaned.csv')
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
        x_cols.append(df.columns[20+col])

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
    print(model.predict_proba(X))
    if prob == True:
        return model.predict_proba(X)
    return model.predict(X)

def get_winning_teams(teams, preds):
    # print(teams)
    print(preds)
    winners = pd.DataFrame()
    ind = 0
    # print(teams.shape)
    for i in range(preds.shape[0]):
        # print(ind, preds[i])
        winners = pd.concat((winners, teams.iloc[ind + preds[i]]), axis=1)
        ind += 2
    return winners.T
