from utils import *
import pandas as pd

real_bracket = np.array(
    [[[
        0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,
        0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,
        1,0,0,0,1,1,1,0,
        0,0,0,0,
        0,1,
        1
    ]]]
)

def team_round(start_ind, end_ind, teams, results):
    # print(start_ind, end_ind)
    # print(teams)
    # print(results[start_ind:end_ind])
    num_games = end_ind - start_ind
    ran = np.arange(num_games) * 2
    inds = ran + results[0][0][start_ind:end_ind]
    # print(inds)
    next_teams = teams.iloc[inds]
    print(next_teams)
    return next_teams

class BracketEvaluator:
    def __init__(self, year):
        self.bracket_data = get_bracket(year-2)

        self.team_data = get_team_data(year)
        if not 'SEED' in self.team_data.columns:
            df = pd.read_csv(f'{DATA_DIR}\\teams{year}.csv')
            self.team_data = pd.merge(df, self.team_data, 'left', on='TEAM')
        
        point_arr = []
        for i in range(6):
            num_games =  2**(5-i)
            points = 2**(i)
            point_arr += [points]*num_games
        self.points = np.array(point_arr)
        print(self.points)
        print(self.points.sum())

    def _bracket_to_strings(self, bracket):
        winning_teams = None
        last_end = 0
        tmp_teams = self.team_data['TEAM']
        for i in range(6):
            num_games =  2**(5-i)
            print('+-'*50)
            print(num_games)
            tmp_teams = team_round(last_end, last_end + num_games, tmp_teams, bracket)
            winning_teams = pd.concat((winning_teams, tmp_teams), axis=0)
            last_end = last_end + num_games
        return winning_teams.reset_index(drop=True)
    
    def evaluate(self, bracket):
        print(bracket.shape)
        # print(self.bracket_data)
        print(self.team_data)
        bracket_str = self._bracket_to_strings(bracket)
        true_str = self._bracket_to_strings(real_bracket)
        print(bracket_str == true_str)
        print(self.points[bracket_str == true_str])
        print(self.points[bracket_str == true_str].sum())
        # print(generate_round(self.team_data))
        # print(self.bracket_data[self.bracket_data.ROUND == 1])
        # print(bracket[0][0][:32])
        
        # t2 = team_round(0, 32, self.team_data['TEAM'], bracket)
        # t3 = team_round(32, 32+16, t2, bracket)
        # a_teams = self.team_data.copy()
        # b_teams = self.team_data.copy()
        # a_teams = a_teams.iloc[::2].reset_index(drop=True)#['TEAM']
        # b_teams = b_teams.iloc[1::2].reset_index(drop=True)#['TEAM']
        # # a_teams = a_teams.drop('TEAM', axis=1)
        # # b_teams = b_teams.drop('TEAM', axis=1)
        # # a_teams = a_teams.drop('CONF', axis=1)
        # # b_teams = b_teams.drop('CONF', axis=1)
        # a_teams = a_teams.add_prefix('A_')
        # b_teams = b_teams.add_prefix('B_')
        # # print(a_teams["TEAM"])
        # # print(b_teams["TEAM"])
        # teams = pd.concat((a_teams, b_teams), axis=1)[["A_TEAM", "B_TEAM"]]
        # print(pd.concat((teams, teams["B_TEAM"].isin(self.bracket_data[self.bracket_data.ROUND == 1]["WTEAM"]).astype(int)), axis=1))

if __name__ == "__main__":
    
    evaluator = BracketEvaluator(year=23)
    evaluator.evaluate(real_bracket)