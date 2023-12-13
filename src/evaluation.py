from utils import *
import pandas as pd
import scipy.stats
from sklearn.metrics import accuracy_score

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
    inds = ran + results[start_ind:end_ind]
    # print(inds)
    next_teams = teams.iloc[inds]
    # print(next_teams)
    return next_teams

def ind_eval(true, pred):
	r1_score = (true[0:32] == pred[0:32]).sum()
	r2_score = (true[32:32] == pred[0:32]).sum()
	r3_score = (true[0:32] == pred[0:32]).sum()
	r4_score = (true[0:32] == pred[0:32]).sum()
	r1_score = (true[0:32] == pred[0:32]).sum()
	r1_score = (true[0:32] == pred[0:32]).sum()
	return r1_score
	
def group_eval(true_brackets, pred_bracket_groups):
	pass
	
class BracketEvaluator:
    def __init__(self, year):
        self.bracket_data = get_bracket(year-2)

        self.team_data = get_team_data(year)
        if not 'SEED' in self.team_data.columns:
            df = pd.read_csv(f'{DATA_DIR}/teams{year}.csv')
            self.team_data = pd.merge(df, self.team_data, 'left', on='TEAM')
        
        point_arr = []
        for i in range(6):
            num_games =  2**(5-i)
            points = 2**(i)
            point_arr += [points]*num_games
        self.points = np.array(point_arr)

        self.best_scores = None

    def _bracket_to_strings(self, bracket):
        winning_teams = None
        last_end = 0
        tmp_teams = self.team_data['TEAM']
        for i in range(6):
            num_games =  2**(5-i)
            # print('+-'*50)
            # print(num_games)
            tmp_teams = team_round(last_end, last_end + num_games, tmp_teams, bracket)
            winning_teams = pd.concat((winning_teams, tmp_teams), axis=0)
            last_end = last_end + num_games
        return winning_teams.reset_index(drop=True)
    
    def eval_one_bracket(self, bracket):
        bracket_str = self._bracket_to_strings(bracket)
        true_str = self._bracket_to_strings(real_bracket[0][0])
        return self.points[bracket_str == true_str].sum()

    def evaluate(self, brackets):
        individual_scores = np.apply_along_axis(self.eval_one_bracket, 2, brackets)
        self.best_scores = np.max(individual_scores, 1)
        mean_best_scores = np.mean(self.best_scores)

        return mean_best_scores
    
    def confidence_interval(self, confidence=0.95, brackets=None):
        if self.best_scores is None:
            self.evaluate(brackets)
        
        def mean_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
            return m, m-h, m+h
        
        return mean_confidence_interval(self.best_scores, confidence)
        
class ModelEvaluator:
    def __init__(self, years, cols) -> None:
        data_path = f'{DATA_DIR}\\training_data.csv'
        data = pd.read_csv(data_path)
        data = data[data.YEAR.isin(years)]

        self.X = column_selector(data, cols)
        self.y = data['RESULT']

    def evaluate(self, model):
        y_pred = model.predict(self.X)
        return accuracy_score(self.y, y_pred)

if __name__ == "__main__":
    
    evaluator = BracketEvaluator(year=23)
    score = evaluator.evaluate(real_bracket)
    print(f'Score: {score}')