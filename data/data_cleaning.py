import pandas as pd
year = 22
val = pd.read_csv(f'C:\\Users\\vauda\\Documents\\work\\PS\\march_madness_24\\data\\cbb{year}.tsv', header=None, sep='\t')
val.columns = ['Rk', 'Team', 'Conf', 'W-L', 'AdjEM', 'AdjO', 'AdjO_rank', 'AdjD', 'AdjD_rank', 'AdjT', 'AdjT_rank', 'Luck', 'Luck_rank', 'AdjEM', 'AdjEM_rank', 'OppO', 'OppO_rank', 'OppD', 'OppD_rank', 'AdjEM', 'AdjEM_rank']
def get_wins(s):
    return s.split('-')[0]
val['W'] = val['W-L'].apply(get_wins)
print(val)
val.to_csv(f'C:\\Users\\vauda\\Documents\\work\\PS\\march_madness_24\\data\\cbb{year}.csv')