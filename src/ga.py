import numpy as np
import pandas as pd

from tqdm import trange, tqdm
from utils import run_pipe, generate_brackets_np
import logging

L = logging.getLogger(__name__)

def data_setup(
        team_names, 
        features, 
        train_range, 
        test_year, 
        population_size, 
        brackets_per_individual=5, 
        test_brackets=100
    ):
    train_start, train_end = train_range
    train_acc, test_acc, clf = run_pipe(train_start, train_end, train_start-1, features)
    L.info(f"Classifier Scores:\nTrain: {train_acc}\nTest: {test_acc}")

    total_num_brackets = (population_size * brackets_per_individual) + test_brackets
    L.info(f"Generating {total_num_brackets} brackets...")
    raw_brackets = generate_brackets_np(clf, features, team_names, test_year, total_num_brackets)
    L.info(f"Resulting Shape: {raw_brackets.shape}")

    brackets, test_set = raw_brackets[:-test_brackets], raw_brackets[-test_brackets:]
    brackets = brackets.reshape((population_size, brackets_per_individual, 63))

    return brackets, test_set

def eval_on_bracket(ind_true_bracket, init_pop):
    ind_true_bracket = np.tile(ind_true_bracket, [init_pop.shape[0], init_pop.shape[1], 1])
    results = init_pop == ind_true_bracket
    population_size, brackets_per_individual, _ = init_pop.shape
    preds = init_pop.copy()

    rounds =  [i + 2 for i in range(5)]
    round_lens = [16 // 2 ** (round - 2) for round in rounds]
    start_inds = [32, 32+16, 32+16+8, 32+16+8+4, 32+16+8+4+2]

    arr = []
    for i in range(6):
        arr = arr + [(2**i)]*(32//(2**i))
    point_values = np.tile(np.array(arr), (population_size, brackets_per_individual, 1))


    for round, round_len, start_ind in zip(rounds, round_lens, start_inds):
        for game in [start_ind + i for i in range(round_len)]:
            prereq_game_ind = (game-(2*round_len) + (game - start_ind) + (preds[:, :, game]).astype(int))#.reshape(2, 1, 1)
            prereq_results = results[:, :, prereq_game_ind][np.arange(population_size), :, np.arange(population_size), :][:, np.arange(brackets_per_individual), np.arange(brackets_per_individual)].reshape(results[:, :, game].shape)
            updated_results = np.logical_and(prereq_results, results[:, :, game])
            results[:, :, game] = updated_results

    fitness_per_individual = (point_values * results).sum(axis=2).max(axis=1)
    return fitness_per_individual

def fitness(init_pop, test_set):
    # TODO: vectorize this to improve efficiency for larger test sets
    return np.apply_along_axis(eval_on_bracket, 1, test_set, init_pop).mean(axis=0)

def softmax(x, em=1):
    em_x = x**em
    e_x = np.exp(em_x - em_x.max())
    return e_x / e_x.sum()

def selection(population, fitnesses, num_elites=0, emphasis=1):
    # print(softmax(fitnesses, emphasis))
    pop_size = population.shape[0]
    num_children_needed = population.shape[0] - num_elites
    # print(pop_size, num_children_needed)
    left_inds = np.random.choice(pop_size, num_children_needed, True, softmax(fitnesses, emphasis))
    right_inds = np.random.choice(pop_size, num_children_needed, True, softmax(fitnesses, emphasis))
    # print(left_inds)
    return population[left_inds], population[right_inds]

def combine(left_parents, right_parents):
    mask = np.random.binomial(1, 0.5, left_parents.shape)
    children = left_parents.copy()
    children[mask == 1] = right_parents[mask == 1]
    return children

def mutate(init_pop, mutation_rate):
    noise = np.random.binomial(1, mutation_rate, init_pop.shape)
    new_pop = np.logical_xor(init_pop, noise).astype(int)
    return new_pop

def generation(population, test_set, num_elites=3, emph=0.75, mr=1/1000):
    fitnesses = fitness(population, test_set)

    left_parents, right_parents = selection(population, fitnesses, num_elites=num_elites, emphasis=emph)
    children = mutate(combine(left_parents, right_parents), mutation_rate=mr)
    # print(children.shape)
    if num_elites > 0:
        elite_inds = np.argsort(fitnesses)[-num_elites:]
        elites = population[elite_inds]

        new_population = np.concatenate((children, elites), axis=0)
    else:
        new_population = children
    return new_population

def print_individual(individual, team_names):
    for bracket_ind in range(individual.shape[0]):
        round_1 = team_names.iloc[((np.arange(32) * 2) + individual[bracket_ind][:32])].reset_index(drop=True)
        round_2 = round_1.iloc[((np.arange(16) * 2) + individual[bracket_ind][32:32+16])].reset_index(drop=True)
        round_3 = round_2.iloc[((np.arange(8) * 2) + individual[bracket_ind][32+16:32+16+8])].reset_index(drop=True)
        round_4 = round_3.iloc[((np.arange(4) * 2) + individual[bracket_ind][32+16+8:32+16+8+4])].reset_index(drop=True)
        round_5 = round_4.iloc[((np.arange(2) * 2) + individual[bracket_ind][32+16+8+4:32+16+8+4+2])].reset_index(drop=True)
        round_6 = round_5.iloc[((np.arange(1) * 2) + individual[bracket_ind][32+16+8+4+2:32+16+8+4+2+1])].reset_index(drop=True)
        print(pd.concat([round_1, round_2, round_3, round_4, round_5, round_6], axis=1).fillna(""))

# def run_gym(
#         pop,
#         num_elites = 1, 
#         emph = 0.75, 
#         mr=1/1000, 
#         test_subset_size = 40, 
#         num_gens = 10000, 
#         log_fitness=True, 
#         log_freq=1000, 
#         change_hps=True,
#         early_stop=True
# ):  
#     last_fitness = 0.0
#     best_fits = np.zeros((1))
#     mean_fits = np.zeros((1))
#     for i in trange(num_gens):
#         if change_hps:
#             test_subset_size = min(min(test_brackets, 100), int(1/50 * i + 10))
#             mr = 1/(i+100)
#         test_subset = test_set[np.random.choice(np.arange(test_set.shape[0]), min(test_subset_size, test_set.shape[0]), False)]
#         pop = generation(pop, test_subset, num_elites=num_elites, emph=emph, mr=mr)
#         # print(fits[-1])
#         if log_fitness and (i % log_freq == 0):
#             fits = fitness(pop, test_set)
#             best_fits = np.concatenate((best_fits, [fits.max()]))
#             mean_fits = np.concatenate((mean_fits, [fits.mean()]))
#             if early_stop and (fits.max() < last_fitness*1.001):
#                 return pop, best_fits, mean_fits
    
#     return pop, best_fits, mean_fits

class GeneticAlgorithmGym:
    def __init__(
            self, 
            team_names: pd.DataFrame,
            features: list,
            train_range: tuple,
            test_year: int,
            population_size: int,
            brackets_per_individual: int,
            num_test_brackets: int,
            num_elites: int = 1,
            mutation_rate: float = 1e-3,
            emphasis: float = 0.75,
            test_subset_size: int = 40,
        ):
        self.population_size = population_size
        self.brackets_per_individual = brackets_per_individual
        self.num_test_brackets = num_test_brackets
        self.num_elites = num_elites
        self.mutation_rate = mutation_rate
        self.emphasis = emphasis
        self.test_subset_size = test_subset_size

        self.population, self.test_brackets = data_setup(
            team_names=team_names,
            features=features,
            train_range=train_range, 
            test_year=test_year, 
            population_size=population_size,
            brackets_per_individual=brackets_per_individual,
            test_brackets=num_test_brackets

        )

    def eval_fitness(self):
        return fitness(self.population, self.test_brackets)
    
    def run_algorithm(
        self, 
        num_generations, 
        log_fitness=True, 
        log_freq=1000, 
        change_hyperparams=True,
        early_stop=True
    ):
        last_fitness = 0.0
        best_fits = np.zeros((1))
        mean_fits = np.zeros((1))

        for i in trange(num_generations):
            if change_hyperparams:
                self.mutation_rate /= .99
            test_subset = self.test_brackets[np.random.choice(
                np.arange(self.num_test_brackets), 
                min(self.test_subset_size, self.test_brackets.shape[0]), 
                False
            )]
            self.population = generation(
                self.population, 
                test_subset, 
                num_elites=self.num_elites, 
                emph=self.emphasis, 
                mr=self.mutation_rate
            )

            fits = self.eval_fitness()

            if log_fitness and (i % log_freq == 0):
                best_fits = np.concatenate((best_fits, [fits.max()]))
                mean_fits = np.concatenate((mean_fits, [fits.mean()]))

            if early_stop and (fits.max() < last_fitness*1.001):
                return best_fits, mean_fits
        best_fits, mean_fits = fits.max(), fits.mean()
        last_fitness = best_fits
        return best_fits, mean_fits
    
    

def main():
    from utils import get_team_names, best_features
    test_year = 2023

    start_team_names = get_team_names(test_year)
    print(start_team_names)

    
    jim = GeneticAlgorithmGym(
        team_names=start_team_names,
        features=best_features,
        train_range=(2011, test_year-1),
        test_year=test_year,
        population_size=5,
        brackets_per_individual=3,
        num_test_brackets=20,
        num_elites=1,
    )
    print(jim.population.shape)
    print(jim.test_brackets.shape)

    print(jim.run_algorithm(
        1
    ))

if __name__ == "__main__":
    main()