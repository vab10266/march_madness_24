from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils import *

class TrainingDataIngestionComponent:
    def __init__(self, years, cols=[1, 2, 3, 4, 5, 18, 19]):
        self.years = years
        self.cols = cols

    def execute(self, _):
        print("TrainingDataIngestionComponent")
        # Example: Reading data from CSV
        data_path = f'{DATA_DIR}\\training_data.csv'
        print(data_path)
        data = pd.read_csv(data_path)
        print(data)
        data = data[data.YEAR.isin(self.years)]
        print(data)

        X = column_selector(data, self.cols)
        y = data['RESULT']
        print(X)
        print(y)
        return X, y

class ModelTrainingComponent:
    model_dict = {
        "rfc": RandomForestClassifier
    }
    def __init__(self, model_key, param_dict={}):
        self.model = ModelTrainingComponent.model_dict[model_key](**param_dict)

    def execute(self, input):
        print("ModelTrainingComponent")
        print(input)
        X, y = input
        # Example: Training the model
        self.model.fit(X, y)

        return self.model

class InferenceBracketComponent:
    def __init__(self, year, cols=[1, 2, 3, 4, 5, 18, 19]):
        self.year = year
        self.cols = cols
        
        self.team_data = get_team_data(year)
        if not 'SEED' in self.team_data.columns:
            df = pd.read_csv(f'{DATA_DIR}\\teams{year}.csv')
            self.team_data = pd.merge(df, self.team_data, 'left', on='TEAM')


    def execute(self, model):
        print("InferenceBracketComponent")
        # Example: Reading data from CSV
        r1_preds = predict_round(self.team_data, model, self.cols)
        r2_teams = get_winning_teams(self.team_data, r1_preds)
        print(r2_teams)
        print('='*100)
        r2_preds = predict_round(r2_teams, model, self.cols)
        r3_teams = get_winning_teams(r2_teams, r2_preds)
        print(r3_teams)
        print('='*100)
        r3_preds = predict_round(r3_teams, model, self.cols)
        r4_teams = get_winning_teams(r3_teams, r3_preds)
        print(r4_teams)
        print('='*100)
        r4_preds = predict_round(r4_teams, model, self.cols)
        r5_teams = get_winning_teams(r4_teams, r4_preds)
        print(r5_teams)
        print('='*100)
        r5_preds = predict_round(r5_teams, model, self.cols)
        r6_teams = get_winning_teams(r5_teams, r5_preds)
        print(r6_teams)
        print('='*100)
        r6_preds = predict_round(r6_teams, model, self.cols)
        r7_teams = get_winning_teams(r6_teams, r6_preds)
        print(r7_teams)
        return r7_teams

class GeneticAlgorithmComponent:
    def __init__(self, population_size, generations):
        self.population_size = population_size
        self.generations = generations

    def execute(self, initial_population, X_test, y_test):
        population = initial_population

        for generation in range(self.generations):
            # Evaluate fitness of each individual in the population
            fitness_scores = self.evaluate_fitness(population, X_test, y_test)

            # Select top performers
            selected_indices = self.selection(fitness_scores)

            # Crossover to create new individuals
            new_population = self.crossover(population, selected_indices)

            # Mutate new individuals
            new_population = self.mutation(new_population)

            population = new_population

        # Return the final evolved population
        return population

    def evaluate_fitness(self, population, X_test, y_test):
        fitness_scores = []

        for individual in population:
            # Assuming individual is a set of parameters for the model
            model = RandomForestClassifier(random_state=42, **individual)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            fitness_scores.append(accuracy)

        return np.array(fitness_scores)

    def selection(self, fitness_scores):
        # Select top performers based on fitness scores
        selected_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
        return selected_indices

    def crossover(self, population, selected_indices):
        new_population = []

        for _ in range(self.population_size // 2):
            parent1 = population[np.random.choice(selected_indices)]
            parent2 = population[np.random.choice(selected_indices)]

            # Simple crossover: Take the average of parameters
            child = {}
            for key in parent1:
                child[key] = (parent1[key] + parent2[key]) / 2.0

            new_population.append(child)

        return population + new_population

    def mutation(self, population):
        # Randomly mutate some parameters
        for individual in population:
            for key in individual:
                if np.random.rand() < 0.1:  # 10% chance of mutation
                    individual[key] += np.random.normal(scale=0.1)  # Small random change

        return population
