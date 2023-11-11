from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class TrainingDataIngestionComponent:
    def __init__(self, data_path, years, cols=[1, 2, 3, 4, 5, 18, 19]):
        self.data_path = data_path
        self.years = years
        self.cols = cols

    def execute(self):
        # Example: Reading data from CSV
        data = pd.read_csv(self.data_path)
        data = data[data.YEAR.isin(self.years)]

        X, y = self.column_selector(data, self.cols)
        y = data['RESULT']

        return X, y

class ModelTrainingComponent:
    model_dict = {
        "rfc": RandomForestClassifier
    }
    def __init__(self, model_key, param_dict):
        self.model = ModelTrainingComponent.model_dict[model_key](**param_dict)

    def execute(self, X, y):
        # Example: Training the model
        self.model.fit(X, y)

        return self.model

class InferenceDataIngestionComponent:
    def __init__(self, data_path, years, cols=[1, 2, 3, 4, 5, 18, 19]):
        self.data_path = data_path
        self.years = years
        self.cols = cols


    def execute(self, model):
        # Example: Reading data from CSV
        data = pd.read_csv(self.data_path)
        data = data[data.YEAR.isin(self.years)]
        X = self.column_selector(data, self.cols)
        return model, X

class ProbMatrixComponent:
    def __init__(self, teams):
        pass

    def execute(self, model):
        # Example: Making predictions
        predictions = model.predict(X_test)

        # Example: Evaluating model performance
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        return predictions

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
