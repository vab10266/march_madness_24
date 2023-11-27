# march_madness_24
March Madness ML pipeline project for 2024

# To Run
python .\src\main.py

# Plan
## Data
1. ~~Get Team and Bracket Data~~
2. Consolidate Team Data
3. Consolidate Bracket Data
4. Get Team Order Data
5. Consolidate Team Order Data
6. file structure
   - data
   - saved models
   - bracket predictons
   - experiments
     - experiment results
     - experiment jsons

## Models
1. ~~pipeline~~
2. change pipeline params to one passthrough dictionary
3. p=0.5
4. Method from this paper[^1]
5. Other sklearn/ML models

## Genetic Algorithm Gym
1. ~~initialize population~~
2. fitness function
3. combination method
4. mutation method
5. gym for population iteration[^2]

## Evaluator
1. ~~Bracket Score~~
2. model accuracy
3. confidence interval
4. return time

## Experimenter
1. ~~json to model pipeline~~
2. input hyper parameter ranges, output pipeline jsons
3. run pipeline jsons
4. aggregate results

# Desired outputs
## Metrics
E() -> Expected max bracket score
1. E(p=0.5)
2. E(chalky bracket)
3. E(p=model)
4. E(method from paper[^1])
5. E(GA(p=0.5))
6. E(GA(chalky bracket))
7. E(GA(p=model))
8. E(GA(method from paper[^1]))

## Graphs
Genetic Algorithm performance for all methods
1. over time
2. over populations

## Quantile Optimization
In the GA, use different fitness functions to optimize likelihood of winning in different size bracket pools
### Quantiles
1. q1
2. q2
3. q3
4. mean

### Bracket Pools
1. 10
2. 100
3. 1000

[^1]: https://arxiv.org/abs/2308.14339
[^2]: needs to return [time, best fitness, mean fitness] per iteration
