# NumberCatchingAI
An exploratory project of deep reinforcement learning inside of a custom environment

## Introduction
Reinforcement learning is interesting to me because it's a very active area of interest and the possibilities for applications are very exciting. However, like any fast moving field, it proved to be a challenge for a beginner like me to truly understand what the algorithms did and notation really meant. Resources online were often vague, and many guides on implementing algorithms relied on handy-wavy explanations.

I decided that the best way to enter the world of reinforcement learning was to implement the Proximal Policy Optimization algorithm from the ground up (with exception to my use of tensorflow/keras for neural network functionality, which I've done from scratch in the past in C++).

Goals for this project:

1. Get an agent to successfully interact with the environment

2. Explore how changing the epsilon hyperparameter effects training speed and stability

3. Explore the relationship between policy network size and agent performance.

## The Environment
Choosing a good environment was the first major obstacle for this project. Many environments exist for reinforcement learning, such as OpenAI's Gym, but I was trying to avoid using potentially heavyweight and difficult to setup environments. I also wanted the environment to have a reasonable state and action space. I ended up creating a simple text based single player game with very simple rules. The game starts on a 25 tall by 15 wide board, with a player's location indicated by a basket represented by a "U". 5 numbers in the range [1,9] are placed, spaced 5 rows apart and at random columns. The player chooses between staying still, moving left, or moving right during their turn in order to capture the numbers as they fall. Should a number not be "caught", the number's value will be deducted from the player's score. Else, the number is added to the player's score.

This environment requires a reasonable amount of intelligence to play successfully. Players must consider the numbers' location with respect to their bucket, but also make decisions regarding potential points to be earned. For example, players might want to "sacrifice" certain lower value numbers in order to catch a larger number, resulting in more net reward.

Also, since rewards are obtained every 5 timesteps, this environment serves as a good middleground between easier to learn continous reward tasks and much more difficult sparse reward tasks.

The state vector is of size 16. Each of the 5 numbers is represented by row, column coordinates, and the value itself. The 16th number is the player's position.

The action vector is of size 3 for the discrete actions of moving left, staying still, and moving right.

## Human gameplay
One advantage that this game has for human players is that it is not realtime. Players have an indefinite amount of time to consider their next move. With this in mind, the goal of this project is for the AI agent to have a score comparable to a human's. I played 14 games with 200 turns each and got an average (mean) score of 81 and a sample standard deviation of 22.97.

## AI Agent
PPO relies on using 2 neural networks for both training and acting: a value and policy function. Both networks take in the state vector. The policy network output softmax probabilities for the 3 potential actions, while the value function outputted just the anticipated discounted future rewards from the input state onwards.

According to the PPO paper, it's recommended to have a policy and value function to have 2 hidden layers of 64 neurons each. This was probably overkill for this task, but the number wasn't high enough for overfitting, so I stuck with that architecture.
Probably the most challenging part of implementing the agent was deciding what to set the numerous number of hyperparameters to. What made training more successful was changing the discount rate from 0.9 to 0.95, rather than any of the other hyperparameters.


Specific hyperparameters are listed in the results section per experiment.

## Experiment 1 (Learning) Results

![alt text](https://github.com/Jason-Whitmore/NumberCatchingAI/blob/master/exp1_graph.png "Experiment 1 results")

After extensive hyperparameter tuning, I'm pleased with these results. What's interesting to me is that it takes a long time for the agent to break even on score at iteration 200 (or 2 million timesteps). I speculate that this is due to the semi-sparse rewards slowing down training. Dips in performance are likely due to the policy diverging because of changes being to drastic.

Compared to my personal average score of 81, I would say that this experiment was a success. However, I would use the policy that scored the best versus the most recent policy, since even late in training the score does deviate.



## Experiment 2 (Epsilon) Results

![alt text](https://github.com/Jason-Whitmore/NumberCatchingAI/blob/master/exp2_graph.png "Experiment 2 results")

Note: Epsilon = 0.2 data is the same data from experiment 1, for simplicity.

The epsilon hyperparameter seems to work as intended. Over the course of training, especially on experiment 1, the performance can drop dramatically, presumably when the policy diverges too much. As the PPO paper says, the epsilon can be adjusted to control how much the policy should change.

As seen from this experiment, epsilon values of 0.2 and 0.3 resulted in both faster training, but also more diverges in the policy function. The epsilon value of 0.1 resulted in very stable and slow training which only diverged near the end.

This experiment would be further improved if I collected more than one training run per epsilon to better illustrate results. However, one run does take 16 hours on my machine, so it's not feasible for me to collect more data.


## Experiment 3 (Policy function size) Results
![alt text](https://github.com/Jason-Whitmore/NumberCatchingAI/blob/master/exp3_graph.png "Experiment 3 results")

For this experiment, I ran one training session per policy function size, then retrieved the highest score it ever achieved for the data point.

The results are close to what I expected. Small amounts of parameters don't perform as well compared to more parameters. However, large amounts of parameters offer diminishing returns.

The largest policy plotted, with 2 hidden layers of 32 nodes, performed just as well as experiment 1's 64 node hidden layer size. Ideally, we would choose the smallest policy that still achieves optimal results, as experiment 1's policy would be more prone to overfitting and would train slightly slower.

|Policy architecture|Number of parameters|Best score|
|-------------------|--------------------|----------|
|16,8,8,3|235|-170|
|16,12,12,3|399|31.3|
|16,16,16,3|595|53.51|
|16,24,24,3|1083|38.86|
|16,32,32,3|1699|85.21|

## Hyperparameters

|Hyperparameter|Value|Notes|
|--------------|-----|-----|
|Policy network architecture| 16,64,64,3 |From PPO paper|
|Value network architecture| 16,64,64,1 |From PPO paper|
|Policy network learning rate| 3e-2| OK for target probabilities to be a bit off, hence large LR|
|Value network learning rate| 3e-4| From PPO paper|
|Discount factor| 0.95||
|GAE lambda | 0.95| From PPO paper |
|Epsilon| 0.2 | From PPO paper |
|Activation function (both)| tanh | From PPO paper|
|Minibatch size (both) | 64 | From PPO paper |
|Timesteps per training iteration | 10,000 | Large number means more stable training|
|Number of training iterations | 1000 | Total of 10 Million timesteps of training |
|Number of samples for measuring performance| 100 | Makes graph less noisy |

## How to run the code
I've included 2 source files, one with a "_3_6.py" suffix. When I started this project, I originally made a version using python 2.7. I decided to convert over to python 3.6, but for some reason, it ran roughly 3 times slower. This might've been due to my installation of tensorflow/keras/python on a separate machine. The only difference code wise is that the 3.6 file has different print statements and input function calls.

Prerequisites (standard ML/AI libraries):
1. Tensorflow

2. Keras

3. Numpy

Compilation example: `python NumberCatchingAI.py w`

When running the program, you must supply a single extra command line arg, 't','p', or 'w'. 't' will train the agent, 'p' allows you to play the game yourself, 'w' will let you watch the policy stored in "BestPolicy.h5"

During training, the program will output a .csv file in append mode containing training statistics, including iteration, current mean score, and mean probability for actions taken (confidence).

Note: Training takes a very long time. 10 million timesteps takes 16 hours on a laptop with a 2.2 Ghz clock speed. I'd recommend skipping training and watching with the best policy. Please feel free to redownload/clone this repo if you unintentionally overwrote the h5 file.
