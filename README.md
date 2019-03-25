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
Choosing a good environment was the first major obstacle for this project. Many environments exist for reinforcement learning, such as OpenAI's Gym, but I was trying to avoid using potentially heavyweight and difficult to setup environments. I also wanted the environment to have a reasonable state and action space. I ended up creating a simple text based single player game with very simple rules. The game starts on a 25 tall by 15 wide board, with a player's location indicated by a basket represented by a "U". 5 numbers in the range [1,9] are placed, spaced 5 rows apart and at random colomns. The player chooses between staying still, moving left, or moving right during their turn in order to capture the numbers as they fall. Should a number not be "caught", the number's value will be deducted from the player's score. Else, the number is added to the player's score.

This environment requires a reasonable amount of intelligence to play successfully. Players must consider the numbers' location with respect to their bucket, but also make decisions regarding potential points to be earned. For example, players might want to "sacrifice" certain lower value numbers in order to catch a larger number, resulting in more net reward.

Also, since rewards are obtained every 5 timesteps, this environment serves as a good middleground between easier to learn continous reward tasks and much difficult sparse reward tasks.

The state vector is of size 16. Each of the 5 numbers is represented by row, coloumn coordinates, and the value itself. The 16th number is the player's position.

The action vector is of size 3 for moving left, staying still, and moving right.

## Human gameplay
One advantage that this game has for human players is that it is not realtime. Players have an indefinite amount of time to consider their next move. With this in mind, the goal of this project is for the AI agent to have a score comparable to a human's. I played 8 games with 200 turns each and got an average (mean) score of 91 and a sample standard deviation of 26.

## AI Agent
PPO relies on using 2 neural networks for both training and acting: a value and policy function. Both networks take in the state vector. The policy network output softmax probabilities for the 3 potential actions, while the value function outputted just the anticipated discounted future rewards from the input state onwards.

According to the PPO paper, it's recommended to have a policy and value function to have 2 hidden layers of 64 neurons each. This was probably overkill for this task, but the number wasn't high enough for overfitting, so I stuck with that architecture.
Probably the most challenging part of implementing the agent was deciding what to set the numerous number of hyperparameters to. What made training more successful was changing the discount rate from .9 to .97, rather than any of the other hyperparameters.


Specific hyperparameters are listed in the results section per experiment.

## Experiment 1 (Learning) Results


After extensive hyperparameter tuning, I'm pleased with these results. What's interesting to me is that it takes a long time for the agent to make any meaningful progress on its learning around the iteration ( timestep) mark. I speculate that the semi-sparse  reward nature of the environment significantly slows down training.



|Hyperparameter|Value|Notes|
|--------------|-----|-----|
|Policy network architecture| 16,64,64,3 |From PPO paper|
|Value network architecture| 16,64,64,1 |From PPO paper|
|Policy network learning rate| 3e-2| OK for target probabilities to be a bit off, hence large LR|
|Value network learning rate| 3e-4| From PPO paper|
|Discount factor| 0.97||
|GAE lambda | 0.95| From PPO paper |
|Epsilon| 0.1 | For more stable training |
|Activation function (Both)| tanh | From PPO paper|
|Minibatch size (Both) | 64 | From PPO paper |
|Timesteps per training iteration | 10,000 | Large number means more stable training|
|Number of training iterations | 500 | Total of 5 Million timesteps of training |
|Number of samples for measuring performance| 100 | Makes graph less noisy |

## Experiment 2 (Epsilon) Results



## Experiment 3 (Policy function size) Results


