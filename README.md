# NumberCatchingAI
An exploratory project of deep reinforcement learning inside of a custom environment

## Introduction

## The Environment
During my initial interest in reinforcement learning, I originally wanted to tackle chess as an environment. However, given the 2 player nature, and relatively complicated rules, I realized I needed to be more realistic with my chosen environment. I ended up creating a simple text based 1 player game with very simple rules. The game starts on a 25 tall by 15 wide board, with a player's location indicated by a basket represented by a "U". 5 numbers in the range [1,9] are placed, spaced 5 rows apart and at random colomns. The player chooses between staying still, moving left, or moving right in order to capture the numbers as they fall. Should a number not be "caught", the number's value will be deducted from the player's score. Else, the number is added to the player's score.

This environment requires a reasonable amount of intelligence to play successfully. Players must consider the numbers' location with respect to their bucket, but also make decisions regarding potential points to be earned. For example, players might want to "sacrifice" certain lower value numbers in order to catch a larger number, resulting in more net reward.

Also, since rewards are obtained every 5 timesteps, this environment serves as a good mix between continuous and sparse reward functions.

## Human gameplay
One advantage that this game has for human players is that it is not realtime. Players have an indefinite amount of time to consider their next move. Still, even an "experienced" player may still make a less than optimal choice even with consideration. With this in mind, the goal of this project is for the AI agent to have a score comparable to a human's. I played 8 games, 200 turns each and got an average (mean) score of 91 and a sample standard deviation of 26.47.

## AI Agent
PPO relies on using 2 neural networks for both training and acting: a value and policy function. Both functions were neural networks with 16 inputs (5 sets of 3 numbers for x coordinate, y coordinate, and value for each number, plus a position for the player). The policy network output softmax probabilities for the 3 potential actions, while the value function outputted just the anticipated discounted future rewards from the input state on.

According to the PPO paper, it's recommended to have a policy and value function to have 2 hidden layers of 64 neurons each. This was probably overkill for this task, but the number wasn't high enough for overfitting, so I stuck with that architecture.
Probably the most challenging part of implementing the agent was deciding what to set the numerous number of hyperparameters to. What made training more successful was changing the discount rate from .9 to .95, rather than any of the other hyperparameters.

I also used a much lower constant epsilon of 0.05 instead of OpenAI's recommended 0.2. This was because I wanted more stable, albeit slower training.

Specific hyperparameters are found at the end of this writeup.

## Results
