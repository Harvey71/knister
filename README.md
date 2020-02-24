# Knister

Deep Learning Project to create an AI to play the dice game "Knister"

## About the game

Knister is a simple but nice dice game created by Heinz Wüppen and published by NSV.

## Game rules

You can watch a quite intuitive video (in German languague) explaining the game rules here:
https://nsv.de/nsvspielttv/knister_video/

The game is played with an arbitrary number of players in 25 rounds.
Each player has a paper sheet with a 5x5 cell matrix.
In each round one player will roll two D6 dices. The number of pips will be added,
which will result in a number between 2 and 12.
Each player will write the result in one empty cell of their sheet.

When all cells are filled (after 25 rounds), the scores will be calculated.
Scores are determined per row, per column and per diagonal by the following table:

- one pair: 1 point
- one triple: 3 points
- one qudruple: 6 points
- one quintuplet: 10 points
- two pairs: 3 points
- one pair and one triplet (full house): 8 points
- street with a 7 (e.g. 5,6,7,8,9): 8 points
- street without a 7 (2,3,4,5,6 or 8,9,10,11,12): 12 points

The order of the values in a row/column/diagonal is irrelevant, e.g. 6,8,5,7,4 is a valid street.

Diagonals score double. The final score is the sum of the 5 row scores, the 5 column scores and the double scores of the two diagonals.

Typical game results for experienced players are about 50 points, depending on their strategy and luck.

## Game AI

There are 11^25*25! possible game sequences, that's ca. 10^51. Impossible to precalculate a complete game graph.

One solution is to use reinforcement learning to train a game agent.
You can find a tremendous amount on papers, examples and videos about that topic on the internet.
Unfortunately these are mostly about games with other principles (like cart pole or Atari games).

Here are some differences/caveats and how I solved them:

- Most examples have continous state values. With this game, there are discrete state values:
the values of the 25 cells and the dice result. And there is no order in the values, meaning a 10 is not the double of a 5.
The number are just symbols. Some symbols (e.g. 7) are just more likely than others (e.g. 12). That's why i used one-hot-vectors
for the state values, where the cell values have a special state (0) for empty cells.
- The action value is the cell number (0-24), where the agent will put the dice result on the sheet matrix. That's quite similar
to common examples. But this game has one strict rule: You cannot put a result to a already filled cell. So the action space gets
smaller with each round. I made sure, that forbidden moves will be eliminated in the get_action method.
- It is not possible to give a reward for each move. You could calculate a partial score whenever a row, column or diagonal is completed, 
and use this as an immediate reward, but this would train the agent to complete rows/columns/diagonals early,
which will not lead to best game results.
So I use a delayed reward in the last round and then calculate discounted rewards for each step.

After disappointing results with cross-entropy learning, I implemented a Deep-Q-Learning agent,
which reached (after seveveral hours of training) a performance whicht can compete with novice or medium experienced human players.

## How to use

The repository comes with a pre-trained model (`knister_dqn.h5`). If you just want to try the game, start the agent with `python agent.py play`.
After each round, you have to press Enter to continue. If you want to compete with the AI, either buy a knister game and
use the dice results from the agent instead rolling dices on your own, or just draw a 5x5 cell matrix on a piece of paper
and use this.

If you want to train the Deep-Q-Network on your own, you can start the agent with `python agent.py train`. Be aware: to reach
the defined performance of an average score of 50, this will take about a day, depending on your hardware.

You can change the goal altering the parameter of the train method (`agent.train(50.0)`) in the main part of `agent.py`.
Or maybe you want to try different values for the other parameters (epsilion, gamma, buffer size, train start, etc.),
or even a different neural network topology.

Please let me know your setup, when you reach a better performance! :-)

## Game sample
````
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
next dice: 1+1=2

+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  | 2|  |
+--+--+--+--+--+
next dice: 5+1=6

+--+--+--+--+--+
| 6|  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  | 2|  |
+--+--+--+--+--+
next dice: 5+1=6

+--+--+--+--+--+
| 6|  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  | 6|  |
+--+--+--+--+--+
|  |  |  | 2|  |
+--+--+--+--+--+
next dice: 4+2=6

+--+--+--+--+--+
| 6|  |  | 6|  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  | 6|  |
+--+--+--+--+--+
|  |  |  | 2|  |
+--+--+--+--+--+
next dice: 6+5=11

+--+--+--+--+--+
| 6|  |  | 6|  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  | 6|  |
+--+--+--+--+--+
|  |  |11| 2|  |
+--+--+--+--+--+
next dice: 3+4=7

+--+--+--+--+--+
| 6|  |  | 6|  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  | 6|  |
+--+--+--+--+--+
| 7|  |11| 2|  |
+--+--+--+--+--+
next dice: 6+1=7

+--+--+--+--+--+
| 6|  |  | 6|  |
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
|  |  |  |  |  |
+--+--+--+--+--+
|  |  |  | 6|  |
+--+--+--+--+--+
| 7|  |11| 2|  |
+--+--+--+--+--+
next dice: 3+4=7

+--+--+--+--+--+
| 6|  |  | 6|  |
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
|  |  | 7|  |  |
+--+--+--+--+--+
|  |  |  | 6|  |
+--+--+--+--+--+
| 7|  |11| 2|  |
+--+--+--+--+--+
next dice: 2+1=3

+--+--+--+--+--+
| 6|  |  | 6|  |
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3|  | 7|  |  |
+--+--+--+--+--+
|  |  |  | 6|  |
+--+--+--+--+--+
| 7|  |11| 2|  |
+--+--+--+--+--+
next dice: 2+6=8

+--+--+--+--+--+
| 6|  |  | 6|  |
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3|  | 7|  |  |
+--+--+--+--+--+
|  |  | 8| 6|  |
+--+--+--+--+--+
| 7|  |11| 2|  |
+--+--+--+--+--+
next dice: 3+6=9

+--+--+--+--+--+
| 6|  |  | 6|  |
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3|  | 7|  |  |
+--+--+--+--+--+
|  |  | 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2|  |
+--+--+--+--+--+
next dice: 5+2=7

+--+--+--+--+--+
| 6|  |  | 6|  |
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3|  | 7|  |  |
+--+--+--+--+--+
|  |  | 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 6+2=8

+--+--+--+--+--+
| 6|  |  | 6| 8|
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3|  | 7|  |  |
+--+--+--+--+--+
|  |  | 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 3+3=6

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3|  | 7|  |  |
+--+--+--+--+--+
|  |  | 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 3+6=9

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3|  | 7| 9|  |
+--+--+--+--+--+
|  |  | 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 1+1=2

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3| 2| 7| 9|  |
+--+--+--+--+--+
|  |  | 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 4+4=8

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3| 2| 7| 9|  |
+--+--+--+--+--+
|  | 8| 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 5+5=10

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
|  |  |  | 7|  |
+--+--+--+--+--+
| 3| 2| 7| 9|10|
+--+--+--+--+--+
|  | 8| 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 3+2=5

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
| 5|  |  | 7|  |
+--+--+--+--+--+
| 3| 2| 7| 9|10|
+--+--+--+--+--+
|  | 8| 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 3+1=4

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
| 5|  |  | 7|  |
+--+--+--+--+--+
| 3| 2| 7| 9|10|
+--+--+--+--+--+
| 4| 8| 8| 6|  |
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 6+5=11

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
| 5|  |  | 7|  |
+--+--+--+--+--+
| 3| 2| 7| 9|10|
+--+--+--+--+--+
| 4| 8| 8| 6|11|
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 5+3=8

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
| 5|  |  | 7| 8|
+--+--+--+--+--+
| 3| 2| 7| 9|10|
+--+--+--+--+--+
| 4| 8| 8| 6|11|
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 6+1=7

+--+--+--+--+--+
| 6| 6|  | 6| 8|
+--+--+--+--+--+
| 5| 7|  | 7| 8|
+--+--+--+--+--+
| 3| 2| 7| 9|10|
+--+--+--+--+--+
| 4| 8| 8| 6|11|
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 3+6=9

+--+--+--+--+--+
| 6| 6| 9| 6| 8|
+--+--+--+--+--+
| 5| 7|  | 7| 8|
+--+--+--+--+--+
| 3| 2| 7| 9|10|
+--+--+--+--+--+
| 4| 8| 8| 6|11|
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+
next dice: 6+3=9

+--+--+--+--+--+
| 6| 6| 9| 6| 8|
+--+--+--+--+--+
| 5| 7| 9| 7| 8|
+--+--+--+--+--+
| 3| 2| 7| 9|10|
+--+--+--+--+--+
| 4| 8| 8| 6|11|
+--+--+--+--+--+
| 7| 9|11| 2| 7|
+--+--+--+--+--+

Score: 53
````

Maybe you want to compete with the AI with this sample. But be fair: do not look ahead.