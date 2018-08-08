



#include "NeuralNetwork/NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <stdlib.h>


struct NumberRecord{
    int row;
    int column;
    int value;
};

class NumberCatchingAI{
    public:

    NumberCatchingAI();
    void printGame();
    
    std::vector<std::vector<char>> environment;
    std::deque<NumberRecord> numbers;

    double score;
    int turnLimit;
    int turnNumber;

    int playerLocation;


    void resetGame();

    void performAction(int action);


    static int getRandomInt(int min, int max);

};