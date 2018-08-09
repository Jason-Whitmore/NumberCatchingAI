
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <stdlib.h>
#include "NeuralNetwork/NeuralNetwork.h"

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

    double runGame();

    void performAction(int action);

    void trainAI(int numGames);

    std::vector<double> encodeStateAction(int action);

    static int getRandomInt(int min, int max);
    static double getAverage(std::vector<double> data);
    static double getReward(std::deque<double> scores, double discountFactor);
    static double getReward(std::vector<double> scores, double discountFactor);


    private:
    NeuralNetwork* Qfunction;

};