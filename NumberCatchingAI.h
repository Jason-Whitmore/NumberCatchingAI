
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
    double discountFactor;
    

    int playerLocation;


    void resetGame();

    double runGame();
    double runGame(int numGames);

    void performAction(int action);

    void trainAI(int numGames);

    std::vector<double> encodeStateAction(int action);
    std::vector<double> encodeState();
    
    int highestIndex(std::vector<double> outputVector);

    int getBestAction();

    static int getRandomInt(int min, int max);
    static double getAverage(std::vector<double> data);
    static double getReward(std::deque<double> scores, double discountFactor);
    static double getReward(std::vector<double> scores, double discountFactor);

    double getReward(int timeStep);

    static double min(std::vector<double> vec);
    static double clip(double value);

    double getAdvantage(int timeStep, std::vector<double> scores);
    double getValue(int timeStep, std::vector<double> scores);

    void trainAIPPO(int iterations, int timeSteps, int epochs, double learningRate);

    NeuralNetwork* Qfunction;

    void trainAIExperimental(int iterations, int numGames, double learningRate);
    private:
    

};