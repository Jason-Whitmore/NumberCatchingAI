#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <tuple>



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
    std::vector<NumberRecord> numbers;

    std::vector<double> getGradPolicy(std::vector<double> state, int action);

    void updatePolicy(std::vector<int> batchIndicies, std::vector<std::vector<double>> states, std::vector<int> actions, std::vector<int> advantages, double epsilon, double learningRate);
    void PPOupdate(std::vector<double> state, int action, double advantage, double epsilon, double learningRate);
    void trainNetwork(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs, uint iterations, double learningRate);
    std::vector<double> networkPredict(std::vector<double> inputs);
    

    double score;
    int turnLimit;
    int turnNumber;
    double discountFactor;
    

    int playerLocation;


    void resetGame();

    double runGame();

    void performAction(int action);

    void trainAI(int numGames);

    std::vector<double> encodeStateAction(int action);
    std::vector<double> encodeState();
    std::vector<double> normalizeState(std::vector<double> state);
    
    int highestIndex(std::vector<double> outputVector);

    int getBestAction();

    static int randomInt(int min, int max);
    static double randomDouble(double min, double max);
    

    static double getAverage(std::vector<double> data);
    static double getStandardDeviation(std::vector<double> data);
    static double getMedian(std::vector<double> data);
    static void printTukeySummary(std::vector<double> data);
    static double positiveCount(std::vector<double> data);
    
    double runGameHuman();
    
    static std::vector<double> getDeltas(std::vector<double> scores);

    static double getReward(std::deque<double> scores, double discountFactor);
    static double getReward(std::vector<double> scores, double discountFactor);
    double getReward(std::vector<double> state, int action);

    double probability(std::vector<double> state, int action);
    static double sigmoid(double value);


    double probRatio(std::vector<double> state, int action);
    double avgProbRatio(std::vector<std::vector<double>> states, std::vector<int> actions);

    static std::vector<double> applySoftmax(std::vector<double> vec);
    void printStateInformation(std::vector<std::vector<double>> states);
    static std::vector<double> calculateProbabilities(std::vector<double> normalized);

    double getReward(int timeStep);
    double getReward(int timeStep, int sampleCount);

    void SGDIteration(int timeStep, double advantage, double learningRate);

    static bool isNormalizedState(std::vector<double> state);
    

    static double min(std::vector<double> vec);
    static double clip(double value, double min, double max);

    double getAdvantage(int timeStep, std::vector<double> scores, std::vector<std::vector<double>> states);
    double getAdvantageGAE(int timeStep, std::vector<double> scores, std::vector<std::vector<double>> states);
    double getAdvantageGAE(int timeStep, std::vector<double> GAEdeltas);

    void setState(std::vector<double> state);

    double getValue(int timeStep, std::vector<double> scores);
    double getValue(int timeStep);
    double getValue(std::vector<double> state);

    void trainAIPPO(int iterations, int timeSteps, int epochs, int minibatchSize, double learningRate);



    double getAvgDiscountedReward(int timeSteps);
    void trainAIGradient(int iterations, int timeSteps, int epochs, double learningRate);

    void trainAISupervised(int epochs);


    NeuralNetwork policyFunction;

    NeuralNetwork valueFunction;





    void trainAIExperimental(int iterations, int numGames, double learningRate);

    void watchGame();
    

};