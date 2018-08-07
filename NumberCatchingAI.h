#include <iostream>
#include <vector>
#include <queue>
#include <random>
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
    std::queue<NumberRecord> numbers;

    double score;
    int turnLimit;

    int playerLocation;


    void resetGame();

    void performAction(int action);


    static int getRandomInt(int min, int max);

};