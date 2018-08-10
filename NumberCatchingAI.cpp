#include "NumberCatchingAI.h"


NumberCatchingAI::NumberCatchingAI(){
    srand(time(0));
    int numCols = 15;
    int numRows = 25;

    playerLocation = 8;

    std::vector<char> newRow;

    for(int r = 0; r < numRows; r++){
        newRow = std::vector<char>();
        newRow.clear();
        for(int c = 0; c < numCols; c++){
            newRow.push_back(' ');
        }

        environment.push_back(newRow);
    }

    environment[numRows - 1][playerLocation] = 'U';

    int colLocation;
    int value;
    NumberRecord n;
    for(int row = 0; row < 25; row += 5){
        colLocation = getRandomInt(0, 15);
        value = getRandomInt(1,9);
        environment[row][colLocation] = '0' + value;
        n.column = colLocation;
        n.row = row;
        n.value = value;

        numbers.push_front(n);
    }

    score = 0;
    turnNumber = 0;
    turnLimit = 200;

    std::vector<int> config = std::vector<int>();
    config.push_back(20);
    config.push_back(16);
    config.push_back(10);
    config.push_back(5);
    Qfunction = new NeuralNetwork(17,1,config);
}

void NumberCatchingAI::resetGame(){
    playerLocation = 8;

    for(int r = 0; r < 25; r++){
        for(int c = 0; c < 15; c++){
            environment[r][c] = ' ';
        }
    }
    environment[24][playerLocation] = 'U';

    int colLocation;
    int value;
    NumberRecord n;
    numbers.clear();
    for(int row = 0; row < 25; row += 5){
        colLocation = getRandomInt(0, 15);
        value = getRandomInt(1,9);
        environment[row][colLocation] = '0' + value;
        n.column = colLocation;
        n.row = row;
        n.value = value;

        numbers.push_front(n);
    }

    score = 0;
    turnNumber = 0;


}

void NumberCatchingAI::printGame(){
    for(int r = 0; r < environment.size(); r++){
        //std::cout << "-------------------------------" << std::endl;
        for(int c = 0; c < environment[0].size(); c++){
            std::cout << "|" << environment[r][c];
        }

        std::cout << "|\n";
    }

    std::cout << "-------------------------------" << std::endl;
    std::cout << std::endl << "Score: " << score << std::endl << "Turn Number: " << turnNumber << std::endl;
}

void NumberCatchingAI::performAction(int action){
    //move player
    int newPos = playerLocation + action;
    if(newPos < 0){
        newPos = 0;
    }

    if(newPos > 14){
        newPos = 14;
    }
    environment[24][playerLocation] = ' ';
    playerLocation = newPos;
    environment[24][playerLocation] = 'U';

    //minor movement penalty
    score -= 0.0;

    //move numbers down

    NumberRecord n;
    //adjust score if needed
    for(int i = 0; i < 5; i++){
        environment[numbers[i].row][numbers[i].column] = ' ';
        numbers[i].row += 1;
        if(numbers[i].row == 24){
            if(numbers[i].column == playerLocation){
                score += numbers[i].value;
            } else {
                score -= numbers[i].value;
            }

            numbers.pop_front();

            //insert new number
            n.row = 0;
            n.column = getRandomInt(0, 15);
            n.value = getRandomInt(1,9);
            numbers.push_back(n);
        }
        
    }

    //adjust environment
    for(int i = 0; i < numbers.size(); i++){
        //std::cout << "r " << numbers[i].row << " c " << numbers[i].column << std::endl;
        environment[numbers[i].row][numbers[i].column] = '0' + numbers[i].value;
    }
    turnNumber++;
}


void NumberCatchingAI::trainAI(int numGames){
    Qfunction->randomizeVariables(-1,1);

    std::vector<double> avgData = std::vector<double>();

    
    std::deque<double> totalReward = std::deque<double>();
    int horizon = 20;
    double discount = 0.9;
    int gameLength = 1000;

    
    double bestQ;
    int bestAction;

    Data* inputs = new Data(51,17);
    Data* outputs = new Data(51,1);
    int currentDataIndex = 0;
    std::vector<double> currentStateAction;
    std::vector<double> networkOutput;

    std::vector<double> stateAction = std::vector<double>();
    std::vector<double> rewards = std::vector<double>();
        
    currentDataIndex = 0;

    resetGame();

    score = 0;
    
    //individual turns in game:
    for(int t = 0; t < gameLength; t++){

        
        //record current score

        //pick best action with Q function Neural network
        bestQ = -9999999;
        bestAction = -1;
        for(int a = -1; a <= 1; a++){
            currentStateAction.clear();
            currentStateAction.reserve(17);
            currentStateAction = std::vector<double>(encodeStateAction(a));
            networkOutput.reserve(1);
            
            networkOutput = Qfunction->runNetwork(currentStateAction);
            if(networkOutput[0] > bestQ){
                bestQ = networkOutput[0];
                bestAction = a;
            }
        }
        //record the best state/action
        if(stateAction.empty()){
            stateAction = encodeStateAction(bestAction);
        }
        //stateActions.push_back(encodeStateAction(bestAction));
        //perform action
        rewards.push_back(score);
        performAction(bestAction);


        //once we have enough data collected, place data into neural network for training
        if(turnNumber % horizon == 0){
            for(int c = 0; c < inputs->getNumCols(); c++){
                inputs->setIndex(currentDataIndex, c, stateAction[c]);
            }
            outputs->setIndex(currentDataIndex, 0, getReward(rewards, discount));
            //remove from the collection
            stateAction.clear();
            rewards.clear();
            currentDataIndex++;
            
        }

    }

    //clear data collection
    totalReward.clear();
    

    //now that data is collection, train the network for a few iterations
    Qfunction->setTrainingInputs(inputs);
    Qfunction->setTrainingOutputs(outputs);
    Qfunction->gradientDescent(0, 10, 0.0001);
    
}

std::vector<double> NumberCatchingAI::encodeStateAction(int action){
    std::vector<double> ret = std::vector<double>();
    ret.reserve(17);

    //possible issue here of too many entries in the numbers deque
    for(int i = 0; i < numbers.size(); i++){
        ret.push_back((double)(numbers[i].column));
        ret.push_back((double)(numbers[i].row));
        ret.push_back((double)(numbers[i].value));
    }

    ret.push_back((double)playerLocation);
    ret.push_back((double)action);

    return ret;
}

double NumberCatchingAI::getReward(std::deque<double> scores, double discountFactor){
    //create new collection to store reward deltas
    std::vector<double>* deltas = new std::vector<double>();
    for(int i = 0; i < scores.size() - 1; i++){
        deltas->push_back(scores[i + 1] - scores[i]);
    }

    double ret = 0;
    double currentDiscount = 1;
    for(int i = 0; i < deltas->size(); i++){
        ret += currentDiscount * deltas->at(i);
        currentDiscount *= discountFactor;
    }
    delete deltas;
    return ret;
}

double NumberCatchingAI::getReward(std::vector<double> scores, double discountFactor){
    //create new collection to store reward deltas
    std::vector<double>* deltas = new std::vector<double>();
    for(int i = 0; i < scores.size() - 1; i++){
        deltas->push_back(scores[i + 1] - scores[i]);
    }

    double ret = 0;
    double currentDiscount = 1;
    for(int i = 0; i < deltas->size(); i++){
        ret += currentDiscount * deltas->at(i);
        currentDiscount *= discountFactor;
    }
    delete deltas;
    return ret;
}


double NumberCatchingAI::getAverage(std::vector<double> data){
    double ret = 0;

    for(int i = 0; i < data.size(); i++){
        ret += data[i];
    }

    return ret / data.size();
}


int NumberCatchingAI::getRandomInt(int min, int max){


    if(min == max){
        return min;
    }

    return min + (rand() % (max - min));
}

double NumberCatchingAI::runGame(){
    resetGame();

    int bestAction;
    double bestQ;
    for(int i = 0; i < turnLimit; i++){
        //choose action
        bestAction = -1;
        bestQ = -100000000000000;
        for(int a = -1; a <= 1; a++){
            if(Qfunction->runNetwork(encodeStateAction(a)).at(0) > bestQ){
                bestQ = Qfunction->runNetwork(encodeStateAction(a)).at(0);
                bestAction = a;
            }
        }

        //perform action
        performAction(bestAction);
    }

    return score;
}


int main(){
    NumberCatchingAI n = NumberCatchingAI();

    std::vector<double> avgData = std::vector<double>();


    //compiles with: g++ -std=c++11 *.h *.cpp NeuralNetwork/*.h NeuralNetwork/*.cpp
    for(int i = 0; i < 1000; i++){
        n.trainAI(1);

        avgData.push_back(n.runGame());

        if(i % 10 == 0){
            std::cout << "Avg Score = " << NumberCatchingAI::getAverage(avgData) << std::endl;
            avgData.clear();
        }
    }

}