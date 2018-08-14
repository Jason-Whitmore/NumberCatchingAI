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
    config.push_back(64);
    config.push_back(64);
    Qfunction = new NeuralNetwork(16,3,config);
    Qfunction->randomizeVariables(-10,10);
    Qfunction->saveNetwork("networkData.txt");
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
            i--;

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
    
    int horizon = 17;
    double discount = 0.9;
    int gameLength = 200;

    
    double bestQ;
    int bestAction;

    std::vector<std::vector<double>> inputs = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> outputs = std::vector<std::vector<double>>();
    std::vector<double> currentStateAction;
    std::vector<double> networkOutput;

    //input samples
    std::vector<std::vector<double>> stateActions = std::vector<std::vector<double>>();
    //output samples
    std::vector<double> rewards = std::vector<double>();
    

    resetGame();

    score = 0;
    //separate games
    for(int g = 1; g < numGames; g++){
        //individual turns in game:
            for(int t = 0; t < gameLength; t++){

                
                //record current score

                //pick best action with Q function Neural network
                bestQ = -9999999;
                bestAction = -1;
                for(int a = -1; a <= 1; a++){
                    currentStateAction.clear();
                    currentStateAction = std::vector<double>(encodeStateAction(a));
                    networkOutput.reserve(1);
                    
                    networkOutput = Qfunction->runNetwork(currentStateAction);
                    if(networkOutput[0] > bestQ){
                        bestQ = networkOutput[0];
                        bestAction = a;
                    }
                }
                //record the best state/action
                if(t < gameLength - horizon){
                    stateActions.push_back(encodeStateAction(bestAction));
                }
                
                
                
                //perform action
                rewards.push_back(score);
                performAction(bestAction);


                //once we have enough data collected, place data into neural network for training
                if(stateActions.size() >= horizon && t < gameLength - horizon){
                    inputs.push_back(stateActions[0]);
                    
                    outputs.push_back(std::vector<double>{getReward(rewards, discount)});
                    
                    //remove from the collections
                    stateActions.erase(stateActions.begin());
                    rewards.erase(rewards.begin());
                    
                }

            }

            //clear data collection
            

            //now that data is collection, train the network for a few iterations
            
            Qfunction->setTrainingInputs(inputs);
            Qfunction->setTrainingOutputs(outputs);
            Qfunction->loadNetwork("networkData.txt");
            Qfunction->gradientDescent(0, inputs.size(), 0.00001);
            std::cout << "Avg score = " << runGame(30) << " Iteration " << g << std::endl;
    }
    
    //inputs.clear();
    //outputs.clear();
    
}

void NumberCatchingAI::trainAIExperimental(int iterations, int numGames, double learningRate){

    double bestScore = -200;
    double currentScore;
    double verySmallNumber = 0.1;

    int numberOfSamples = 10;

    double scoreBefore;
    double scoreAfter;

    double valueBefore;
    double derivative;
    
    for(int i = 0; i < iterations; i++){
        std::cout << "Starting new iteration. Randomizing variables" << std::endl;
        Qfunction->randomizeVariables(-5,5);
        for(int g = 1; g <= numGames; g++){


            //"nudge" variables down the loss slope
            for(int v = 0; v < Qfunction->getNumBiases() + Qfunction->getNumWeights(); v++){
                //biases
                if(v < Qfunction->getNumBiases()){
                    scoreBefore = runGame(numberOfSamples);
                    //nudge value
                    valueBefore = Qfunction->getBias(v);
                    Qfunction->setBias(v, valueBefore + verySmallNumber);
                    //record new score
                    scoreAfter = runGame(numberOfSamples);
                    //calculate derivative
                    derivative = (scoreAfter - scoreBefore) / verySmallNumber;
                    //set the weight
                    Qfunction->setBias(v, valueBefore + derivative*learningRate);

                } else {
                    //weights
                    scoreBefore = runGame(numberOfSamples);
                    //nudge value
                    valueBefore = Qfunction->getWeight(v - Qfunction->getNumBiases());
                    Qfunction->setWeight(v - Qfunction->getNumBiases(), valueBefore + verySmallNumber);
                    //record new score
                    scoreAfter = runGame(numberOfSamples);
                    //calculate derivative
                    derivative = (scoreAfter - scoreBefore) / verySmallNumber;
                    //set the weight
                    Qfunction->setWeight(v - Qfunction->getNumBiases(), valueBefore + derivative*learningRate);

                }
            }


            currentScore = runGame(numberOfSamples);
            std::cout << "Current score = " << currentScore << ". Iteration " << g << std::endl;
            if(currentScore > bestScore){
                bestScore = currentScore;
                std::cout << "Best score = " << bestScore << ". Iteration " << g << std::endl;
            }

            

        }
    }
    

    
}

double NumberCatchingAI::runGame(int numGames){
    std::vector<double> avgData = std::vector<double>();

    for(int i = 0; i < numGames; i++){
        avgData.push_back(runGame());
    }
    return NumberCatchingAI::getAverage(avgData);
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


std::vector<double> NumberCatchingAI::encodeState(){
    std::vector<double> ret = std::vector<double>();

    //possible issue here of too many entries in the numbers deque
    for(int i = 0; i < numbers.size(); i++){
        ret.push_back((double)(numbers[i].column));
        ret.push_back((double)(numbers[i].row));
        ret.push_back((double)(numbers[i].value));
    }

    ret.push_back((double)playerLocation);

    return ret;
}

int NumberCatchingAI::highestIndex(std::vector<double> outputVector){
    int highestIndex = 0;
    double highestValue = outputVector[0];

    for(int i = 1; i < outputVector.size(); i++){
        if(outputVector[i] > highestValue){
            highestIndex = i;
            highestValue = outputVector[i];
        }
    }
    return highestIndex;
}


int NumberCatchingAI::getBestAction(){
    //get the game state
    std::vector<double> gameState = encodeState();

    //feed game state into neural network. Get output vector

    std::vector<double> outputs = Qfunction->runNetwork(gameState);

    //calculate the best action based on the current policy

    return highestIndex(outputs) - 1;
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
    Qfunction->loadNetwork("networkData.txt");

    int bestAction;
    double bestQ;
    for(int i = 0; i < turnLimit; i++){
        //choose action
        bestAction = getBestAction();
        //perform action
        performAction(bestAction);
    }

    return score;
}
double NumberCatchingAI::getReward(int timeStep){
    //variables for calculating reward
    double scoreBefore = 0;
    double scoreAfter = 0;

    for(int t = 0; t <= timeStep; t++){
        scoreBefore = score;
        performAction(getBestAction());
        scoreAfter = score;

    }

    return scoreAfter - scoreBefore;
}

double NumberCatchingAI::min(std::vector<double> vec){
    if(vec.size() == 0){
        std::cout << "Error: min() given vector is empty." << std::endl;
    }
    double min = vec[0];

    for(int i = 0; i < vec.size(); i++){
        if(vec[i] < min){
            min = vec[i];
        }
    }
    return min;
}

double NumberCatchingAI::clip(double value){
    if(value < -1){
        return -1;
    } else if (value > 1){
        return 1;
    }
    return value;
}

void NumberCatchingAI::trainAIPPO(int iterations, int timeSteps, int epochs, double learningRate){
    
    std::vector<double> scores = std::vector<double>();

    for(int i = 0; i < iterations; i++){
        //run for timesteps, collect data long the way
        for(int t = 0; t < timeSteps; t++){
            //get score before action
            scores.push_back(score);
            //perform action with current policy
            performAction(getBestAction());
        }

    }
}

int main(){
    NumberCatchingAI n = NumberCatchingAI();
    

    std::cout << "Random score: " << n.runGame(100) << std::endl;

    n.trainAIExperimental(10000, 100, 1);

    
    

    //compiles with: g++ -g -std=c++11 *.h *.cpp NeuralNetwork/*.h NeuralNetwork/*.cpp
    //n.trainAI(10000);

}