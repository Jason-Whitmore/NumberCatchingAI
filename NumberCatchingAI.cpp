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
    policyFunction = new NeuralNetwork(16,3,config);
    policyFunction->randomizeVariables(-.1,.1);
    policyFunction->saveNetwork("networkData.txt");
    discountFactor = 0.9;
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
                    
                    networkOutput = policyFunction->runNetwork(currentStateAction);
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
            
            policyFunction->setTrainingInputs(inputs);
            policyFunction->setTrainingOutputs(outputs);
            policyFunction->loadNetwork("networkData.txt");
            policyFunction->gradientDescent(0, inputs.size(), 0.00001);
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
        policyFunction->randomizeVariables(-5,5);
        for(int g = 1; g <= numGames; g++){


            //"nudge" variables down the loss slope
            for(int v = 0; v < policyFunction->getNumBiases() + policyFunction->getNumWeights(); v++){
                //biases
                if(v < policyFunction->getNumBiases()){
                    scoreBefore = runGame(numberOfSamples);
                    //nudge value
                    valueBefore = policyFunction->getBias(v);
                    policyFunction->setBias(v, valueBefore + verySmallNumber);
                    //record new score
                    scoreAfter = runGame(numberOfSamples);
                    //calculate derivative
                    derivative = (scoreAfter - scoreBefore) / verySmallNumber;
                    //set the weight
                    policyFunction->setBias(v, valueBefore + derivative*learningRate);

                } else {
                    //weights
                    scoreBefore = runGame(numberOfSamples);
                    //nudge value
                    valueBefore = policyFunction->getWeight(v - policyFunction->getNumBiases());
                    policyFunction->setWeight(v - policyFunction->getNumBiases(), valueBefore + verySmallNumber);
                    //record new score
                    scoreAfter = runGame(numberOfSamples);
                    //calculate derivative
                    derivative = (scoreAfter - scoreBefore) / verySmallNumber;
                    //set the weight
                    policyFunction->setWeight(v - policyFunction->getNumBiases(), valueBefore + derivative*learningRate);

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
    policyFunction->loadNetwork("networkData.txt");
    //get the game state
    std::vector<double> gameState = encodeState();

    //feed game state into neural network. Get output vector

    std::vector<double> outputs = policyFunction->runNetwork(gameState);

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

double NumberCatchingAI::positiveCount(std::vector<double> data){

    double count = 0;

    for(int i = 0; i < data.size(); i++){
        if(data[i] > 0){
            count+=1;
        }
    }

    return count / data.size();
}

double NumberCatchingAI::getStandardDeviaton(std::vector<double> data){
    double mean = getAverage(data);

    double sum = 0;

    for(int i = 0; i < data.size(); i++){
        sum += (data[i] - mean) * (data[i] - mean);
    }

    sum = sum / data.size();

    return std::sqrt(sum);
}


int NumberCatchingAI::getRandomInt(int min, int max){


    if(min == max){
        return min;
    }

    return min + (rand() % (max - min));
}

double NumberCatchingAI::runGame(){
    resetGame();
    policyFunction->loadNetwork("networkData.txt");

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
    resetGame();
    //variables for calculating reward
    double scoreBefore = 0;
    double scoreAfter = 0;

    for(int t = 0; t <= timeStep; t++){
        scoreBefore = score;
        performAction(getBestAction());
    }
    scoreAfter = score;
    resetGame();
    return scoreAfter - scoreBefore;
}

double NumberCatchingAI::getReward(int timeStep, int sampleCount){
    resetGame();
    std::vector<double> avgData = std::vector<double>();

    for(int i = 0; i < sampleCount; i++){
        avgData.push_back(getReward(timeStep));
    }

    return NumberCatchingAI::getAverage(avgData);
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

double NumberCatchingAI::clip(double value, double min, double max){
    if(value < min){
        return min;
    } else if (value > max){
        return max;
    }
    return value;
}

double NumberCatchingAI::getValue(int timeStep, std::vector<double> scores){
    resetGame();
    double sum = 0;
    for(int t = timeStep; t < scores.size(); t++){
        sum += powf64(discountFactor, t - timeStep) * scores[t];
    }
    return sum;
}

std::vector<double> NumberCatchingAI::getDeltas(std::vector<double> scores){
    std::vector<double> ret = std::vector<double>();

    for(int i = 1; i < scores.size(); i++){
        ret.push_back(scores[i] - scores[i-1]);
    }

    return ret;
}

double NumberCatchingAI::getValue(int timeStep){
    resetGame();
    const int horizonPastTimeStep = (int)(std::log10(0.01) / std::log10(discountFactor));
    std::vector<double> scores = std::vector<double>();
    double sum = 0;
    for(int t = 0; t < timeStep + horizonPastTimeStep; t++){
        if(t > timeStep){
            scores.push_back(score);
        }

        //perform action
        performAction(getBestAction());
    }

    //convert scores to score deltas
    std::vector<double> deltas = NumberCatchingAI::getDeltas(scores);

    //calculate value of the state (deltas from that state to infinity, or in our practical case, where the discount rate limits the reward to near zero)
    for(int t = timeStep; t < deltas.size(); t++){
        sum += deltas[t] * std::pow(discountFactor, t - timeStep);
    }
    //return the sum
    return sum;
}

double NumberCatchingAI::getValue(std::vector<double> state){
    std::vector<double> oldState = encodeState();
    double oldScore = score;


    setState(state);
    const int localHorizon = (int)(std::log10(0.01) / std::log10(discountFactor));
    const int iterations = 100;
    double sum = 0;
    double currentSum = 0;
    std::vector<double> rewards = std::vector<double>();
    int bestAction;

    for(int i = 0; i < iterations; i++){
        //collect rewards
        resetGame();
        currentSum = 0;
        setState(state);
        rewards.clear();
        for(int t = 0; t < localHorizon; t++){
            bestAction = NNHelper::randomInt(-1, 2);
            rewards.push_back(getReward(encodeState(), bestAction));
            performAction(bestAction);
        }
        //calculate discounted rewards
        for(int t = 0; t < rewards.size(); t++){
            currentSum += rewards[t] * std::pow(discountFactor, t);
        }
        sum += currentSum;
    }
    rewards.clear();
    setState(oldState);
    score = oldScore;
    //return average reward
    return currentSum / iterations;
}

double NumberCatchingAI::getAdvantage(int timeStep, std::vector<double> scores, std::vector<std::vector<double>> states){
    resetGame();

    double sum = -1 * getValue(states[timeStep]);

    for(int t = timeStep; t < scores.size(); t++){
        sum += std::pow(discountFactor, t - timeStep) * scores[t];
    }

    sum += std::pow(discountFactor, scores.size() - timeStep) * getValue(states[states.size() - 1]);
    return sum;
}



void NumberCatchingAI::setState(std::vector<double> state){
    numbers.clear();
    NumberRecord n;


    //Note: highest numbers are at the top
    //possible issue here of too many entries in the numbers deque
    for(int i = 0; i < state.size() - 1; i+= 3){
        n.column = state[i];
        n.row = state[i + 1];
        n.value = state[i + 2];
        numbers.push_back(n);
    }

    playerLocation = state[state.size() - 1];
}

double NumberCatchingAI::runGameHuman(){
    resetGame();
    std::string input;
    while(turnNumber < turnLimit){
        printGame();
        std::cin >> input;
        if(input == "l"){
            performAction(-1);
        } else if (input == "r"){
            performAction(1);
        } else {
            performAction(0);
        }
    }

    return score;
}

double NumberCatchingAI::getReward(std::vector<double> state, int action){
    const double c = 0.1;
    double r = 0;

    std::vector<double> prevState = encodeState();
    double oldScore = score;
    setState(state);

    double rewardBefore = score;
    performAction(action);

    r = score - rewardBefore;

    //calculate distances

    double deltaX;
    double deltaY;

    for(int i = 0; i < 5; i++){
        deltaX = numbers[i].column - playerLocation;
        deltaY = 25 - numbers[i].row;
        r += c * ((double)numbers[i].value) / std::sqrt(deltaX * deltaX + deltaY * deltaY);
    }

    //old state
    setState(prevState);
    score = oldScore;
    return r;
}

double NumberCatchingAI::sigmoid(double value){
    return 1.0/(1 + std::pow(2, -value));
}

std::vector<double> NumberCatchingAI::normalizeVector(std::vector<double> vec){
    std::vector<double> ret = std::vector<double>();

    //forward declaration
    double min(std::vector<double> v);
    //find min
    double minimum = NumberCatchingAI::min(vec);

    //normalize wrt min

    for(int i = 0; i < vec.size(); i++){
        ret.push_back(vec[i] - minimum + 1);
    }

    return ret;
}

std::vector<double> NumberCatchingAI::calculateProbabilities(std::vector<double> normalized){
    //calculate sum
    double sum = 0;

    for(int i = 0; i < normalized.size(); i++){
        sum += normalized[i];
    }

    //create return vector
    std::vector<double> prob = std::vector<double>();
    
    //calculate probabilities
    for(int i = 0; i < normalized.size(); i++){
        prob.push_back(normalized[i] / sum);
    }

    return prob;
}

double NumberCatchingAI::probRatio(std::vector<double> state, int action){
    //get old prob ratio
    policyFunction->loadNetwork("oldNetworkData.txt");
    std::vector<double> networkOutput = policyFunction->runNetwork(state);
    double oldProb = calculateProbabilities(normalizeVector(networkOutput))[action + 1];

    //get new prob ratio
    policyFunction->loadNetwork("networkData.txt");
    networkOutput = policyFunction->runNetwork(state);
    double newProb = calculateProbabilities(normalizeVector(networkOutput))[action + 1];

    if(newProb/oldProb > 10000){
        std::cout << "Big prob = " << newProb/oldProb << std::endl;
        std::cout << "Numerator = " << newProb << " Denominator = " << oldProb << std::endl;
    }
    //std::cout << "Prob ratio = " << newProb / oldProb << std::endl;
    return newProb/oldProb;
}

void NumberCatchingAI::trainAIPPO(int iterations, int timeSteps, int epochs, double learningRate){
    
    std::vector<double> scores = std::vector<double>();
    std::vector<std::vector<double>> states = std::vector<std::vector<double>>();
    std::vector<int> actions = std::vector<int>();

    std::vector<std::vector<double>> nnOutputs = std::vector<std::vector<double>>();
    std::vector<double> advantages = std::vector<double>();
    std::vector<int> ordering = std::vector<int>();
    const double epsilon = 0.2;

    std::vector<std::vector<double>> trainInputs = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> trainOutputs = std::vector<std::vector<double>>();

    std::vector<double> surrogate = std::vector<double>();

    std::vector<double> rewards = std::vector<double>();
    int bestAction;

    std::vector<double> avgScores = std::vector<double>();

    std::string csvString = "";

    for(int i = 0; i < iterations; i++){
        
        resetGame();
        //run for timesteps, collect data long the way
        for(int t = 0; t < timeSteps; t++){

            //perform action with current policy
            states.push_back(encodeState());
            nnOutputs.push_back(policyFunction->runNetwork(encodeState()));
            bestAction = getBestAction();
            actions.push_back(bestAction);
            rewards.push_back(getReward(encodeState(), bestAction));
            performAction(bestAction);
            
        }
        //std::cout << "Finished running episode." << std::endl;
        //calculate advantages for each timestep
        for(int t = 0; t < timeSteps; t++){
            advantages.push_back(getAdvantage(t, rewards, states));
        }
        //std::cout << "Num positive advantages = " << NumberCatchingAI::positiveCount(advantages) << std::endl;
        //std::cout << "Avg advantage = " << NumberCatchingAI::getAverage(advantages) << std::endl;
        //add training data based on advantages
        
        for(int t = 0; t < advantages.size(); t++){
            trainInputs.push_back(states[t]);
            surrogate.push_back(advantages[t] * probRatio(states[t], actions[t]));
            surrogate.push_back(clip(probRatio(states[t], actions[t]), 1 - epsilon, 1 + epsilon) * advantages[t]);



            nnOutputs[t][NumberCatchingAI::highestIndex(nnOutputs[t])] =  1 * min(surrogate);
            surrogate.clear();
            trainOutputs.push_back(nnOutputs[t]);
        }
        //set training data
        policyFunction->setTrainingInputs(trainInputs);
        policyFunction->setTrainingOutputs(trainOutputs);
        policyFunction->saveNetwork("oldNetworkData.txt");
        //now perform SGD optimization
        //policyFunction->trainNetwork(0.05, 10, 100, 1.5, 0.01, -10,10,false);
        policyFunction->gradientDescent(0.1, 3, learningRate);
        policyFunction->saveNetwork("networkData.txt");
        advantages.clear();
        trainInputs.clear();
        trainOutputs.clear();
        nnOutputs.clear();
        states.clear();
        rewards.clear();
        
        policyFunction->loadNetwork("networkData.txt");
        //std::cout << "Current Loss = " << policyFunction->calculateCurrentLoss() << std::endl;
        avgScores.push_back(runGame(10));
        
        std::cout << "Iteration " << i << " complete" << std::endl;
        csvString += i;
        csvString += "," + std::to_string(runGame(10)) + "\n";
        
        //std::cout << "Avg Score iteration " << i << " = " << runGame(100) << std::endl;
        //std::cout << std::endl;
    }
}



int main(){
    NumberCatchingAI n = NumberCatchingAI();


    n.trainAIPPO(1000, 10000, 10, 0.00001);

    double bestScore = -300;
    double currentScore = -300;

    for(int i = 0; i < 10000; i++){
        n.policyFunction->randomizeVariables(-10,10);
        currentScore = n.runGame(10);
        if(currentScore > bestScore){
            bestScore = currentScore;
            std::cout << "New best score = " << bestScore << " Iteration = " << i << std::endl;
            
        }

    }


    //compiles with: g++ -g -std=c++11 *.h *.cpp NeuralNetwork/*.h NeuralNetwork/*.cpp
    //n.trainAI(10000);

}