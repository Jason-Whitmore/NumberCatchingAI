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
    numbers = std::vector<NumberRecord>(5);
    for(int row = 0; row < 25; row += 5){
        colLocation = getRandomInt(0, 15);
        value = getRandomInt(1,9);
        //environment[row][colLocation] = '0' + value;
        n.column = colLocation;
        n.row = row;
        n.value = value;

        numbers[row / 5] = n;
    }

    score = 0;
    turnNumber = 0;
    turnLimit = 200;

    std::vector<int> config = std::vector<int>();
    config.push_back(20);
    config.push_back(20);
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
    numbers = std::vector<NumberRecord>(5);
    for(int row = 0; row < 25; row += 5){
        colLocation = getRandomInt(0, 15);
        value = getRandomInt(1,9);
        //environment[row][colLocation] = '0' + value;
        n.column = colLocation;
        n.row = row;
        n.value = value;

        numbers[row / 5] = n;
    }

    score = 0;
    turnNumber = 0;


}

void NumberCatchingAI::printGame(){
    //first, clear the board
    for(int r = 0; r < environment.size(); r++){
        
        for(int c = 0; c < environment[0].size(); c++){
            environment[r][c] = ' ';
        }
    }
    //now look at the records at create board from there
    for(int i = 0; i < numbers.size(); i++){
        environment[numbers[i].row][numbers[i].column] = '0' + numbers[i].value;
    }

    //player position
    environment[24][playerLocation] = 'U';
    
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

    playerLocation = newPos;


    //move numbers down

    NumberRecord n;
    //adjust score if needed
    NumberRecord temp;
    for(int i = 0; i < 5; i++){
        environment[numbers[i].row][numbers[i].column] = ' ';
        numbers[i].row += 1;
        if(numbers[i].row == 24){
            if(numbers[i].column == playerLocation){
                score += numbers[i].value;
            } else {
                score -= numbers[i].value;
            }

            //shift all numbers to the "right" one spot "erasing" the bottommost entry
            for(int j = numbers.size() - 1; j > 0; j--){
                numbers[j] = numbers[j - 1];
            }
            
            //i--;

            //insert new number
            n.row = 0;
            n.column = getRandomInt(0, 15);
            n.value = getRandomInt(1,9);
            numbers[0] = n;
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
    //how the encoding looks like right now: topmost ... bottommost entries, position of player
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


    //stochastic policy here
    std::vector<double> probabilities = normalizeVector(outputs);

    for(int i = 0; i < probabilities.size(); i++){
        if(NNHelper::randomDouble(0,1) < probabilities[i]){
            return i - 1;
        }
    }

    //use this return for deterministic
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

void NumberCatchingAI::printTukeySummary(std::vector<double> data){
    double min = data[0];
    double max = data[0];

    for(int i = 0; i < data.size(); i++){
        if(data[i] < min){
            min = data[i];
        }
        if(data[i] > max){
            max = data[i];
        }
    }
    //sort data to find the q2, median and q3 numbers
    std::sort(data.begin(), data.end());
    double median = data[data.size() / 2];

    double q1 = data[data.size() / 4];
    double q3 = data[(int)((3.0/4.0) * data.size())];

    std::cout << "Five number summary = (" << min << ", " << q1 << ", " << median << ", " << q3 << ", " << max << ")" << std::endl;
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
    int oldTurnNumber = turnNumber;


    setState(state);
    const int localHorizon = 15;
    const int iterations = 1000;
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
    turnNumber = oldTurnNumber;
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


    numbers = std::vector<NumberRecord>(5);
    int currentNumber = 0;
    //possible issue here of too many entries in the numbers deque
    for(int i = 0; i < state.size() - 1; i+= 3){
        n.column = state[i];
        n.row = state[i + 1];
        n.value = state[i + 2];
        numbers[currentNumber] = n;
        currentNumber++;
    }
    
    playerLocation = state[state.size() - 1];
}

double NumberCatchingAI::runGameHuman(){
    resetGame();
    std::string input;
    while(turnNumber < turnLimit){
        printGame();
        std::cout << "Current Value = " << getValue(encodeState()) << std::endl;
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

double NumberCatchingAI::probability(std::vector<double> state, int action){

    std::vector<double> probabilities = policyFunction->runNetwork(state);

    probabilities = normalizeVector(probabilities);

    return probabilities[action + 1];
}

std::vector<double> NumberCatchingAI::normalizeVector(std::vector<double> vec){
    std::vector<double> ret = std::vector<double>();


    //normalize with softmax
    double sum = 0;
    for(int i = 0; i < vec.size(); i++){
        sum += std::exp(vec[i]);
    }
    sum += 1e-5;

    for(int i = 0; i < vec.size(); i++){
        ret.push_back(std::exp(vec[i]) / sum);
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
    double oldProb = normalizeVector(networkOutput)[action + 1];

    //get new prob ratio
    policyFunction->loadNetwork("networkData.txt");
    networkOutput = policyFunction->runNetwork(state);
    double newProb = normalizeVector(networkOutput)[action + 1];

    if(newProb/oldProb > 2){
        //std::cout << "Big prob = " << newProb/oldProb << std::endl;
        //std::cout << "Numerator = " << newProb << " Denominator = " << oldProb << std::endl;
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

    std::vector<double> probs = std::vector<double>();

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
            surrogate.push_back(advantages[t] * clip(probRatio(states[t], actions[t]), 0, 10));
            surrogate.push_back(clip(probRatio(states[t], actions[t]), 1 - epsilon, 1 + epsilon) * advantages[t]);

            probs.push_back(probability(states[t], actions[t]));

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
        policyFunction->gradientDescent(0.5, epochs, learningRate);
        policyFunction->saveNetwork("networkData.txt");
        advantages.clear();
        
        nnOutputs.clear();
        states.clear();
        rewards.clear();
        
        policyFunction->loadNetwork("networkData.txt");
        
        std::cout << "Iteration " << i << " complete. Avg score = " << runGame(10) << " Loss = " << policyFunction->calculateCurrentLoss() << std::endl;
        std::cout << "Avg probability for chosen action = " << getAverage(probs) << std::endl;
        printTukeySummary(probs);
        trainInputs.clear();
        trainOutputs.clear();
        probs.clear();
    }
}



int main(){
    NumberCatchingAI n = NumberCatchingAI();

    //n.runGameHuman();

    n.trainAIPPO(1000, 1000, 100, 1e-4);

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