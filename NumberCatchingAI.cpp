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
    int randomInt(int min,int max);
    for(int row = 0; row < 25; row += 5){
        colLocation = 0 + (rand() % (15 - 0));
        value = 1 + (rand() % (9 - 1));
        //environment[row][colLocation] = '0' + value;
        n.column = colLocation;
        n.row = row;
        n.value = value;

        numbers[row / 5] = n;
    }

    score = 0;
    turnNumber = 0;
    turnLimit = 200;


    policyFunction = NeuralNetwork(16,64,64,3);
    policyFunction.randomizeNetwork(-0.01, 0.01);


    valueFunction = NeuralNetwork(16,64,64,1);
    valueFunction.setActivationFunction(1, ActivationFunction::Tanh);
    valueFunction.setActivationFunction(2, ActivationFunction::Tanh);
    valueFunction.randomizeNetwork(-0.01, 0.01);

    Qfunction = NeuralNetwork(17,64,64,1);
    Qfunction.randomizeNetwork(0,1);
    
    discountFactor = 0.9;


    
}

std::vector<double> NumberCatchingAI::networkPredict(std::vector<double> inputs){

    return policyFunction.compute(inputs);
}

void NumberCatchingAI::PPOupdate(std::vector<double> state, int action, double advantage, double epsilon, double learningRate){
    const double delta = 1e-6;

    int numWeights = policyFunction.connections.size();

    //calculate gradient
    std::vector<double> gradient = std::vector<double>();
    double before;
    double after;

    //determine if we need to demote or promote action
    double currentRatio = probRatio(state, action);
    //std::cout << "Current Ratio = " << std::abs(1 - currentRatio) << std::endl;

    for(int i = 0; i < numWeights; i++){
        before = probability(state, action);
        //nudge weight
        policyFunction.connections[i]->weight += delta;

        after = probability(state, action);

        //nudge back
        policyFunction.connections[i]->weight -= delta;

        //push back gradient
        gradient.push_back((after - before)/ delta);
    }



    
    if(currentRatio > 1 + epsilon){
        //always demote
        for(int i = 0; i < numWeights; i++){
            policyFunction.connections[i]->weight -= learningRate * gradient[i];
        }

    } else if (currentRatio < 1 - epsilon){
        //always promote
        for(int i = 0; i < numWeights; i++){
            policyFunction.connections[i]->weight += learningRate * gradient[i];
        }

    } else if (advantage > 0){
        //promote
        for(int i = 0; i < numWeights; i++){
            policyFunction.connections[i]->weight += learningRate * gradient[i];
        }

    } else if (advantage < 0){
        //demote
        for(int i = 0; i < numWeights; i++){
            policyFunction.connections[i]->weight -= learningRate * gradient[i];
        }

    }

    //save to current params
    policyFunction.saveNetwork("paramsCurrent");
}

double NumberCatchingAI::randomDouble(double min, double max) {
	double scalar = (double)rand() / RAND_MAX;

	return min + (scalar * (max - min));
}

int NumberCatchingAI::randomInt(int min, int max) {
	
	return min + (rand() % (max - min));
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
    int randomInt(int,int);
    for(int row = 0; row < 25; row += 5){
        colLocation = 0 + (rand() % (15 - 0));
        value = 1 + (rand() % (9 - 1));
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
            n.column = randomInt(0, 15);
            n.value = randomInt(1,9);
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
std::vector<double> NumberCatchingAI::normalizeState(std::vector<double> state){
    std::vector<double> ret = std::vector<double>(state);
    for(int i = 0; i < ret.size() - 1; i+=3){
        ret[i] /= 15.0;
        ret[i+1] /= 25.0;
        ret[i+2] /= 9.0;
    }

    ret[ret.size()-1] /= 15.0;
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
    //policyFunction->loadNetwork("networkData.txt");
    //get the game state
    std::vector<double> gameState = encodeState();

    //feed game state into neural network. Get output vector

    std::vector<double> outputs = networkPredict(gameState);

    //calculate the best action based on the current policy


    //stochastic policy here
    std::vector<double> probabilities = applySoftmax(outputs);
    double random = randomDouble(0, 1);
    for(int i = 0; i < probabilities.size(); i++){
        if(random < probabilities[i]){
            if(probabilities[i] < 1e-3){
                //std::cout << "break here" << std::endl;
            }
            return i - 1;
        }
        random -= probabilities[i];
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

static double getMedian(std::vector<double> data){
    std::sort(data.begin(), data.end());

    return data[data.size() / 2];
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



int NumberCatchingAI::runGame(){
    resetGame();
    policyFunction.loadNetwork("paramsOld");

    int bestAction;
    double bestQ;
    std::vector<double> probs = std::vector<double>();
    for(int i = 0; i < turnLimit; i++){
        //choose action
        bestAction = getBestAction();
        probs.push_back(probability(encodeState(), bestAction));
        //perform action
        performAction(bestAction);
    }

    double avg = getAverage(probs);
    probs.clear();
    std::vector<double> r = std::vector<double>();
    r.push_back(score);
    r.push_back(avg);
    return score;
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



double NumberCatchingAI::getValue(std::vector<double> state){

    std::vector<double> input = state;
    input = normalizeState(state);

    //NN thing here
    std::vector<double> output = valueFunction.compute(input);
    //std::cout << "Predicted value = " << output[0] * 10 << std::endl;
    //return output[0] * 10;

    const int numSamples = 25;
    double reward = 0;
    policyFunction.loadNetwork("paramsOld");
    for(int i = 0; i < numSamples; i++){
        setState(state);
        for(int t = 0; t < 25; t++){
            int action = getBestAction();
            reward += getReward(state, action) * std::pow(discountFactor, t);

        }
    }
    policyFunction.loadNetwork("paramsCurrent");
    return reward / numSamples;



    std::vector<double> oldState = encodeState();
    double oldScore = score;
    int oldTurnNumber = turnNumber;


    setState(state);
    const int localHorizon = 15;
    const int iterations = 1e4;
    const double stddevTarget = 0.1;
    double sum = 0;
    double currentSum = 0;
    std::vector<double> rewards = std::vector<double>();
    int bestAction;
    std::vector<double> values = std::vector<double>();

    double currentStdDev = 10;
    for(int i = 0; i < iterations && currentStdDev > stddevTarget; i++){
        //collect rewards
        resetGame();
        currentSum = 0;
        setState(state);
        rewards.clear();
        for(int t = 0; t < localHorizon; t++){

            bestAction = randomInt(-1, 2);
            rewards.push_back(getReward(encodeState(), bestAction));
            performAction(bestAction);
        }
        //calculate discounted rewards
        for(int t = 0; t < rewards.size(); t++){
            currentSum += rewards[t] * std::pow(discountFactor, t);
        }
        values.push_back(currentSum);
        if(i % 10 == 0){
            currentStdDev = getStandardDeviaton(values);

        }
    }
    rewards.clear();
    setState(oldState);
    score = oldScore;
    turnNumber = oldTurnNumber;
    //return average reward
    
    return getAverage(values);
    exit(0);
}

double NumberCatchingAI::getAdvantage(int timeStep, std::vector<double> scores, std::vector<std::vector<double>> states){
    resetGame();

    double sum = -1 * getValue(states[timeStep]);

    const double localHorizon = 25;

    for(int t = timeStep; t < t + localHorizon && t < scores.size(); t++){
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
        //std::cout << "Value = " << getValue(encodeState()) << std::endl;
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

    double continousReward;
    for(int i = 0; i < 5; i++){
        deltaX = std::abs(numbers[i].column - playerLocation);
        deltaY = 25 - numbers[i].row;
        continousReward = (deltaY - deltaX) * c * ((double)numbers[i].value) / std::sqrt(deltaX * deltaX + deltaY * deltaY);

        //r += std::max(0.0, continousReward);
    }

    //add bad movements
    if(action == -1 && playerLocation == 0){
        r -= 5;
    }

    if(action == 1 && playerLocation == 14){
        r -= 5;
    }

    //old state
    setState(prevState);
    score = oldScore;
    return r;
}

double NumberCatchingAI::probability(std::vector<double> state, int action){

    std::vector<double> probabilities = networkPredict(state);

    probabilities = applySoftmax(probabilities);

    double r = probabilities[action + 1];


    return probabilities[action + 1];
}

std::vector<double> NumberCatchingAI::applySoftmax(std::vector<double> vec){
    std::vector<double> ret = std::vector<double>();


    //normalize with softmax
    double sum = 0;
    for(int i = 0; i < vec.size(); i++){
        sum += std::exp(vec[i]);
    }

    for(int i = 0; i < vec.size(); i++){
        ret.push_back(std::exp(vec[i]) / sum);
    }

    for(int i = 0; i < ret.size(); i++){
        if(std::isnan(ret[i])){
            ret[i] = 1;
        }

        if(ret[i] < 1e-8){
            ret[i] = 1e-8;
        }
    }
    sum = 0;
    for(int i = 0; i < ret.size(); i++){
        sum += ret[i];
    }

    for(int i = 0; i < ret.size(); i++){
        //ret[i] /= sum;
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
    policyFunction.loadNetwork("paramsOld");
    std::vector<double> networkOutput = networkPredict(state);
    double oldProb = probability(state, action);

    //get current prob ratio
    policyFunction.loadNetwork("paramsCurrent");
    networkOutput = networkPredict(state);
    double newProb = probability(state, action);


    //std::cout << "Prob ratio = " << newProb / oldProb << std::endl;
    return newProb/oldProb;
}

void NumberCatchingAI::trainAIPPO(int iterations, int timeSteps, int epochs, double learningRate){

    //policyFunction.saveNetwork("paramsCurrent");
    //policyFunction.saveNetwork("paramsOld");
    policyFunction.loadNetwork("paramsOld");
    
    std::vector<double> scores = std::vector<double>();
    std::vector<std::vector<double>> states = std::vector<std::vector<double>>();
    std::vector<int> actions = std::vector<int>();

    std::vector<double> advantages = std::vector<double>();
    std::vector<int> ordering = std::vector<int>();
    const double epsilon = 0.2;

    std::vector<double> rewards = std::vector<double>();
    int bestAction;

    std::vector<double> probs = std::vector<double>();
    std::vector<double> probR = std::vector<double>();

    std::vector<std::vector<double>> valueFunctionInputs = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> valueFunctionOutputs = std::vector<std::vector<double>>();

    std::string csvString = "";

    for(int i = 0; i < iterations; i++){
        
        resetGame();
        //run for timesteps, collect data along the way
        //std::cout << "Collecting data..." << std::endl;
        for(int t = 0; t < timeSteps; t++){

            //perform action with current policy
            states.push_back(encodeState());
            bestAction = getBestAction();
            actions.push_back(bestAction);
            probs.push_back(probability(encodeState(), bestAction));
            rewards.push_back(getReward(encodeState(), bestAction));
            performAction(bestAction);
            
        }

        valueFunctionInputs.clear();
        valueFunctionOutputs.clear();

        double sum = 0;
        //get the correct value training data
        std::vector<double> temp;
        for(int s = 0; s < states.size() - 15; s++){

            valueFunctionInputs.push_back(states[s]);
            sum = 0;
            for(int t = s; t < s + 15; t++){
                sum += rewards[t] * std::pow(discountFactor, t - s);
            }
            valueFunctionOutputs.push_back(std::vector<double>(1, sum));
        }

        for(int r = 0; r < valueFunctionInputs.size(); r++){
            valueFunctionInputs[r] = normalizeState(valueFunctionInputs[r]);
            valueFunctionOutputs[r][0] = valueFunctionOutputs[r][0] / 10.0;
        }

        //set training dataset
        valueFunction.trainingInputs = valueFunctionInputs;
        valueFunction.trainingOutputs = valueFunctionOutputs;
        //now train the value function
        //std::cout << "Training value function...." << std::endl;
        

        //calculate advantages for each timestep
        for(int t = 0; t < timeSteps; t++){
            advantages.push_back(getAdvantage(t, rewards, states));
        }
        //valueFunction.stochasticGradientDescent(0.0001, timeSteps * 30, learningRate);
        
        for(int t = 0; t < advantages.size(); t++){
            probR.push_back(probRatio(states[t], actions[t]));
            probs.push_back(probability(states[t], actions[t]));

        }

        
        
        //states.clear();
        rewards.clear();
        

        //perform PPO updates
        
        int randomDataIndex;
        //std::cout << "Updating policy...." << std::endl;
        int iterations = timeSteps * 4;
        policyFunction.saveNetwork("paramsCurrent");
        for(int t = 0; t < 100; t++){
            randomDataIndex = rand() % states.size();
            PPOupdate(states[randomDataIndex], actions[randomDataIndex], advantages[randomDataIndex], epsilon, learningRate);
        }

        policyFunction.saveNetwork("paramsOld");

        double results = runGame();
        std::cout << "Iteration " << i << " complete. Avg score = " << results << std::endl;
        std::cout << "Avg loss for value network: " << valueFunction.calculateAverageLoss() << std::endl;
        std::cout << "Avg advantage = " << getAverage(advantages) << std::endl;
        probR.clear();
 

        advantages.clear();
        probs.clear();
        states.clear();

        
        
        //set old policy
        
    }
}

void NumberCatchingAI::trainAIQ(int iterations, int timeSteps, int epochs, double learningRate){
    std::vector<std::vector<double>> persistantInputs = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> persistantOutputs = std::vector<std::vector<double>>();


    std::vector<std::vector<double>> stateActions = std::vector<std::vector<double>>();
    std::vector<double> rewards = std::vector<double>();

    const double epsilon = 0.98;
    const double discountFactor = 0.95;
    const int maxStorageSize = timeSteps * 10;

    for(int iter = 0; iter < iterations; iter++){
        //Run game for timesteps. Collect data.
        std::cout << "Collecting data..." << std::endl;
        for(int step = 0; step < timeSteps; step++){
            if(randomDouble(0,1) < std::pow(epsilon, iter)){
                //exploration, choose randomly.
                int action = randomInt(-1,1);
                stateActions.push_back(encodeStateAction(action));
                rewards.push_back(getReward(encodeState(), action));
                performAction(action);
            } else {
                //exploitation, choose the best action
                int action = -1;
                double bestReward = Qfunction.compute(encodeStateAction(-1)).at(0);
                for(int i = -1; i < 2; i++){
                    if(Qfunction.compute(encodeStateAction(i)).at(0) > bestReward){
                        action = i;
                        bestReward = Qfunction.compute(encodeStateAction(i)).at(0);
                    }
                }
                rewards.push_back(getReward(encodeState(), action));
                performAction(action);
                stateActions.push_back(encodeStateAction(action));
            }
        }

        std::cout << "Calculating rewards..." << std::endl;
        //calculate the cumulative reward.
        for(int t = 0; t < timeSteps; t++){
            double reward = 0;
            for(int n = t; n < timeSteps; n++){
                reward += std::pow(discountFactor, n - t) * rewards[n];
            }

            persistantInputs.push_back(stateActions[t]);
            persistantOutputs.push_back(std::vector<double>(1, reward / 10.0));
        }

        //remove entries at random from persistent storage if we are over a certain number of data points
        while(persistantInputs.size() > maxStorageSize){
            int randomIndex = randomInt(0, persistantInputs.size());

            persistantInputs.erase(persistantInputs.begin() + randomIndex);
            persistantOutputs.erase(persistantOutputs.begin() + randomIndex);
        }

        //now train the Q network

        Qfunction.trainingInputs = persistantInputs;
        Qfunction.trainingOutputs = persistantOutputs;

        std::cout << "Adjusting policy..." << std::endl;
        Qfunction.stochasticGradientDescent(0.1, timeSteps * epochs, learningRate);

        stateActions.clear();
        rewards.clear();


        std::cout << "Iteration " << iter << " Score = " << runGame() << std::endl; 
    }
}

void NumberCatchingAI::watchGame(){
    resetGame();
    policyFunction.loadNetwork("paramsOld");
    std::string input;
    while(turnNumber < turnLimit){

        int bestAction = getBestAction();
        performAction(bestAction);
        
        printGame();
        std::cout << "Action = " << bestAction << std::endl;
        std::cout << "Prob = " << probability(encodeState(), bestAction) << std::endl;
        std::cout << "Value = " << getValue(encodeState()) << std::endl;
        std::cin.get();
    }
}



int main(){

    NumberCatchingAI n = NumberCatchingAI();
    
    //n.runGameHuman();
    //n.watchGame();

    
    //n.trainAIQ(1e3, 1e3, 10, 1e-4);
    n.trainAIPPO(1e3, 1e3, 10, 1e-4);


    NeuralNetwork net = NeuralNetwork(1,64,64,1);

    std::vector<std::vector<double>> inputs = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> outputs = std::vector<std::vector<double>>();

    for(double x = 0; x < 10; x += 0.01){
        inputs.push_back(std::vector<double>(1, x / 10.0));
        outputs.push_back(std::vector<double>(1, (x * x) / 100.0));
    }

    net.trainingInputs = inputs;
    net.trainingOutputs = outputs;

    net.stochasticGradientDescent(0, inputs.size() * 1000, 1e-4);



    //compiles with: g++ -c -o cpp.o -std=c++11 NumberCatchingAI.cpp
    //then with    : gcc -c -o c.o genann.c
    //finally with: g++ -o prog c.o cpp.o
    //n.trainAI(10000);

}