#include "NeuralNetwork.h"



NeuralNetwork::NeuralNetwork(){

}

NeuralNetwork::NeuralNetwork(int numInputs, int layer1, int layer2, int numOutputs){
    srand(time(NULL));
    numWeights = 0;
    numNodes = 0;
    //create layers
    connections = std::vector<Connection*>();
    std::vector<Node*> currentLayer = std::vector<Node*>();
    Node* n = new Node;
    n->function = ActivationFunction::Tanh;
    n->value = 1;
    n->id = numNodes;
    currentNode++;
    //create first layer
    for(int i = 0; i < numInputs; i++){
        n = new Node;
        n->function = ActivationFunction::Linear;
        n->value = 1;
        n->id = numNodes;
        currentLayer.push_back(n);
        numNodes++;
    }
    //add layer to network
    nodes.push_back(currentLayer);
    currentLayer.clear();

    //create second layer
    for(int i = 0; i < layer1; i++){
        n = new Node;
        n->value = 0;
        n->function = ActivationFunction::LeakyRELU;
        n->id = numNodes;
        currentLayer.push_back(n);
        numNodes++;
    }

    nodes.push_back(currentLayer);
    currentLayer.clear();

    //third
    for(int i = 0; i < layer2; i++){
        n = new Node;
        n->value = 0;
        n->function = ActivationFunction::LeakyRELU;
        n->id = numNodes;
        currentLayer.push_back(n);
        numNodes++;
    }

    nodes.push_back(currentLayer);
    currentLayer.clear();

    //output
    for(int i = 0; i < numOutputs; i++){
        n = new Node;

        n->value = 0;
        n->function = ActivationFunction::Sigmoid;
        n->id = numNodes;
        currentLayer.push_back(n);
        numNodes++;
    }
    nodes.push_back(currentLayer);
    currentLayer.clear();

    //now connect the layers
    Connection* c;
    int currentID = 0;
    //loop through layers
    for(int i = 1; i < nodes.size(); i++){
        //destination nodes
        for(int d = 0; d < nodes[i].size(); d++){
            //start nodes
            for(int s = 0; s < nodes[i - 1].size(); s++){
                numWeights++;
                c = new Connection();
                c->weight = randomDouble(-1,1);
                c->start = nodes[i-1][s];
                c->end = nodes[i][d];
                c->id = currentID;
                currentID++;
                c->start->outputs.push_back(c);
                c->end->inputs.push_back(c);
                connections.push_back(c);
            }
        }
    }
    //now create the bias node which connects to all nodes except for the first layer
    biasNode = new Node;
    biasNode->value = 1;
    biasNode->function = ActivationFunction::Linear;

    for(int layer = 1; layer < nodes.size(); layer++){
        for(int n = 0; n < nodes[layer].size(); n++){
            numWeights++;
            c = new Connection();
            c->isBias = true;
            c->weight = randomDouble(-1,1);
            c->start = biasNode;
            c->end = nodes[layer][n];
            c->id = currentID;
            currentID++;
            c->start->outputs.push_back(c);
            c->end->inputs.push_back(c);
            connections.push_back(c);
        }
    }

    std::default_random_engine gen;
    std::normal_distribution<double> distribution(0,1);

    for(int i = 0; i < connections.size(); i++){
        connections[i]->weight = randomDouble(-1,1);
    }
}

std::vector<double> NeuralNetwork::compute(std::vector<double> inputs){
    std::vector<double> outputs = std::vector<double>();

    for(int i = 0; i < nodes[0].size(); i++){
        nodes[0][i]->value = inputs[i];
    }

    //loop each layer
    for(int layer = 1; layer < nodes.size(); layer++){
        //loop each node in layer
        for(int n = 0; n < nodes[layer].size(); n++){
            nodes[layer][n]->value = getNodeOutput(nodes[layer][n]);
        }
    }

    for(int i = 0; i < nodes[nodes.size() - 1].size(); i++){
        outputs.push_back(getNodeOutput(nodes[nodes.size() - 1][i]));
    }
    return outputs;
}

double NeuralNetwork::getNodeOutput(Node* n){
    if(n->inputs.size() == 0){
        return n->value;
    }
    double dotProduct = 0;

    for(int i = 0; i < n->inputs.size(); i++){
        dotProduct += n->inputs[i]->weight * n->inputs[i]->start->value;
    }
    
    n->inputSum = dotProduct;

    //activation function
    ActivationFunction function = n->function;
    if(function == ActivationFunction::Tanh){
        return tanh(dotProduct);
    } else if(function == ActivationFunction::RELU){
        if(dotProduct > 0){
            return dotProduct;
        } else {
            return 0;
        }
    } else if (function == ActivationFunction::LeakyRELU){
        if(dotProduct > 0){
            return dotProduct;
        } else {
            return dotProduct * 0.1;
        }
    } else if (function == ActivationFunction::Sigmoid){
        return 1.0 / (1 + std::exp(-dotProduct));
    } else if(function == ActivationFunction::Linear){
        return dotProduct;
    }
    //just in case
    return 0;
}

double NeuralNetwork::randomDouble(double min, double max){
    double scalar = ((double)rand()) / RAND_MAX;
    return min + (scalar * (max - min));
}


double NeuralNetwork::calculateAverageLoss(){
    double result = 0;

    //loop through the training data
    for(int dp = 0; dp < trainingInputs.size(); dp++){
        result += calculateLoss(dp);
    }
    return result / trainingInputs.size();
}

double NeuralNetwork::calculateLoss(int sampleIndex){
    double result = 0;
    std::vector<double> input = trainingInputs[sampleIndex];
    std::vector<double> output = trainingOutputs[sampleIndex];
    std::vector<double> actualOutput = compute(input);

    for(int i = 0; i < output.size(); i++){
        result += std::pow(std::abs(actualOutput[i] - output[i]),2);
    }
    return result / output.size();
}

std::vector<double> NeuralNetwork::getGradient(int sampleIndex){

    const double delta = 1e-6;
    std::vector<double> grad = std::vector<double>(numWeights);
    //first, run the data sample through the network
    std::vector<double> output = compute(trainingInputs[sampleIndex]);

    double error;
    Connection* con;
    //set the initial error on the final weights
    for(int n = 0; n < nodes[nodes.size() - 1].size(); n++){
        //loop through the node input connections
        for(int c = 0; c < nodes[nodes.size() - 1][n]->inputs.size(); c++){
            con = nodes[nodes.size() - 1][n]->inputs[c];

            double lossBefore = calculateLoss(sampleIndex);
            con->weight += delta;
            double lossAfter = calculateLoss(sampleIndex);
            con->weight -= delta;

            con->loss = (lossAfter - lossBefore) / delta;

            if(con->loss > 1000){
                    con->loss = 1000;
            }
            if(con->loss < -1000){
                    con->loss = -1000;
            }

            grad[con->id] = con->loss;
            
        }
    }


    //now that the weights have been computed, work backwards to compute the gradient wrt the loss
    for(int layer = nodes.size() - 2; layer > 0; layer--){
        for(int n = 0; n < nodes[layer].size(); n++){
            for(int in = 0; in < nodes[layer][n]->inputs.size(); in++){
                //connection is either a bias or a weight
                if(nodes[layer][n]->inputs[in]->isBias){
                    //bias
                } else {
                    //weight
                }
                con = nodes[layer][n]->inputs[in];

                //double lossWRToutput = con->start->value * getDerivative(nodes[layer][n]) * sumNodeOutputLoss(nodes[layer][n]);
                
                //con->loss = con->start->value * 
                
                con->loss = con->start->value * getDerivative(nodes[layer][n]) * sumNodeOutputLoss(nodes[layer][n]);
                if(con->loss > 1000){
                    con->loss = 1000;
                }
                if(con->loss < -1000){
                    con->loss = -1000;
                }
                grad[con->id] = con->loss;
            }

        }
    }

    //finally, return the gradient
    return grad;
}


std::vector<double> NeuralNetwork::getGradientApprox(int sampleNumber){
    const double delta = 1e-6;
    std::vector<double> grad = std::vector<double>();

    for(int w = 0; w < connections.size(); w++){
        double oldLoss = calculateLoss(sampleNumber);
        connections[w]->weight += delta;
        double newLoss = calculateLoss(sampleNumber);
        connections[w]->weight -= delta;

        grad.push_back((newLoss - oldLoss) / delta);
    }

    return grad;
}

double NeuralNetwork::sumNodeOutputLoss(Node* node){
    double r = 0;
    for(int i = 0; i < node->outputs.size(); i++){
        r += node->outputs[i]->loss;
    }
    return r;

}

double NeuralNetwork::getDerivative(Node* node){
    return getDerivative(node->inputSum, node->function);
}

double NeuralNetwork::getDerivative(double x, ActivationFunction f){
    if(f == ActivationFunction::LeakyRELU){
        if(x >= 0){
            return x;
        }
        return 0.1 * x;
    } else if (f == ActivationFunction::RELU){
        if(x >= 0){
            return x;
        }
        return 0;
    } else if(f == ActivationFunction::Sigmoid){
        return (1.0 / (1 + std::exp(-x))) * (1 - (1.0 / (1 + std::exp(-x))));
    } else if(f == ActivationFunction::Tanh){
        return 1 - std::pow(std::tanh(x),2);
    }


    //just to make the compiler happy...
    std::cout << "Warning. Returning bad derivative." << std::endl;
    return 0;
}

void NeuralNetwork::stochasticGradientDescent(double targetLoss, uint epochs, double learningRate){
    std::vector<double> gradient;
    std::vector<int> ordering;

    const double lambda = 0.04;
    

    double effectiveLearningRate = learningRate;
    uint currentSample = 0;
    for(int iter = 0; iter < epochs; iter++){
        if(iter % trainingInputs.size() == 0){
            ordering = randomOrder(trainingInputs.size());
            currentSample = 0;
        }

        gradient = getGradient(ordering[currentSample]);
        //apply learning rate to gradient
        Connection* c;
        for(int i = 0; i < gradient.size(); i++){
            c = connections[i];
            c->weight = c->weight - (learningRate * gradient[i]) - learningRate * lambda * c->weight;
        }


        if(iter % trainingInputs.size() == 0 && calculateAverageLoss() < targetLoss){
            return;
        }
        if(iter % trainingInputs.size() == 0){
            std::cout << "Loss = " << calculateAverageLoss() << std::endl;
            //system("clear");
        }
        currentSample++;
    }
}

void NeuralNetwork::jasonTrain(double targetLoss, uint iterations, double learningRate){
    std::vector<double> gradient;
    std::vector<int> ordering;

    const double lambda = 0.04;
    const double epsilon = 0.001;

    double effectiveLearningRate = learningRate;

    double bestLoss = calculateAverageLoss();
    std::vector<double> bestParams = getWeights();
    uint currentSample = 0;
    for(uint iter = 0; iter < iterations; iter++){
        if(iter % trainingInputs.size() == 0){
            ordering = randomOrder(trainingInputs.size());
            currentSample = 0;
        }

        gradient = getGradient(ordering[currentSample]);

    
        //apply learning rate to gradient
        Connection* c;
        
        //keep training
        for(int i = 0; i < gradient.size(); i++){
            c = connections[i];
            c->weight = c->weight - (learningRate * gradient[i]) - learningRate * lambda * c->weight;
        }   
        
        


        if(iter % trainingInputs.size() == 0 && calculateAverageLoss() < targetLoss){
            return;
        }


        if(iter % trainingInputs.size() == 0){
            double currentLoss = calculateAverageLoss();
            if(currentLoss < bestLoss){
                bestLoss = currentLoss;
                bestParams = getWeights();
            }
            std::cout << "Loss = " << currentLoss << std::endl;

            double avgSlope = gradientAvgAbsValue(getGradient());
            //"kick" out of a local minimum
            if(avgSlope < epsilon){
                //std::cout << "Kicking out of minimum" << std::endl;
                Connection* c;
                for(int i = 0; i < gradient.size(); i++){
                    c = connections[i];
                    c->weight = c->weight + randomDouble(-5,5);
                }
            }
            
        }
        currentSample++;
    }
    //std::cout << "Best loss = " << bestLoss << std::endl;
    for(int w = 0; w < connections.size(); w++){
        connections[w]->weight = bestParams[w];
    }
}

void NeuralNetwork::stochasticGradientDescentApprox(double targetLoss, uint epochs, double learningRate){
    const double delta = 1e-4;
    int currentEpochs = 0;
    std::vector<double> outputs;
    std::vector<double> actualOutputs;
    std::vector<double> inputs;
    std::vector<int> order;
    int currentSample = 0;
    std::vector<double> error = std::vector<double>();
    std::vector<double> gradient = std::vector<double>(numWeights);

    int currentWeight = numWeights;
    Connection* c;
    double oldLoss;
    double newLoss;

    double avgLoss;

    double lr = learningRate;

    while(currentEpochs < epochs){

        if(currentEpochs % trainingInputs.size() == 0){
            //shuffle the data samples
                order = randomOrder(trainingInputs.size());
                currentSample = 0;
        }


        //loop through weights, using derivative approximations
        for(int w = 0; w < numWeights; w++){
            oldLoss = calculateLoss(order[currentSample]);
            c = connections[w];
            c->weight += delta;
            newLoss = calculateLoss(order[currentSample]);
            c->weight -= delta;
            gradient[w] = (newLoss - oldLoss) / delta;
            if(std::isnan( gradient[w] ) || std::isinf(gradient[w])){
                gradient[w] = 0;
            }

            if(gradient[w] > 1000){
                gradient[w] = 1000;
            }
            if(gradient[w] < -1000){
                gradient[w] = -1000;
            }
        }

        currentSample++;
        //gradient is now calculated. Apply with learning rate

        for(int w = 0; w < numWeights; w++){
            c = connections[w];
            c->weight -= lr * gradient[w];
        }
        
        if(currentEpochs % 10000 == 0 && calculateAverageLoss() < targetLoss){
            break;
        }

        if(currentEpochs % 10000 == 0){
            std::cout << "Avg loss = " << calculateAverageLoss() << std::endl;
        }
        
        currentEpochs++;

    }

    

}

Node* NeuralNetwork::getNode(int ID){
    for(int layer = 0; layer < nodes.size(); layer++){
        for(int node = 0; node < nodes[layer].size(); node++){
            if(nodes[layer][node]->id == ID){
                return nodes[layer][node];
            }
        }
    }
    return nullptr;
}

void NeuralNetwork::setActivationFunction(int ID, ActivationFunction f){
    Node* n = getNode(ID);
    n->function = f;
}


std::vector<double> NeuralNetwork::getWeights(){
    std::vector<double> r = std::vector<double>();

    for(int w = 0; w < connections.size(); w++){
        r.push_back(connections[w]->weight);
    }

    return r;
}

std::vector<int> NeuralNetwork::randomOrder(int size){
    std::vector<int> r = std::vector<int>();
    for(int i = 0; i < size; i++){
        r.push_back(i);
    }

    int temp;
    int index1;
    int index2;
    for(int i = 0; i < size * 2; i++){
        //find two elements
        index1 = (int)randomDouble(0, size);
        index2 = (int)randomDouble(0, size);
        
        //swap
        temp = r[index1];
        r[index1] = r[index2];
        r[index2] = temp;
    }

    return r;
}

void NeuralNetwork::saveNetwork(std::string fileName){
    std::string contents = "";

    for(uint w = 0; w < connections.size(); w++){
        contents += std::to_string(w) + " " + std::to_string(connections[w]->weight) + "\n";
    }
    //file ready to be saved
    std::ofstream file;
    file.open(fileName);
    file << contents;
    file.close();
}

bool NeuralNetwork::contains(std::string s, std::string targetString) {
    int targetLength = targetString.length();

    int currentPosition = 0;

    while (currentPosition + targetLength <= s.length()) {
        if (s.substr(currentPosition,targetLength) == targetString) {
            return true;
        }
        currentPosition++;
    }

    return false;
}

std::vector<std::string> NeuralNetwork::split(std::string s, std::string splitter) {
    std::vector<std::string> r = std::vector<std::string>();

    std::string copy = s;
    
    while (contains(copy, splitter)) {
        r.push_back(copy.substr(0, copy.find_first_of(splitter)));
        copy = copy.substr(copy.find_first_of(splitter) + splitter.length());
    }

    r.push_back(copy);


    return r;
    
}

void NeuralNetwork::loadNetwork(std::string fileName){
    std::ifstream file(fileName);

    std::string singleLine;
    std::vector<std::string> lineSeparated;

    if(file.is_open()){
        while(std::getline(file, singleLine)){
            lineSeparated = split(singleLine, " ");

            connections[std::stoi(lineSeparated[0])]->weight = std::stod(lineSeparated[1]);
        }
    }

    file.close();
}

void NeuralNetwork::randomizeNetwork(double min, double max){
    for(int i = 0; i < connections.size(); i++){
        connections[i]->weight = randomDouble(min,max);
    }
}


double NeuralNetwork::gradientAvgAbsValue(std::vector<double> gradient){
    double sum = 0;

    for(int i = 0; i < gradient.size(); i++){
        sum += std::abs(gradient[i]);

    }

    return sum / gradient.size();
}

std::vector<double> NeuralNetwork::getGradient(){
    std::vector<double> r = std::vector<double>(connections.size());

    int numSamples = std::fmax(10, trainingInputs.size());

    for(int n = 0; n < numSamples; n++){
        int s = rand() % trainingInputs.size();
        std::vector<double> grad = getGradient(s);

        for(int i = 0; i < r.size(); i++){
            r[i] += grad[i];
        }
        
    }

    for(int i = 0; i < r.size(); i++){
        r[i] /= numSamples;
    }

    return r;
}


