#include <vector>
#include <stdio.h>
#include <iostream>
#include <random>
#include <fstream>


enum ActivationFunction{Tanh, Sigmoid, RELU, LeakyRELU, Linear};

struct Connection;

struct Node{
    double value;
    std::vector<Connection*> inputs = std::vector<Connection*>();
    std::vector<Connection*> outputs = std::vector<Connection*>();
    ActivationFunction function;
    int id;
    double inputSum;
};

struct Connection{
    Node* start;
    Node* end;
    double weight;
    int id;
    double loss;
    bool isBias;
};


class NeuralNetwork {
    public:
    NeuralNetwork();
    NeuralNetwork(int,int,int,int);
    std::vector<std::vector<Node*>> nodes;

    std::vector<std::vector<double>> trainingInputs;

    std::vector<std::vector<double>> trainingOutputs;

    std::vector<Connection*> connections;

    int numWeights;
    int numNodes;
    int currentNode;

    Node* biasNode;

    double randomDouble(double,double);
    std::vector<double> compute(std::vector<double>);

    static double getNodeOutput(Node*);

    double calculateAverageLoss();
    double calculateLoss(int);
    std::vector<double> getGradient(int);
    std::vector<double> getGradientApprox(int);
    double getDerivative(Node*);

    double sumNodeOutputLoss(Node*);
    double getDerivative(double x, ActivationFunction f);

    void stochasticGradientDescent(double targetLoss, uint epochs, double learningRate);
    void jasonTrain(double targetLoss, uint iterations, double learningRate);
    void stochasticGradientDescentApprox(double targetLoss, uint epochs, double learningRate);
    std::vector<int> randomOrder(int);

    Node* getNode(int);
    void setActivationFunction(int, ActivationFunction);
    std::vector<double> getWeights();
    void saveNetwork(std::string);
    void loadNetwork(std::string);
    bool contains(std::string, std::string);
    std::vector<std::string> split(std::string, std::string);
    void randomizeNetwork(double min, double max);

};