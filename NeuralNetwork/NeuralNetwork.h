#include <vector>
#include <iostream>
#include "NodeLayer.h"
#include "Data.h"
#include <random>
#include <cmath>
#include <fstream>
#pragma once
class NeuralNetwork {
	public:
	NeuralNetwork(int inputs, int outputs, std::vector<int> layers);
	NeuralNetwork(std::string fileName);

	~NeuralNetwork();

	std::vector<double> runNetwork(std::vector<double> inputs);

	std::vector<double> runNetwork(double* inputs);

	

	void trainNetwork(double targetLoss, int maxIterations, int numOfSteps, double numPassesScalar, double stepSize, double randMin, double randMax, bool displayStats);



	void gradientDescent(double targetLoss, int maxIterations, double learningRate);

	std::vector<double> getLossGradient(int trainingIndex);


	double getLossDerivative(int variableIndex, int trainingIndex);


	void optimizeRandomVariable(int numOfSteps, double stepSize, double randMin, double randMax);

	void randomizeVariables(double min, double max);

	void setTrainingInputs(Data* inputs);
	void setTrainingOutputs(Data* outputs);
	double calculateCurrentLoss();

	Data* getTrainingInputs();
	Data* getTrainingOutputs();

	void debugLayers();
	void debugLayer(int layerNum);

	void saveWeights();
	void saveBiases();

	void saveNetwork(std::string filename);
	void loadNetwork(std::string filename);

	std::vector<int> dataIndexForStrongNodeSignal(int layerIndex, int nodeIndex, double threshold);

	

	private:
	std::vector<NodeLayer>* layers;

	Data* trainingInputs;

	Data* trainingOutputs;

	std::vector<double> getAllBiases();
	std::vector<double> getAllWeights();

	void loadBiases(std::string filePath);
	void loadWeights(std::string filePath);

	double getBias(int index);
	double getWeight(int index);

	void setBias(int index, double value);
	void setWeight(int index, double value);

	int numWeights;

	int numBiases;

	int numInputs;
	int numOutputs;

	double calculateCurrentLoss(int dataIndex);
};

