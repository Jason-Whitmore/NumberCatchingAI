#pragma once
#include "NNHelper.h"
class NodeLayer {
	public:
	NodeLayer(int numInputs, int numNodes);
	~NodeLayer();

	void dotProductAndApplyBias();
	void dotProduct();

	void setInputArray(double a[]);
	void setInputArray(std::vector<double> v);

	double* getOutputArray();
	std::vector<double> getOutputVector();

	int getNumBiases();
	int getNumWeights();

	void setBias(int index, double bias);
	double getBias(int index);

	void setWeight(int index, double weight);
	double getWeight(int index);

	private:

	double* inputArray;

	double** inputWeights;

	double* layerOutputs;

	double* outputBiases;

	int numInputs;

	int numOutputs;

};

