#include "NNHelper.h"



NNHelper::NNHelper() {
}


NNHelper::~NNHelper() {
}

double NNHelper::dotProduct(double a[], double b[], int length) {

	double result = 0;

	bool multithreaded = true;
	int threadCount = 6;


		for (int i = 0; i < length; i++) {
			result += a[i] * b[i];
		}


	return result;

	
}



double NNHelper::RELUFunction(double input, double bias) {
	double newInput = input + bias;

	if (newInput < 0) {
		return newInput * .01;
	} else {
		return newInput;
	}
}



double NNHelper::calculateLoss(double value1, double value2) {
	return (value1 - value2) * (value1 - value2);
}

std::vector<double> NNHelper::arrayToVector(double array[], int arraySize) {
	std::vector<double> r = std::vector<double>();
	
	double num;
	for (int i = 0; i < arraySize; i++) {
		num = array[i];
		r.push_back(array[i]);
	}

	return r;
}

double NNHelper::randomDouble(double min, double max) {
	double scalar = (double)rand() / RAND_MAX;

	return min + (scalar * (max - min));
}

int NNHelper::randomInt(int min, int max) {
	
	return min + (rand() % (max - min));
}



std::vector<int> NNHelper::generateConfig(int numNodes, int minNodeCount) {
	std::vector<int> r = std::vector<int>();
	
	int nodesLeft = numNodes;
	int rand;

	while(nodesLeft >= minNodeCount) {
		if (nodesLeft == 1) {
			r.push_back(1);
			return r;
		}
		

		rand = NNHelper::randomInt(minNodeCount, nodesLeft + 1);

		nodesLeft -= rand;
		
		r.push_back(rand);

	}

	while (minNodeCount > 0) {
		r[NNHelper::randomInt(0,r.size())]++;
		minNodeCount--;
	}


	return r;
}

bool NNHelper::contains(std::vector<int> v, std::vector<std::vector<int>> setOfVectors) {


	for (int i = 0; i < setOfVectors.size(); i++) {
		if (setOfVectors[i] == v) {
			return true;
		}
	}

	return false;
}

std::vector<std::string> NNHelper::split(std::string s, std::string splitter) {
	std::vector<std::string> r = std::vector<std::string>();

	std::string copy = s;
	
	while (NNHelper::contains(copy, splitter)) {
		r.push_back(copy.substr(0, copy.find_first_of(splitter)));
		copy = copy.substr(copy.find_first_of(splitter) + splitter.length());
	}

	r.push_back(copy);


	return r;
	
}

std::vector<double> NNHelper::stringToDoubleVector(std::vector<std::string> v) {
	std::vector<double> r = std::vector<double>();

	for (int i = 0; i < v.size(); i++) {
		r.push_back(std::stod(v[i]));
	}

	return r;
}

bool NNHelper::contains(std::string s, std::string targetString) {
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

std::vector<int> NNHelper::randomOrder(int size) {
	std::vector<int> result = std::vector<int>();
	
	for (int i = 0; i < size; i++) {
		result.push_back(i);
	}

	//shuffle elements around

	int rand1;
	int rand2;

	int placeholder;

	for (int i = 0; i < size * 2; i++) {
		rand1 = randomInt(0, size);
		rand2 = randomInt(0, size);

		//swap elements

		placeholder = result[rand1];
		result[rand1] = result[rand2];
		result[rand2] = placeholder;
	}

	return result;
}
