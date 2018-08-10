#pragma once
#include <string>
#include "NNHelper.h"
#include <fstream>
class Data {
	public:
	Data(int rows, int columns);
	Data(std::string filePath, std::string rowSeparator, std::string columnSeparator);
	~Data();

	double* getRow(int rowIndex);

	double getIndex(int row, int column);

	void setIndex(int row, int column, double value);

	int getNumRows();
	int getNumCols();
	

	private:

	int numRows;
	int numCols;

	double** array;
};

