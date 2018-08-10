#include "Data.h"


Data::Data(int rows, int columns) {
	numRows = rows;
	numCols = columns;

	array = new double*[rows];

	for (int i = 0; i < rows; i++) {
		array[i] = new double[columns];
	}


}


Data::Data(std::string filePath, std::string rowSeparator, std::string columnSeparator) {

	std::string fileContents = "";
	std::string singleLine;
	std::ifstream file(filePath);

	std::vector<std::string> rows = std::vector<std::string>();

	numRows = 0;
	if (file.is_open()) {

		while (std::getline(file, singleLine)) {
			rows.push_back(singleLine);
			numRows++;
		}


	} else {
		//file could not be opened
	}

	
	std::vector<std::string> singleRow;
	std::vector<double> singleRowDoubles;
	numRows = rows.size();
	numCols = NNHelper::split(rows[0], columnSeparator).size();

	array = new double*[numRows];

	for (int i = 0; i < numRows; i++) {
		array[i] = new double[numCols];
	}

	for (int r = 0; r < numRows; r++) {
		singleRow = NNHelper::split(rows[r], columnSeparator);
		singleRowDoubles = NNHelper::stringToDoubleVector(singleRow);
		for (int c = 0; c < numCols; c++) {
			array[r][c] = singleRowDoubles[c];
		}
	}


	

}

Data::~Data() {
	delete array;
}

double* Data::getRow(int rowIndex) {
	if (rowIndex >= numRows) {
		return nullptr;
	}

	

	double* r = new double[numCols];


	for (int i = 0; i < numCols; i++) {
		r[i] = array[rowIndex][i];
	}

	return r;
}


double Data::getIndex(int row, int column) {
	if (row >= numRows) {
		//exception here
	}

	if (column >= numCols) {
		//exception here
	}
	
	
	return array[row][column];
}


void Data::setIndex(int row, int column, double value) {
	if (row >= numRows) {
		//exception here
	}

	if (column >= numCols) {
		//exception here
	}

	array[row][column] = value;
}


int Data::getNumRows() {
	return numRows;
}


int Data::getNumCols() {
	return numCols;
}
