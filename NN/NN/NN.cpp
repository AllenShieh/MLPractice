#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#define LAYERCOUNT 3
#define ROWCOUNT 8
#define ETA 1

class InputNode {
	vector<double> weight;
	double value;

public:
	InputNode(int row) {
		for (int i = 0; i < row; i++) {
			weight.push_back(0);
		}
	}
};

class OutputNode {
	double diff;
	double value;
	double real;
	double bias;

public:
	OutputNode() { bias = 0; }
	double getValue() { return value; }
	double getDiff() { return diff; }
	double calculateDiff() { diff = value*(1 - value)*(value - real); }
};

class HiddenNode {
	vector<double> weight;
	double diff;
	double value;
	double bias;

public:
	HiddenNode(int row) {
		for (int i = 0; i < row; i++) {
			weight.push_back(0);
		}
		bias = 0;
	}

	void setDiff(double v) { diff = v; }
	double getWeight(int i) { return weight[i]; }
	double getDiff() { return diff; }
	double getValue() { return value; }
};

class NN {
	vector<vector<HiddenNode>> hidden;
	vector<InputNode> input;
	OutputNode output;
	int maxLayer;
	int maxRow;
	int maxInput;

public:
	NN(int inputCount, int layerCount, int rowCount);
	void calculateDiff();
	void updateWeigth();
};

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

NN::NN(int inputCount, int layerCount, int rowCount) {
	maxInput = inputCount;
	maxLayer = layerCount;
	maxRow = rowCount;
	for (int i = 0; i < inputCount; i++) {
		input.push_back(InputNode(rowCount));
	}
	vector<HiddenNode> temp;
	for (int i = 0; i < layerCount; i++) {
		temp.push_back(HiddenNode(rowCount));
	}
	for (int i = 0; i < rowCount; i++) {
		hidden.push_back(temp);
	}
}

void NN::calculateDiff() {
	for (int l = maxLayer - 1; l >= 0; l--) {
		if (l == maxLayer - 1) {
			for (int r = 0; r < maxRow; r++) {
				double v = hidden[r][l].getWeight(0)*output.getDiff();
				hidden[r][l].setDiff(v);
			}
		}
		else {
			for (int r = 0; r < maxRow; r++) {
				double v = 0;
				for (int i = 0; i < maxRow; i++) {
					v += hidden[r][l].getWeight(i)*hidden[i][l + 1].getDiff();
				}
				hidden[r][l].setDiff(v*hidden[r][l].getValue()*(1 - hidden[r][l].getValue()));
			}
		}
	}
}

void NN::updateWeigth() {

}