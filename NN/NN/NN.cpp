#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

using namespace std;

#define LAYERCOUNT 2
#define ROWCOUNT 4
#define ETA 0.9
#define THRESHOLD 0.012

vector<vector<double>> inputs;
vector<double> targets;

class InputNode {
	vector<double> weight;
	double value;

public:
	InputNode(int row) {
		for (int i = 0; i < row; i++) {
			weight.push_back(2.0*(double)rand() / RAND_MAX - 1);
		}
	}

	void setWeight(int p, double v) { weight[p] = v; }
	double getWeight(int p) { return weight[p]; }
	double getValue() { return value; }
	void setValue(double v) { value = v; }
};

class OutputNode {
	double diff;
	double value;
	double bias;
	double target;

public:
	OutputNode() { bias = 0; }
	double getValue() { return value; }
	double getDiff() { return diff; }
	void calculateDiff() { diff = value*(1 - value)*(value - target); }
	void setValue(double v) { value = v; }
	double getBias() { return bias; }
	void setTarget(double v) { target = v; }
	double getTarget() { return target; }
};

class HiddenNode {
	vector<double> weight;
	double diff;
	double value;
	double bias;

public:
	HiddenNode(int row) {
		for (int i = 0; i < row; i++) {
			weight.push_back(2.0*(double)rand() / RAND_MAX - 1);
		}
		bias = 0;
	}

	void setDiff(double v) { diff = v; }
	double getWeight(int i) { return weight[i]; }
	double getDiff() { return diff; }
	double getValue() { return value; }
	void setWeight(int p, double v) { weight[p] = v; }
	void setValue(double v) { value = v; }
	double getBias() { return bias; }
};

class NN {
	vector<vector<HiddenNode>> hidden;
	vector<InputNode> input;
	OutputNode output;
	vector<double> loss;
	int maxLayer;
	int maxRow;
	int maxInput;
	int lossCount;

public:
	NN(int inputCount, int layerCount, int rowCount, int entryCount);
	void calculateDiff();
	void updateWeigth();
	void inputOneData(vector<double> d, double r);
	void calculateOneData(int p);
	double calculateLoss();
	void readData(string filename);
	void preProcess();
	void train();
	double predict(vector<double> x);
};

double sigmoid(double x) {
	double result = (double)1 / ((double)1 + exp(-x));
	return result;
}

double NN::calculateLoss() {
	double r = 0;
	for (int i = 0; i < loss.size(); i++) {
		r += loss[i] * loss[i];
	}
	return r / loss.size();
}

NN::NN(int inputCount, int layerCount, int rowCount, int entryCount) {
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
	for (int i = 0; i < entryCount; i++) {
		loss.push_back(0);
	}
	lossCount = entryCount;
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
	for (int i = 0; i < input.size(); i++) {
		for (int j = 0; j < maxRow; j++) {
			double v = input[i].getWeight(j) - ETA*input[i].getValue()*hidden[j][0].getDiff();
			input[i].setWeight(j, v);
		}
	}
	for (int i = 0; i < maxLayer; i++) {
		for (int j = 0; j < maxRow; j++) {
			for (int k = 0; k < maxRow; k++) {
				if (i < maxLayer - 1) {
					double v = hidden[j][i].getWeight(k) - ETA*hidden[j][i].getValue()*hidden[j][i].getDiff();
					hidden[j][i].setWeight(k, v);
				}
				else {
					double v = hidden[j][i].getWeight(k) - ETA*hidden[j][i].getValue()*output.getDiff();
					hidden[j][i].setWeight(k, v);
				}
			}
		}
	}
}

void NN::inputOneData(vector<double> d, double r) {
	for (int i = 0; i < maxInput; i++) {
		input[i].setValue(d[i]);
	}
	output.setTarget(r);
}

void NN::calculateOneData(int p) {
	for (int i = 0; i < maxInput; i++) {
		input[i].setValue(inputs[p][i]);
	}
	output.setTarget(targets[p]);
	for (int i = 0; i < maxLayer; i++) {
		for (int j = 0; j < maxRow; j++) {
			double v = 0;
			if (i == 0) {
				for (int k = 0; k < maxInput; k++) {
					v += input[k].getValue()*input[k].getWeight(j);
				}
			}
			else {
				for (int k = 0; k < maxRow; k++) {
					v += hidden[j][i - 1].getValue()*hidden[j][i - 1].getWeight(j);
				}
			}
			hidden[j][i].setValue(sigmoid(v + hidden[j][i].getBias()));
		}
	}
	double v = 0;
	for (int i = 0; i < maxRow; i++) {
		v += hidden[i][maxLayer - 1].getValue()*hidden[i][maxLayer - 1].getWeight(0);
	}
	output.setValue(sigmoid(v + output.getBias()));
	output.calculateDiff();
	loss[p] = output.getTarget() - output.getValue();
}

void NN::readData(string filename) {
	ifstream fin(filename);
	int m, n;
	fin >> m >> n;
	for (int i = 0; i < m; i++) {
		vector<double> t;
		double a;
		for (int j = 0; j < n; j++) {
			fin >> a;
			t.push_back(a);
		}
		inputs.push_back(t);
		fin >> a;
		targets.push_back(a);
	}
}

void NN::preProcess() {
	for (int i = 0; i < inputs.size(); i++) {
		calculateOneData(i);
	}
}

void NN::train() {
	double oldV, newV;
	newV = calculateLoss();
	oldV = newV + 0.00001;
	while (newV >= THRESHOLD) {
		int p = rand() % lossCount;
		int count = 0;
		while (newV < oldV && count < 50) {
			oldV = newV;
			calculateOneData(p);
			calculateDiff();
			updateWeigth();
			newV = calculateLoss();
			count++;
		}
		oldV = newV + 0.00001;
		cout << newV << endl;
	}
	/*
	for (int i = 0; i < inputs.size(); i++) {
		double v = calculateLoss();
		while (v >= THRESHOLD) {
			calculateOneData(i);
			calculateDiff();
			updateWeigth();
			v = calculateLoss();
		}
	}
	*/
}

double NN::predict(vector<double> x) {
	for (int i = 0; i < maxInput; i++) {
		input[i].setValue(x[i]);
	}
	for (int i = 0; i < maxLayer; i++) {
		for (int j = 0; j < maxRow; j++) {
			double v = 0;
			if (i == 0) {
				for (int k = 0; k < maxInput; k++) {
					v += input[k].getValue()*input[k].getWeight(j);
				}
			}
			else {
				for (int k = 0; k < maxRow; k++) {
					v += hidden[j][i - 1].getValue()*hidden[j][i - 1].getWeight(j);
				}
			}
			hidden[j][i].setValue(sigmoid(v + hidden[j][i].getBias()));
		}
	}
	double v = 0;
	for (int i = 0; i < maxRow; i++) {
		v += hidden[i][maxLayer - 1].getValue()*hidden[i][maxLayer - 1].getWeight(0);
	}
	output.setValue(sigmoid(v + output.getBias()));
	return output.getValue();
}

int main() {
	NN nn(3, 2, 4, 7);
	nn.readData("t.txt");
	nn.preProcess();
	nn.train();
	vector<double> x;
	x.push_back(1);
	x.push_back(1);
	x.push_back(1);
	double result = nn.predict(x);
	cout << result << endl;
	return 0;
}