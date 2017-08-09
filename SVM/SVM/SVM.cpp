#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

vector<vector<double>> x;
vector<double> y;

class SVM {
	vector<double> alpha;
	double beta;
	double slack;
	vector<int> alphaToAdjust;
	int alphaOne;
	int alphaTwo;
	vector<double> E;

public:
	SVM(double s) :slack(s) {
		for (int i = 0; i < x.size(); i++) alpha.push_back(0);
		for (int i = 0; i < x.size(); i++) E.push_back(G(i) - y[i]);
	}
	bool CheckAlpha();
	void PickAlpha();
	void Update();
	double G(int id);
	double Y(vector<double> e);
	void BuildSVM();
	double Predict(vector<double> e);
};

vector<double> AddVector(vector<double> a, vector<double> b) {
	vector<double> result;
	for (int i = 0; i < a.size(); i++) {
		result.push_back(a[i] + b[i]);
	}
	return result;
}

vector<double> MulVector(vector<double> a, double m) {
	vector<double> result;
	for (int i = 0; i < a.size(); i++) {
		result.push_back(m*a[i]);
	}
	return result;
}

double InnerVector(vector<double> a, vector<double> b) {
	double result = 0;
	for (int i = 0; i < a.size(); i++) {
		result += a[i] * b[i];
	}
	return result;
}

// g(x) = sigma alpha y x * x + b
double SVM::G(int id) {
	vector<double> sigma = MulVector(MulVector(x[0], y[0]), alpha[0]);
	for (int i = 1; i < x.size(); i++) {
		sigma = AddVector(sigma, MulVector(MulVector(x[i], y[i]), alpha[i]));
	}
	return InnerVector(sigma, x[id]) + beta;
}

double SVM::Y(vector<double> e) {
	vector<double> sigma = MulVector(MulVector(x[0], y[0]), alpha[0]);
	for (int i = 1; i < x.size(); i++) {
		sigma = AddVector(sigma, MulVector(MulVector(x[i], y[i]), alpha[i]));
	}
	return InnerVector(sigma, e) + beta;
}

bool SVM::CheckAlpha() {
	vector<int> newList;
	for (int i = 0; i < alphaToAdjust.size(); i++) {
		if (y[alphaToAdjust[i]] * E[alphaToAdjust[i]] > 0.0001) {
			if (alpha[alphaToAdjust[i]] > 0) newList.push_back(alphaToAdjust[i]);
		}
		else if (y[alphaToAdjust[i]] * E[alphaToAdjust[i]] < -0.0001) {
			if (alpha[alphaToAdjust[i]] < slack) newList.push_back(alphaToAdjust[i]);
		}
	}
	if (newList.size() >= 2) {
		alphaToAdjust = newList;
		return false;
	}
	newList.clear();
	for (int i = 0; i < alpha.size(); i++) {
		if (y[i] * E[i] > 0.0001) {
			if (alpha[i] > 0) newList.push_back(i);
		}
		else if (y[i] * E[i] < -0.0001) {
			if (alpha[i] < slack) newList.push_back(i);
		}
	}
	if (newList.size() <= 3) return true;
	alphaToAdjust = newList;
	return false;
}

void SVM::PickAlpha() {
	alphaOne = alphaToAdjust[rand() % alphaToAdjust.size()];
	double m = E[alphaToAdjust[0]];
	int id = alphaToAdjust[0];
	if (E[alphaOne] > 0) {
		for (int i = 1; i < alphaToAdjust.size(); i++) {
			if (E[alphaToAdjust[i]] < m && alphaToAdjust[i] != alphaOne) {
				m = E[alphaToAdjust[i]];
				id = alphaToAdjust[i];
			}
		}
	}
	else {
		for (int i = 1; i < alphaToAdjust.size(); i++) {
			if (E[alphaToAdjust[i]] > m && alphaToAdjust[i] != alphaOne) {
				m = E[alphaToAdjust[i]];
				id = alphaToAdjust[i];
			}
		}
	}
	alphaTwo = id;
}

void SVM::Update() {
	double oldOne = alpha[alphaOne];
	double oldTwo = alpha[alphaTwo];
	double K11 = InnerVector(x[alphaOne], x[alphaOne]);
	double K12 = InnerVector(x[alphaOne], x[alphaTwo]);
	double K22 = InnerVector(x[alphaTwo], x[alphaTwo]);
	double eta = K11 + K22 - 2 * K12;
	double L, H;
	if (y[alphaOne] == y[alphaTwo]) {
		L = (0 > oldTwo + oldOne - slack) ? 0 : oldTwo + oldOne - slack;
		H = (slack < oldTwo + oldOne) ? slack : oldTwo + oldOne;
	}
	else {
		L = (0 > oldTwo - oldOne) ? 0 : oldTwo - oldOne;
		H = (slack < slack + oldTwo - oldOne) ? slack : slack + oldTwo - oldOne;
	}
	alpha[alphaTwo] = oldTwo + y[alphaTwo] * (E[alphaOne] - E[alphaTwo]) / eta;
	if (alpha[alphaTwo] > H) alpha[alphaTwo] = H;
	if (alpha[alphaTwo] < L) alpha[alphaTwo] = L;
	alpha[alphaOne] = oldOne + y[alphaOne] * y[alphaTwo] * (oldTwo - alpha[alphaTwo]);
	// update b
	double b1 = -E[alphaOne] - y[alphaOne] * K11 * (alpha[alphaOne] - oldOne)
		- y[alphaTwo] * K12 * (alpha[alphaTwo] - oldTwo) + beta;
	double b2 = -E[alphaTwo] - y[alphaOne] * K12 * (alpha[alphaOne] - oldOne)
		- y[alphaTwo] * K22 * (alpha[alphaTwo] - oldTwo) + beta;
	if (alpha[alphaOne] < slack && alpha[alphaOne] > 0) {
		beta = b1;
	}
	else if (alpha[alphaTwo] < slack && alpha[alphaTwo] > 0) {
		beta = b2;
	}
	else {
		beta = (b1 + b2) / 2;
	}
	// update E
	for (int i = 0; i < x.size(); i++) E[i] = G(i) - y[i];
}

void SVM::BuildSVM() {
	while (CheckAlpha() == false) {
		PickAlpha();
		Update();
	}
}

double SVM::Predict(vector<double> e) {
	double y = Y(e);
	if (y > 0) return 1;
	else return -1;
}

void DataProcess(string name) {
	double t;
	int featureCount;
	ifstream fin(name);
	fin >> featureCount;
	while (fin >> t) {
		vector<double> e;
		e.push_back(t);
		for (int i = 1; i < featureCount; i++) {
			fin >> t;
			e.push_back(t);
		}
		x.push_back(e);
		fin >> t;
		y.push_back(t);
	}
}

int main() {
	DataProcess("test.txt");
	SVM svm(0.6);
	svm.BuildSVM();
	vector<double> t1, t2, t3, t4;
	t1.push_back(3);
	t1.push_back(2);
	t2.push_back(8);
	t2.push_back(2);
	t3.push_back(0);
	t3.push_back(5);
	t4.push_back(4);
	t4.push_back(-10);
	double result1 = svm.Predict(t1);
	double result2 = svm.Predict(t2);
	double result3 = svm.Predict(t3);
	double result4 = svm.Predict(t4);
	cout << endl;
	return 0;
}



