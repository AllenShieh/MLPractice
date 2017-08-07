#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <fstream>

using namespace std;

vector<vector<string>> featureList;
vector<vector<string>> samples;
vector<string> classes;
#define ob samples[0].size()-1

class DecisionTreeNode {
	int currentKey;
	string keyName;
	vector<DecisionTreeNode> sons;
	vector<int> featureID;
	vector<int> sampleID;
	bool isLeaf;
	string className;

public:
	DecisionTreeNode(string n) : keyName(n), isLeaf(false) {}
	DecisionTreeNode(vector<int> f, vector<int> s) : featureID(f), sampleID(s), isLeaf(false) {}
	int SelectKey();
	void PickSons(int key);
	void ConstructNode();
	void SetFeatureID(vector<int> f) { featureID = f; }
	bool CheckLeaf();
	string PredictClass(vector<string> e);
	
};

double ComputeEntropy(vector<int> ids) {
	vector<int> counter;
	for (int i = 0; i < classes.size(); i++) counter.push_back(0);
	for (int i = 0; i < ids.size(); i++) {
		for (int j = 0; j < classes.size(); j++) {
			if (classes[j] == samples[ids[i]][ob]) {
				counter[j]++;
				break;
			}
		}
	}
	double res = 0;
	for (int i = 0; i < classes.size(); i++) {
		if (counter[i] != 0) {
			double ratio = double(counter[i]) / double(ids.size());
			res += (-ratio*log2(ratio));
		}
	}
	return res;
}

double ComputeInfoKey(int key, vector<int> ids) {
	vector<vector<int>> counter;
	vector<int> temp;
	for (int i = 0; i < featureList[key].size(); i++) counter.push_back(temp);
	for (int i = 0; i < ids.size(); i++) {
		for (int j = 1; j < featureList[key].size(); j++) {
			if (samples[ids[i]][key] == featureList[key][j]) {
				counter[j].push_back(ids[i]);
				break;
			}
		}
	}
	double res = 0;
	for (int i = 1; i < featureList[key].size(); i++) {
		res += double(counter[i].size()) / double(ids.size())*ComputeEntropy(counter[i]);
	}
	return res;
}

double ComputeInfo(int key, vector<int> ids) {
	return ComputeEntropy(ids) - ComputeInfoKey(key, ids);
}

int DecisionTreeNode::SelectKey() {
	double m = 0;
	int key = 0;
	for (int i = 0; i < featureID.size(); i++) {
		double t = ComputeInfo(featureID[i], sampleID);
		if (t > m) {
			m = t;
			key = i;
		}
	}
	return key;
}

bool DecisionTreeNode::CheckLeaf() {
	string c = samples[sampleID[0]][ob];
	for (int i = 1; i < sampleID.size(); i++) {
		if (c != samples[sampleID[i]][ob]) {
			isLeaf = false;
			return false;
		}
	}
	className = c;
	isLeaf = true;
	return true;
}

void DecisionTreeNode::PickSons(int key){
	for (int i = 0; i < sampleID.size(); i++) {
		if (sons.size() == 0) {
			DecisionTreeNode t(samples[sampleID[i]][key]);
			sons.push_back(t);
		}
		else {
			for (int j = 0; j < sons.size(); j++) {
				if (sons[j].keyName == samples[sampleID[i]][key]) break;
				if (j == sons.size() - 1) {
					DecisionTreeNode t(samples[sampleID[i]][key]);
					sons.push_back(t);
				}
			}
		}
	}
}

void DecisionTreeNode::ConstructNode() {
	if (CheckLeaf()) return;
	int p = SelectKey();
	int key = featureID[p];
	currentKey = key;
	featureID.erase(featureID.begin() + p);
	PickSons(key);
	for (int i = 0; i < sampleID.size(); i++) {
		for (int j = 0; j < sons.size(); j++) {
			sons[j].SetFeatureID(featureID);
			if (samples[sampleID[i]][key] == sons[j].keyName) {
				sons[j].sampleID.push_back(sampleID[i]);
			}
		}
	}
	for (int i = 0; i < sons.size(); i++) {
		sons[i].ConstructNode();
	}
}

string DecisionTreeNode::PredictClass(vector<string> e) {
	if (isLeaf == true) return className;
	for (int i = 0; i < sons.size(); i++) {
		if (sons[i].keyName == e[currentKey]) {
			return sons[i].PredictClass(e);
		}
	}
}

void DataProcess(string f) {
	ifstream fin(f);
	string s;
	while (fin >> s) {
		if (s != "start") {
			vector<string> t;
			t.push_back(s);
			featureList.push_back(t);
		}
		else break;
	}
	featureList.pop_back();
	while (fin >> s) {
		vector<string> t;
		t.push_back(s);
		for (int i = 1; i <= featureList.size(); i++) {
			fin >> s;
			t.push_back(s);
		}
		samples.push_back(t);
	}
	for (int i = 0; i < samples.size(); i++) {
		for (int j = 0; j < featureList.size(); j++) {
			if (featureList[j].size() == 1) {
				featureList[j].push_back(samples[i][j]);
			}
			else {
				for (int k = 1; k < featureList[j].size(); k++) {
					if (featureList[j][k] == samples[i][j]) break;
					if (k == featureList[j].size() - 1) featureList[j].push_back(samples[i][j]);
				}
			}
		}
	}
}

int main() {
	DataProcess("data.txt");
	vector<int> featureID;
	for (int i = 0; i < featureList.size(); i++) featureID.push_back(i);
	vector<int> sampleID;
	for (int i = 0; i < samples.size(); i++) sampleID.push_back(i);
	DecisionTreeNode root(featureID, sampleID);
	root.ConstructNode();
	vector<string> test;
	test.push_back("Rainy");
	test.push_back("Cool");
	test.push_back("Normal");
	test.push_back("Strong");
	string res = root.PredictClass(test);
	cout << res << endl;
	return 0;
}