#include "AIStructs.h"

float relu(const float x) {
	if (x < 0) {
		return exp(x) - 1;
	}
	else {
		return x;
	}
}

float drelu(const float x) {
	if (x < 0) {
		return exp(x);
	}
	else {
		return 1;
	}
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> softmax(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> in) {
	Eigen::Matrix<float, Action::MAX_ACTION, 1> out;
	float sum = 0;
	float max = in.maxCoeff();
	auto normed = in.array() - max;
	for (int i = 0; i < Action::MAX_ACTION; ++i) {
		sum += exp(normed(i));
	}
	for (int i = 0; i < Action::MAX_ACTION; ++i) {
		out(i) = exp(normed(i)) / sum;
	}
	return out;
}

//template<int rows, int cols>
Action argMax(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> zout) {
		float rc = zout(0);
		int indexMax = 0;
		for (int i = 1; i < Action::MAX_ACTION; ++i) {
			if (rc < zout(i)) {
				rc = zout(i);
				indexMax = i;
			}
		}
		return static_cast<Action>(indexMax);
}

void saveModel(const Model &m, std::string name) {
	std::stringstream ss;
	cereal::BinaryOutputArchive ar{ss};
	std::cout << "Cerealising..." << std::endl;
	ar(cereal::make_nvp("brain", m));
	std::ofstream out(name, std::ios_base::binary);
	auto serial{ ss.str() };
	std::cout << "Writing..." << std::endl;
	out.write(serial.c_str(), serial.length());
}

bool loadModel(Model &m, std::string name) {
	std::ifstream in(name, std::ios_base::binary);
	if (in.is_open()) {
		cereal::BinaryInputArchive ar{ in };
		ar(m);
		return true;
	}
	return false;
}
