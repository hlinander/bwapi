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
	Eigen::Matrix<float, Eigen::Dynamic, 1> out(in.rows());
	float sum = 0;
	float max = in.maxCoeff();
	auto normed = in.array() - max;
	for (int i = 0; i < in.rows(); ++i) {
		sum += exp(normed(i));
	}
	for (int i = 0; i < in.rows(); ++i) {
		out(i) = exp(normed(i)) / sum;
	}
	return out;
}

