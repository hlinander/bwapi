#ifndef __AI_STRUCTS_H_DEF__
#define __AI_STRUCTS_H_DEF__

#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/list.hpp>
#include "EigenCereal.h"
#include <Eigen/Dense>

enum Param {
	TEAM_HP = 0,
	TEAM_COUNT,
	TEAM_DISTANCE,
	TEAM_MINERALS,
	ENEMY_DISTANCE,
	ENEMY_COUNT,
	ENEMY_HP,
	ME_HP,
	ME_ATTACKED,
	ME_REPAIRED,
	MAX_PARAM
};

const std::map<Param, const char*> ParamS = {
	{ Param::TEAM_HP, "TEAM_HP"},
	{ Param::TEAM_COUNT, "TEAM_COUNT" },
	{ Param::TEAM_DISTANCE, "TEAM_DISTANCE" },
	{ Param::TEAM_MINERALS, "TEAM_MINERALS" },
	{ Param::ENEMY_DISTANCE, "ENEMY_DISTANCE" },
	{ Param::ENEMY_COUNT, "ENEMY_COUNT" },
	{ Param::ENEMY_HP, "ENEMY_HP" },
	{ Param::ME_HP, "ME_HP" },
	{ Param::ME_ATTACKED, "ME_ATTACKED" },
	{ Param::ME_REPAIRED, "ME_REPAIRED" }
};

enum Action {
	ATTACK = 0,
	REPAIR,
	FLEE,
	MAX_ACTION
};

const std::map<Action, const char*> ActionS = {
	{Action::ATTACK, "ATTACK"},
	{Action::REPAIR, "REPAIR"},
	{Action::FLEE, "FLEE"}
};

const int N_HIDDEN = 10;

struct State {
	State() {
		edata.setZero();
		for (int i = 0; i < Param::MAX_PARAM; ++i) {
			data[i] = 0.0;
		}
	}

	friend std::ostream & operator<<(std::ostream &os, const State &m) {
		os << "[State] ";
		for (int i = 0; i < Param::MAX_PARAM; ++i) {
			std::cout << ParamS.at(static_cast<Param>(i)) << ": " << m.edata[i] << ", ";
		}
		os << std::endl;
		return os;
	}

	template <class Archive>
	void serialize(Archive &a)
	{
		a(cereal::make_nvp("edata", edata));
	}

	float data[Param::MAX_PARAM];
	Eigen::Matrix<float, Param::MAX_PARAM, 1> edata;
};

//template<int rows, int cols>
//Eigen::Matrix<float, rows, cols> grads(Dense &layer) {
//	Eigen::Matrix<float, rows, cols> ret;
//
//}

float relu(const float x);
float drelu(const float x);

//template<int rows, int cols>
Action argMax(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> zout);
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> softmax(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> in);

struct Model {
	Model() :
		winner(false)
	{
		params.resize(Action::MAX_ACTION);
		for (auto &ref : params) {
			ref.resize(Param::MAX_PARAM);
		}
		hidden.setRandom();
		out.setRandom();
	}

	friend std::ostream & operator<<(std::ostream &os, const Model &m) {
		os << "Model is " << (m.winner ? "WINNER" : "looooser") << std::endl;
		os << "Model took " << m.actions.size() << " actions " << std::endl;
		os << "Hidden:" << std::endl << m.hidden << std::endl;
		os << "Output:" << std::endl << m.out << std::endl;
		return os;
	}

	template <class Archive>
	void serialize(Archive &a)
	{
		a(cereal::make_nvp("params", params));
		a(cereal::make_nvp("hidden", hidden));
		a(cereal::make_nvp("out", out));
		a(cereal::make_nvp("dhiddens", dhiddens));
		a(cereal::make_nvp("douts", douts));
		a(cereal::make_nvp("actions", actions));
		a(cereal::make_nvp("winner", winner));
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("probs", probs));
	}

	Eigen::Matrix<float, Action::MAX_ACTION, 1> forward(const State &s) {
		//std::cout << "State: " << s.edata << std::endl;
		//std::cout << 
		zhidden = hidden * s.edata;
		ahidden = zhidden.unaryExpr(&relu);
		return out * ahidden;
	}

	std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> grads(const State &s, const Action a) {
		// DlogpDout = DlogpDzout * DzoutDout = dlogp * ahidden
		//Eigen::Matrix<float, N_HIDDEN, Param::MAX_PARAM> dhidden;
		Eigen::Matrix<float, Action::MAX_ACTION, N_HIDDEN> dout;

		dout.setZero();
		dout.row(a) = ahidden;
		auto dzhidden = zhidden.unaryExpr(&drelu);
		auto dhidden = out.row(a).transpose().cwiseProduct(dzhidden)*s.edata.transpose();
		dhiddens.push_back(dhidden);
		douts.push_back(dout);
		actions.push_back(a);
		states.push_back(s);
		return { dhidden, dout };
	}

	std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> saved_grads(int frame) {
		return { dhiddens[frame], douts[frame] };
	}

	Action saved_action(int frame) {
		return actions[frame];
	}

	int get_frames() {
		return actions.size();
	}

	void descent(std::vector < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> grads, float lr) {
		hidden += lr * grads[0];
		out += lr * grads[1];
	}

	std::vector< std::vector<float> > params;
	Eigen::Matrix<float, N_HIDDEN, Param::MAX_PARAM> hidden;
	Eigen::Matrix<float, Action::MAX_ACTION, N_HIDDEN> out;

	Eigen::Matrix<float, N_HIDDEN, 1> ahidden;
	Eigen::Matrix<float, N_HIDDEN, 1> zhidden;

	std::vector<Eigen::Matrix<float, N_HIDDEN, Param::MAX_PARAM>> dhiddens;
	std::vector<Eigen::Matrix<float, Action::MAX_ACTION, N_HIDDEN>> douts;
	std::vector<Action> actions;
	std::vector<State> states;
	std::vector<float> probs;
	bool winner;

};

void saveModel(const Model &m, std::string name);

bool loadModel(Model &m, std::string name);

#endif // __AI_STRUCTS_H_DEF__

