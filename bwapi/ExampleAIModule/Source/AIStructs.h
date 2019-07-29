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
	ME_SCV,
	ME_MARINE,
	MAX_PARAM
};

enum BuildParam {
	MINERALS,
	GAS,
	N_SCVS,
	N_MARINES,
	N_SUPPLY_DEPOTS,
	N_BARRACKS,
	MAX_BUILD
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
	{ Param::ME_REPAIRED, "ME_REPAIRED" },
	{ Param::ME_SCV, "ME_SCV" },
	{ Param::ME_MARINE, "ME_MARINE" }
};

enum class BA {
	BUILD_SCV = 0,
	BUILD_SUPPLY,
	BUILD_BARRACK,
	BUILD_MARINE,
	IDLE,
	MAX
};

enum class UA {
	ATTACK = 0,
	REPAIR,
	FLEE,
	MINE,
	MAX
};

static const char* action_to_string(UA a) {
	static const std::map<UA, const char*> UnitActionS = {
		{UA::ATTACK, "ATTACK"},
		{UA::REPAIR, "REPAIR"},
		{UA::FLEE, "FLEE"},
		{UA::MINE, "MINE"}
	};
	return UnitActionS.at(a);
}

#define STRACTION(x) case x: return #x

static const char* action_to_string(BA a) {
	switch(a) {
		STRACTION(BA::BUILD_SCV);
		STRACTION(BA::BUILD_BARRACK);
		STRACTION(BA::BUILD_MARINE);
		STRACTION(BA::BUILD_SUPPLY);
		STRACTION(BA::IDLE);
	}
	return "BA::Unknown";
}

template<typename ActionEnum>
struct Action {
	typedef ActionEnum Type;
	static constexpr const size_t MAX = static_cast<size_t>(ActionEnum::MAX);
	Action(ActionEnum a = static_cast<ActionEnum>(0)) : action{a} {}
	operator int() const {
		return static_cast<int>(action);
	}
	Action<ActionEnum>& operator =(const int rhs) {
		action = static_cast<Action>(rhs);
		return *this;
	}

	bool operator ==(const ActionEnum rhs) {
		return action == rhs;
	}

	Action<ActionEnum>& operator++() {
		action = static_cast<ActionEnum>(static_cast<int>(action) + 1);
		return *this;
	}

	template <class Archive>
	void serialize(Archive &a)
	{
		a(action);
	}

	const char* name() const {
		return action_to_string(action);
	}

	ActionEnum action;
};


const int N_HIDDEN = 10;

template<size_t N>
using State = Eigen::Matrix<float, N, 1>;
using UnitState = State<static_cast<size_t>(Param::MAX_PARAM)>;
using BuildState = State<static_cast<size_t>(BuildParam::MAX_BUILD)>;
//template<int rows, int cols>
//Eigen::Matrix<float, rows, cols> grads(Dense &layer) {
//	Eigen::Matrix<float, rows, cols> ret;
//
//}

float relu(const float x);
float drelu(const float x);

//template<int rows, int cols>
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> softmax(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> in);

template<typename TAction, size_t StateSize>
struct Model {
	typedef State<StateSize> StateType;
	Model()
	{
		params.resize(TAction::MAX);
		for (auto &ref : params) {
			ref.resize(StateSize);
		}
		hidden.setRandom();
		out.setRandom();
	}

	friend std::ostream & operator<<(std::ostream &os, const Model &m) {
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
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("probs", probs));
	}

	Eigen::Matrix<float, TAction::MAX, 1> forward(const StateType &s) {
		//std::cout << "State: " << s.edata << std::endl;
		//std::cout << 
		zhidden = hidden * s;
		ahidden = zhidden.unaryExpr(&relu);
		return out * ahidden;
	}

	TAction get_action(const StateType &s) {
		auto logp{forward(s)};
		auto distribution = softmax(logp);
		float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		float v = 0;
		TAction action;
		for (; action != TAction::MAX; ++action) {
			v += distribution(static_cast<int>(action));
			if (r <= v || action == TAction::MAX - 1) {
				probs.push_back(distribution(static_cast<int>(action)));
				break;
			}
		}
		grads(s, action);
		return action;
	}

	std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> grads(const StateType &s, const TAction a) {
		// DlogpDout = DlogpDzout * DzoutDout = dlogp * ahidden
		//Eigen::Matrix<float, N_HIDDEN, Param::MAX> dhidden;
		Eigen::Matrix<float, TAction::MAX, N_HIDDEN> dout;

		dout.setZero();
		dout.row(a) = ahidden;
		auto dzhidden = zhidden.unaryExpr(&drelu);
		auto dhidden = out.row(a).transpose().cwiseProduct(dzhidden)*s.transpose();
		dhiddens.push_back(dhidden);
		douts.push_back(dout);
		actions.push_back(a);
		states.push_back(s);
		return { dhidden, dout };
	}

	std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> saved_grads(int frame) {
		return { dhiddens[frame], douts[frame] };
	}

	TAction saved_action(int frame) {
		return actions[frame];
	}

	int get_frames() {
		return actions.size();
	}

	void descent(const std::vector < Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> &grads, float lr) {
		hidden += lr * grads[0];
		out += lr * grads[1];
		float alr = fabs(lr);
		hidden -= alr*hidden;
		out -= alr*out;
	}


	std::vector< std::vector<float> > params;
	Eigen::Matrix<float, N_HIDDEN, StateSize> hidden;
	Eigen::Matrix<float, TAction::MAX, N_HIDDEN> out;

	Eigen::Matrix<float, N_HIDDEN, 1> ahidden;
	Eigen::Matrix<float, N_HIDDEN, 1> zhidden;

	std::vector<Eigen::Matrix<float, N_HIDDEN, StateSize>> dhiddens;
	std::vector<Eigen::Matrix<float, TAction::MAX, N_HIDDEN>> douts;
	std::vector<TAction> actions;
	std::vector<State<StateSize>> states;
	std::vector<float> probs;
};

using UnitAction = Action<UA>;
using BuildAction = Action<BA>;

using UnitModel = Model<UnitAction, static_cast<size_t>(Param::MAX_PARAM)>;
using BuildModel = Model<BuildAction, static_cast<size_t>(BuildParam::MAX_BUILD)>;

struct BrainHerder {
	UnitModel umodel;
	BuildModel bmodel;
	bool winner;

	void save(const std::string& path) {
		std::stringstream ss;
		cereal::BinaryOutputArchive ar{ss};
		ar(cereal::make_nvp("winner", winner));
		ar(cereal::make_nvp("umodel", umodel));
		ar(cereal::make_nvp("bmodel", bmodel));
		std::ofstream out(path, std::ios_base::binary);
		auto serial{ ss.str() };
		out.write(serial.c_str(), serial.length());
	}

	bool load(const std::string& path) {
		std::ifstream in(path, std::ios_base::binary);
		if (in.is_open()) {
			cereal::BinaryInputArchive ar{ in };
			ar(winner, umodel, bmodel);
			return true;
		}
		return false;
	}
};

#endif // __AI_STRUCTS_H_DEF__

