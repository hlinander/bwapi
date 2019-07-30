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
#include <cereal/types/array.hpp>
#include <cereal/types/list.hpp>
#include "EigenCereal.h"
#include <Eigen/Dense>
#include <torch/torch.h>

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

// using State = Eigen::Matrix<float, N, 1>;
template<size_t N>
using State = std::array<float, N>;
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
struct Net : torch::nn::Module {
	Net() {
		fc1 = register_module("fc1", torch::nn::Linear(StateSize, N_HIDDEN));
		fc2 = register_module("fc2", torch::nn::Linear(N_HIDDEN, TAction::MAX));
		torch::nn::init::xavier_normal_(fc1->weight);
		torch::nn::init::xavier_normal_(fc2->weight);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(fc1->forward(x.reshape({1, StateSize})));
		x = fc2->forward(x);
		return x;
	}

	torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

template<typename TAction, size_t StateSize>
struct Model {
	typedef State<StateSize> StateType;
	typedef Net<TAction, StateSize> NetType;
	Model(float lr) : net{std::make_shared<NetType>()}, optimizer(net->parameters(), lr)
	{
	}

	friend std::ostream & operator<<(std::ostream &os, const Model &m) {
		os << "Model took " << m.actions.size() << " actions " << std::endl;
		m.net->pretty_print(os);
		std::cout << m.net->parameters() << std::endl;
		return os;
	}

	template <class Archive>
	void save(Archive &a) const
	{
		a(cereal::make_nvp("actions", actions));
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("probs", probs));
		std::stringstream ss;
		torch::save(net, ss);
		a(cereal::make_nvp("net", ss.str()));
		std::stringstream sso;
		torch::save(optimizer, sso);
		a(cereal::make_nvp("opt", sso.str()));
	}

	template <class Archive>
	void load(Archive &a)
	{
		a(cereal::make_nvp("actions", actions));
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("probs", probs));
		std::string s;
		a(s);
		std::stringstream ss{s};
		torch::load(net, ss);		
		std::string so;
		a(so);
		std::stringstream sso{so};
		torch::load(optimizer, sso);		
	}

	torch::Tensor forward(StateType &s) {
		auto ts = torch::from_blob(static_cast<void*>(s.data()), {1, StateSize}, torch::kFloat32);
		return net->forward(ts);
	}

	TAction get_action(StateType &s) {
		torch::Tensor out = torch::softmax(forward(s), -1);
		auto out_a = out.accessor<float,2>();
		float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		float v = 0;
		TAction action;
		for (; action != TAction::MAX; ++action) {
			v += out_a[0][static_cast<int>(action)];
			if (r <= v || action == TAction::MAX - 1) {
				probs.push_back(out_a[0][static_cast<int>(action)]);
				break;
			}
		}
		actions.push_back(action);
		states.push_back(s);
		return action;
	}

	TAction saved_action(int frame) {
		return actions[frame];
	}

	int get_frames() {
		return actions.size();
	}

	// Net<TAction, StateSize> net;
	std::shared_ptr<NetType> net;
	torch::optim::Adam optimizer;
	std::vector<TAction> actions;
	std::vector<StateType> states;
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
	float avg_ureward;
	float avg_breward;

	BrainHerder(float lr) : umodel(lr), bmodel(lr), winner{false}, avg_ureward{0}, avg_breward{0} {}

	void save(const std::string& path) {
		std::stringstream ss;
		cereal::BinaryOutputArchive ar{ss};
		ar(cereal::make_nvp("winner", winner));
		ar(cereal::make_nvp("umodel", umodel));
		ar(cereal::make_nvp("bmodel", bmodel));
		ar(cereal::make_nvp("avg_ureward", avg_ureward));
		ar(cereal::make_nvp("avg_breward", avg_breward));
		std::ofstream out(path, std::ios_base::binary);
		auto serial{ ss.str() };
		out.write(serial.c_str(), serial.length());
	}

	bool load(const std::string& path) {
		std::ifstream in(path, std::ios_base::binary);
		if (in.is_open()) {
			cereal::BinaryInputArchive ar{ in };
			ar(winner, umodel, bmodel, avg_ureward, avg_breward);
			return true;
		}
		return false;
	}
};

#endif // __AI_STRUCTS_H_DEF__

