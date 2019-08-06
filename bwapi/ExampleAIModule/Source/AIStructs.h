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

#define CASEPRINT(x) case x: return #x

// #define DEBUG(...) printf(__VA_ARGS__)
#define DEBUG(...) {}

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

static const char* param_to_string(BuildParam b) {
	switch(b) {
		CASEPRINT(MINERALS);
		CASEPRINT(GAS);
		CASEPRINT(N_SCVS);
		CASEPRINT(N_MARINES);
		CASEPRINT(N_SUPPLY_DEPOTS);
		CASEPRINT(N_BARRACKS);
	}
	return "BA::Unknown";
}

const std::map<Param, const char*> ParamS = {
};

static const char* param_to_string(Param p) {
	switch(p) {
		CASEPRINT(Param::TEAM_HP);
		CASEPRINT(Param::TEAM_COUNT);
		CASEPRINT(Param::TEAM_DISTANCE);
		CASEPRINT(Param::TEAM_MINERALS);
		CASEPRINT(Param::ENEMY_DISTANCE);
		CASEPRINT(Param::ENEMY_COUNT);
		CASEPRINT(Param::ENEMY_HP);
		CASEPRINT(Param::ME_HP);
		CASEPRINT(Param::ME_ATTACKED);
		CASEPRINT(Param::ME_REPAIRED);
		CASEPRINT(Param::ME_SCV);
		CASEPRINT(Param::ME_MARINE);
	}
	return "BA::Unknown";
}

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


const int N_HIDDEN = 64;

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
		bn = register_module("bn", torch::nn::BatchNorm(StateSize));
		fc1 = register_module("fc1", torch::nn::Linear(StateSize, N_HIDDEN));
		fc2 = register_module("fc2", torch::nn::Linear(N_HIDDEN, N_HIDDEN));
		fc3 = register_module("fc3", torch::nn::Linear(N_HIDDEN, TAction::MAX));
		torch::nn::init::xavier_normal_(fc1->weight);
		torch::nn::init::xavier_normal_(fc2->weight);
		torch::nn::init::xavier_normal_(fc3->weight);
		torch::nn::init::zeros_(fc3->bias);
		torch::nn::init::zeros_(fc2->bias);
		torch::nn::init::zeros_(fc1->bias);
	}

	torch::Tensor forward(torch::Tensor x) {
		// std::cout << "x: " << torch::mean(x) << std::endl;
		// std::cout << "stdx: " << torch::std(x) << std::endl;
		x = bn->forward(x);
		// std::cout << "bn: " << torch::mean(x) << std::endl;
		// std::cout << "bnstdx: " << torch::std(x) << std::endl;
		x = torch::leaky_relu(fc1->forward(x));
		x = torch::leaky_relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}

	torch::nn::BatchNorm bn{nullptr};
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

template<typename TAction, size_t StateSize, size_t BatchSize>
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

	void init_grad() {

	}

	template <class Archive>
	void save(Archive &a) const
	{
		a(cereal::make_nvp("actions", actions));
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("immidiate_rewards", immidiate_rewards));
		a(cereal::make_nvp("probs", probs));
		a(cereal::make_nvp("avg_rewards", avg_rewards));
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
		a(cereal::make_nvp("immidiate_rewards", immidiate_rewards));
		a(cereal::make_nvp("probs", probs));
		a(cereal::make_nvp("avg_rewards", avg_rewards));
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

	torch::Tensor forward_batch(std::vector<StateType> &s) {
		// std::vector<float> data;
		// data.reserve(s.size() * StateSize);
		for(int i = 0; i < BatchSize; ++i) {
			std::copy(s[i].begin(), s[i].end(), &batch_data[i * StateSize]);
		}
		// for(int i = 0; i < BatchSize; ++i) {
		// for(int j = 0; j < StateSize; ++j) {
		// 	std::cout << batch_data[i * StateSize + j] << "|";
		// }
		// std::cout << std::endl;
		// }
		auto ts = torch::from_blob(static_cast<void*>(batch_data.data()), 
								   {static_cast<long>(BatchSize), StateSize}, torch::kFloat32);
		return net->forward(ts);
	}

	TAction get_action(StateType &s) {
		torch::Tensor out = torch::softmax(forward(s), -1);
		auto out_a = out.accessor<float,2>();
		float eps = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		float v = 0;
		//TAction action{static_cast<typename TAction::Type>(torch::argmax(out[0]).item<uint32_t>())};
		TAction action;
		TAction sampled_action;
		TAction max_action;
		float max = 0.0f;
		for (; action != TAction::MAX; ++action) {
			float p = out_a[0][static_cast<int>(action)];
			if(p > max) {
				max = p;
				max_action = action;	
			}
		}
		for (; sampled_action != TAction::MAX; ++sampled_action) {
			v += out_a[0][static_cast<int>(sampled_action)];
			if (r <= v || sampled_action == TAction::MAX - 1) {
				//probs.push_back(out_a[0][static_cast<int>(sampled_action)]);
				break;
			}
		}
		if(eps < 1.0) {
			action = sampled_action;
		}
		else {
			action = max_action;
		}
		// actions.push_back(action);
		// states.push_back(s);
		return action;
	}

	void record_action(StateType &s, TAction &a, float immidiate_reward) {
		states.push_back(s);
		actions.push_back(a);
		immidiate_rewards.push_back(immidiate_reward);
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
	std::array<float, BatchSize * StateSize> batch_data;
	std::vector<TAction> actions;
	std::vector<StateType> states;
	std::vector<float> immidiate_rewards;
	std::vector<float> probs;
	std::vector<float> avg_rewards;
};

using UnitAction = Action<UA>;
using BuildAction = Action<BA>;

template<size_t BatchSize>
using UnitModel = Model<UnitAction, static_cast<size_t>(Param::MAX_PARAM), BatchSize>;
template<size_t BatchSize>
using BuildModel = Model<BuildAction, static_cast<size_t>(BuildParam::MAX_BUILD), BatchSize>;

template<size_t BatchSize>
struct BrainHerder {
	UnitModel<BatchSize> umodel;
	BuildModel<BatchSize> bmodel;
	int winner;
	float avg_ureward;
	float avg_breward;

	BrainHerder(float lr) : umodel(lr), bmodel(lr), winner{0}, avg_ureward{0}, avg_breward{0} {}

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

