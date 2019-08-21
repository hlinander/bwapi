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

#define N_UNITS 200

enum UnitTypes {
	SCV,
	MARINE,
	N_UNIT_TYPES
};

struct UnitParam {
	float hp;
	float attacked;
	float repaired;
	float moving;
	float attacking;
	float gathering;
	float repairing;
	float constructing;
	std::array<float, N_UNIT_TYPES> unit_types;

	template<class Archive>
	void serialize(Archive &a) {
		a(hp, attacked, repaired, moving, attacking, gathering, repairing, constructing, unit_types);
	}
};

#define STATE_OFF(name) ((((size_t)&((StateParam *)nullptr)->name)) / sizeof(float))
struct StateParam {
	StateParam() {
		memset(this, 0, sizeof(*this));
	}

	float& operator [](size_t n) {
		return reinterpret_cast<float*>(this)[n];
	}
	template<class Archive>
	void serialize(Archive &a) {
		a(minerals, gas, supply, n_barracks, n_supply_depots, n_marines, me, friendly, enemy);
	}

	float *data() { return reinterpret_cast<float*>(this); }
	static constexpr size_t count() { return sizeof(StateParam) / sizeof(float); }
	float minerals;
	float gas;
	float supply;
	float n_barracks;
	float n_supply_depots;
	float n_marines;
	UnitParam me;
	std::array<UnitParam, N_UNITS> friendly;
	std::array<UnitParam, N_UNITS> enemy;
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
		default:
			break;
	}
	return "BA::Unknown";
}

template<typename ActionEnum>
struct Action {
	typedef ActionEnum Type;
	static constexpr size_t max() { return static_cast<size_t>(ActionEnum::MAX); }
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

float relu(const float x);
float drelu(const float x);

template<typename TAction>
struct Net : torch::nn::Module {
	Net() : device(get_device()) {
		bn = register_module("bn", torch::nn::BatchNorm(StateParam::count()));
		fc1 = register_module("fc1", torch::nn::Linear(StateParam::count(), N_HIDDEN));
		fc2 = register_module("fc2", torch::nn::Linear(N_HIDDEN, N_HIDDEN));
		fc3 = register_module("fc3", torch::nn::Linear(N_HIDDEN, TAction::max()));
		torch::nn::init::xavier_normal_(fc1->weight);
		torch::nn::init::xavier_normal_(fc2->weight);
		torch::nn::init::xavier_normal_(fc3->weight);
		torch::nn::init::zeros_(fc3->bias);
		torch::nn::init::zeros_(fc2->bias);
		torch::nn::init::zeros_(fc1->bias);
		// fc1->to(device);
		// fc2->to(device);
		// fc3->to(device);
		to(device);
	}

	static torch::Device get_device() {
		if(getenv("NO_CUDA")) {
			return torch::kCPU;
		}
		if (torch::cuda::is_available()) {
			return torch::kCUDA;
		}
		else {
			return torch::kCPU;
		}
	}

	torch::Tensor forward(torch::Tensor x) {
		// std::cout << "x: " << torch::mean(x) << std::endl;
		// std::cout << "stdx: " << torch::std(x) << std::endl;
		// x = bn->forward(x);
		// std::cout << "bn: " << torch::mean(x) << std::endl;
		// std::cout << "bnstdx: " << torch::std(x) << std::endl;
		x = torch::leaky_relu(fc1->forward(x));
		x = torch::leaky_relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}

	torch::nn::BatchNorm bn{nullptr};
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	torch::Device device;
};

template<typename TAction>
struct Model {
	typedef Net<TAction> NetType;
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
		a(cereal::make_nvp("oh_actions", one_hot_actions));
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("immidiate_rewards", immidiate_rewards));
		a(cereal::make_nvp("probs", probs));
		a(cereal::make_nvp("time_stamps", time_stamps));
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
		a(cereal::make_nvp("oh_actions", one_hot_actions));
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("immidiate_rewards", immidiate_rewards));
		a(cereal::make_nvp("probs", probs));
		a(cereal::make_nvp("time_stamps", time_stamps));
		a(cereal::make_nvp("avg_rewards", avg_rewards));
		std::string s;
		a(s);
		std::stringstream ss{s};
		torch::load(net, ss, net->device);		
		std::string so;
		a(so);
		std::stringstream sso{so};
		torch::load(optimizer, sso, net->device);		
	}

	torch::Tensor forward(StateParam &s) {
		auto ts = torch::from_blob(static_cast<void*>(s.data()), {1, StateParam::count()}, torch::kFloat32);
		// std::cout << "To device!" << std::endl << std::flush;
		auto dts = ts.to(net->device);
		// std::cout << "Forward!" << std::endl << std::flush;
		return net->forward(dts);
		// return torch::ones({1, TAction::max()});
	}

	torch::Tensor forward_batch_nice(size_t first, size_t last) {
		return net->forward(get_batch(first, last));
	}

	torch::Tensor get_batch(size_t first, size_t last) {
		auto ret = torch::from_blob(static_cast<void *>(states.data() + first), {static_cast<long>(last - first), StateParam::count()}, torch::kFloat32);
		auto dret = ret.to(net->device);
		return dret;
	}

	torch::Tensor forward_batch_nice(torch::Tensor t) {
		return net->forward(t);
	}

	TAction get_action(StateParam &s) {
		auto tout = forward(s);
		torch::Tensor out = torch::softmax(tout, -1).to(torch::kCPU);
		// std::cout << "On CPU!" << std::endl << std::flush;
		auto out_a = out.accessor<float,2>();
		float eps = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		float v = 0;
		//TAction action{static_cast<typename TAction::Type>(torch::argmax(out[0]).item<uint32_t>())};
		TAction action;
		TAction sampled_action;
		TAction max_action;
		float max = 0.0f;
		for (; action != TAction::max(); ++action) {
			float p = out_a[0][static_cast<int>(action)];
			if(p > max) {
				max = p;
				max_action = action;	
			}
		}
		for (; sampled_action != TAction::max(); ++sampled_action) {
			v += out_a[0][static_cast<int>(sampled_action)];
			if (r <= v || sampled_action == TAction::max() - 1) {
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

	void record_action(StateParam &s, TAction &a, float immidiate_reward, float seconds) {
		states.push_back(s);
		actions.push_back(a);
		one_hot_actions.emplace_back(std::array<float, TAction::max()>{});
		one_hot_actions.back()[a] = 1.0;
		immidiate_rewards.push_back(immidiate_reward);
		time_stamps.push_back(seconds);
	}

	TAction saved_action(int frame) {
		return actions[frame];
	}

	int get_frames() {
		return actions.size();
	}

	// Net<TAction, StateParam::count()> net;
	std::shared_ptr<NetType> net;
	torch::optim::Adam optimizer;
	std::vector<TAction> actions;
	std::vector<std::array<float, TAction::max()>> one_hot_actions;
	std::vector<StateParam> states;
	std::vector<uint32_t> time_stamps;
	std::vector<float> immidiate_rewards;
	std::vector<float> probs;
	std::vector<float> avg_rewards;
};

using UnitAction = Action<UA>;
using BuildAction = Action<BA>;

using UnitModel = Model<UnitAction>;
using BuildModel = Model<BuildAction>;

struct BrainHerder {
	UnitModel umodel;
	BuildModel bmodel;
	int winner;
	float avg_ureward;
	float avg_breward;

	BrainHerder(float lr) : umodel(lr), bmodel(lr), winner{0}, avg_ureward{0}, avg_breward{0} {}

	void save(const std::string& path) {
		std::stringstream ss;
		cereal::BinaryOutputArchive ar{ss};
		ar(cereal::make_nvp("winner", winner));
		DEBUG("before models\n");
		ar(cereal::make_nvp("umodel", umodel));
		ar(cereal::make_nvp("bmodel", bmodel));
		DEBUG("after models\n");
		ar(cereal::make_nvp("avg_ureward", avg_ureward));
		ar(cereal::make_nvp("avg_breward", avg_breward));
		std::ofstream out(path, std::ios_base::binary);
		DEBUG("opened file\n");
		auto serial{ ss.str() };
		DEBUG("serialized file\n");
		out.write(serial.c_str(), serial.length());
		DEBUG("wrote file file\n");
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

struct CumBenchmark {
	struct Report {
		Report()
		: time_total{0}
		, count{0} {}
		std::chrono::duration<double> time_total;
		size_t count;
	};

	void report(const std::string &name, const std::chrono::duration<double> dur) {
		auto &b = benchmarks[name];
		b.time_total += dur;
		++b.count;
	}

	void print() {
		for(auto& b: benchmarks) {
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(b.second.time_total).count();
			std::cout << "[CBench] " << b.first << ": " << static_cast<float>(ms) / b.second.count 
					  << " (" << ms << " over " << b.second.count << ")" << std::endl;
		}
	}
	std::unordered_map<std::string, Report> benchmarks;
};

struct Benchmark {
	Benchmark(CumBenchmark &cb, std::string &&n)
		: start{std::chrono::high_resolution_clock::now()}
		, name(std::move(n))
		, cum(&cb) 
		, stopped(false)
	{}
	Benchmark(std::string &&n)
		: start{std::chrono::high_resolution_clock::now()}
		, name{std::move(n)}
		, cum(nullptr)
		, stopped(false)
	 {
		//  std::cout << "[Bench] " << name << std::endl;
	 }
	~Benchmark() {
		stop();
	}

	void stop() {
		if(stopped) {
			return;
		}
		stopped = true;
		auto dur = std::chrono::high_resolution_clock::now() - start;
		if(!cum) {
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
			std::cout << "[Bench] " << name << " took " << ms << " ms" << std::endl;
		}
		else {
			cum->report(name, dur);
		}
	}
	std::chrono::time_point<std::chrono::system_clock> start;
	std::string name;
	CumBenchmark *cum;
	bool stopped;
};


#endif // __AI_STRUCTS_H_DEF__

