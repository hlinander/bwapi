// Overmind.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <ctime>
#include "AIStructs.h"
#include "json.hpp"
//#include <torch/torch.h>

const float LR = 0.0001;//0.00000001;
const int BATCH_SIZE = 1000;

using stat_map = std::unordered_map<std::string, size_t>;
using state_reward_map = std::unordered_map<size_t, float>;

state_reward_map u_reward_map = {{STATE_OFF(minerals), 5.0}};
state_reward_map b_reward_map = {{STATE_OFF(minerals), 5.0}};

void torch_tests();
void test_load_save();
void test_grad();

void print_state(StateParam &state) {
	std::cout << "[";
	for(size_t i = 0; i < STATE_OFF(friendly); ++i) {
		std::cout << state[i] << ", ";
	}
	std::cout << "]" << std::endl;
}

void print_tensor_1d(torch::Tensor t) {
	auto a = t.accessor<float, 1>();
	std::cout << "[";
	for(int i = 0; i < a.size(0); ++i) {
		std::cout << a[i] << ", ";
	}
	std::cout << "]" << std::endl;
}

void generate_plot(std::string path, BrainHerder &bh) {
	std::ofstream outfile;
	outfile.open(path, std::ios::trunc);
	for(int i = 0; i < bh.umodel.avg_rewards.size(); ++i) {
		outfile << i << " " << bh.umodel.avg_rewards[i] << " " << bh.bmodel.avg_rewards[i] << std::endl;
	}
}

void write_json(std::string path, nlohmann::json &json) {
	std::ofstream outfile;
	outfile.open(path);
	outfile << json;
}

void print_stats(const stat_map &s, size_t total_frames) {
	if(total_frames) {
		for(auto it = s.begin(); s.end() != it; ++it) {
			std::cout << (it->first) << ": " << static_cast<int>(100.0 * static_cast<float>(it->second) / total_frames) << "%, ";
		}
		std::cout << std::endl;
	}
}

struct Reward {
	std::vector<float> rewards;
	float total_reward;
};

std::vector<float> normalize_rewards(std::vector<float>& rewards) {
	std::vector<float> normalized_rewards(rewards);

	double sum = std::accumulate(rewards.begin(), rewards.end(), 0.0);
	double mean = sum / rewards.size();

	double sq_sum = std::inner_product(rewards.begin(), rewards.end(), rewards.begin(), 0.0);
	double stddev = std::sqrt(sq_sum / rewards.size() - mean * mean);
	
	if(stddev > 0.0f) {
		std::transform(normalized_rewards.begin(), normalized_rewards.end(), normalized_rewards.begin(),
					[mean, stddev](float r) -> float { return (r) / stddev; });
	} 
	return normalized_rewards;
}

template<typename T>
Reward calculate_rewards(int winner, Model<T> &experience, state_reward_map &sr) {
	DEBUG("Calculating rewards\n");
	Reward ret;
	ret.rewards.resize(experience.get_frames());
	std::fill(std::begin(ret.rewards), std::end(ret.rewards), 0.0f);
	ret.total_reward = 0.0;
	float reward = winner * 2.0f;
	for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
		reward += experience.immidiate_rewards[frame];
		for(auto& reward_pair: sr) {
			float diff = experience.states[frame][reward_pair.first] - experience.states[frame - 1][reward_pair.first];
			if(diff > 0) {
				reward += reward_pair.second * diff;
			}
			// reward += reward_pair.second * (experience.states[frame][reward_pair.first]);
		}
		reward *= 0.99;
		ret.rewards[frame] = reward;
	}
	ret.total_reward = std::accumulate(ret.rewards.begin(), ret.rewards.end(), 0.0f);
	ret.rewards = normalize_rewards(ret.rewards);
	return ret;
}

template<typename T>
void debug_models(bool winner, Model<T> &prev, Model<T> &next) {
	prev.net->eval();
	prev.net->eval();
	next.net->eval();
	next.net->eval();
	stat_map ustats;
	stat_map bstats;
	Reward reward = calculate_rewards(winner, prev, u_reward_map);
	std::cout << "reward[0]: " << reward.rewards[0] << std::endl;
	for(int frame = 0; frame < prev.get_frames(); ++frame) {
		print_state(prev.states[frame]);
		auto p = torch::softmax(prev.forward(prev.states[frame]), -1).to(torch::kCPU);
		auto pnext = torch::softmax(next.forward(prev.states[frame]), -1).to(torch::kCPU);
		auto action = prev.actions[frame];
		float r = pnext[0][action].template item<float>() / p[0][action].template item<float>();
		auto rs = pnext[0] / p[0];
		print_tensor_1d(rs);
		float creward = reward.rewards[frame];
		float adv = reward.rewards[frame];// - prev.avg_ureward;
		// std::cout << "rs: " << rs << std::endl;
		// std::cout << "pnext: " << pnext << std::endl;
		std::cout << "-{" << frame << ", " << prev.time_stamps[frame] << "}- " << action.name() << " @ " << p[0][action].template item<float>() << " reward: " << creward << " adv: " << adv  << " r: " << r << std::endl;
		std::cout << prev.states[frame].minerals * 2000.0 << std::endl;

	}
}


template<typename T>
float update_model(int winner, Model<T> &m, Model<T> &experience, stat_map &stats, state_reward_map &sr, const float avg_reward, bool debug) {
	torch::Tensor loss = torch::tensor({0.0f});
	Reward reward = calculate_rewards(winner, experience, sr);

	for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
		stats[experience.actions[frame].name()]++;
	}
	int steps = 0;
	DEBUG("Update with batches...\n");
	DEBUG("Frames: %d, Batchsize: %d\n", experience.get_frames(), BATCH_SIZE);
	// for (int frame = experience.get_frames() - 1; frame >= BATCH_SIZE; frame-=BATCH_SIZE) {
	for (int frame = 0; frame < experience.get_frames(); frame+=BATCH_SIZE) {
		DEBUG("Frame %d\n", frame);
		size_t actual_bs = std::min(BATCH_SIZE, experience.get_frames() - frame);

		auto logp = m.forward_batch_nice(experience.get_batch(frame, frame + actual_bs));
		auto p = torch::softmax(logp, -1);
		auto old_p = torch::softmax(experience.forward_batch_nice(frame, frame + actual_bs), -1);

		std::array<float, BATCH_SIZE * T::max()> rewards_batch{};
		for(int i = 0; i < actual_bs; ++i) {
			rewards_batch[i * T::max() + experience.actions[frame + i]] = reward.rewards[frame + i];
		}
		auto trewards =torch::from_blob(static_cast<void*>(rewards_batch.data()), {actual_bs, T::max()}, torch::kFloat32).to(m.net->device);
		auto masked_r = (p / old_p);
		auto lloss = torch::min(masked_r * trewards, torch::clamp(masked_r, 1.0 - 0.2, 1.0 + 0.2) * trewards).sum();
		(-lloss).backward();
	}
	DEBUG("Loss backwards\n");
	DEBUG("Returning...\n");
	return reward.total_reward;
}


int main(int argc, char* argv[])
{
	if (torch::cuda::is_available()) {
		std::cout << "CUDA!" << std::endl;
	}
	else {
		std::cout << "CPU byxor!" << std::endl;
	}
	// torch::Tensor tensor = torch::rand({2, 3});
	// auto s = at::sum(tensor);
	// torch::save(s, "test.pt");
	srand((unsigned int)time(0));
	if (argc < 3) {
		std::cout << "Overmind\n"; 
		std::cout << "-create name\n"; 
		exit(0);
	}
	BrainHerder bh(LR);
	if (std::string(argv[1]) == "-create") {
		bh.save(argv[2]);
		//saveModel(m, argv[2]);
	}
	else if (std::string(argv[1]) == "-torchtest") {
		//saveModel(m, argv[2]);
		torch_tests();
	}
	else if (std::string(argv[1]) == "-batchtest") {
		//saveModel(m, argv[2]);
	}
	else if (std::string(argv[1]) == "-savetest") {
		//saveModel(m, argv[2]);
		test_load_save();
	}
	else if (std::string(argv[1]) == "-gradtest") {
		//saveModel(m, argv[2]);
		test_grad();
	}
	else if (std::string(argv[1]) == "-debug") {
		BrainHerder prev(0.0f);
		BrainHerder next(0.0f);
		prev.load(argv[2]);
		next.load(argv[3]);
		std::cout << "Unit models" << std::endl;
		debug_models(prev.winner, prev.umodel, next.umodel);
		std::cout << "Build models" << std::endl;
		debug_models(prev.winner, prev.bmodel, next.bmodel);
		// bh.umodel.optimizer.zero_grad();
		// bh.bmodel.optimizer.zero_grad();
		// std::cout << bh.umodel.net->parameters()[2].grad() << std::endl;
		// std::cout << bh.umodel << std::endl;
		// update_model(bh.winner, bh.umodel, c.umodel, ustats, u_reward_map, bh.avg_ureward, true);
		// update_model(bh.winner, bh.bmodel, c.bmodel, bstats, b_reward_map, bh.avg_breward, true);
		// bh.umodel.optimizer.step();
		// bh.bmodel.optimizer.step();
	}
	else if (std::string(argv[1]) == "-show") {
		std::cout << "Showing " << argv[2] << std::endl;
		// loadModel(m, argv[2]);
		bh.load(argv[2]);
		std::cout << (bh.winner == 1 ? "WINNER" : "LOSER") << std::endl;
		std::cout << bh.umodel << std::endl;
	}
	else if (std::string(argv[1]) == "-update") {
		if (argc < 5) {
			std::cout << "-update model result_list_file model_out" << std::endl;
			exit(0);
		}
		nlohmann::json json;
		Benchmark update{"update"};
		std::vector<BrainHerder> cs;
		{
			Benchmark load{"load"};
			bh.load(argv[2]);
			std::ifstream infile(argv[3]);
			std::string line;
			while (std::getline(infile, line)) {
				cs.emplace_back(0.0f);
				cs.back().load(line);
			}
		}
		int winners = 0;
		int total_uframes = 0;
		int total_bframes = 0;
		int total = 0;
		stat_map ustats;
		stat_map bstats;
		float total_ureward = 0;
		float total_breward = 0;
		float high_reward = -10000;
		std::string high_game;
		std::cout << (bh.umodel.net->is_training() ? "TRAINING" : "EVALUATING") << std::endl;
		std::cout << "LR: " << bh.umodel.optimizer.options.learning_rate() << std::endl;
		for(int epoch = 0; epoch < 5; epoch++) {
			Benchmark bepoch{"epoch"};
			bh.umodel.optimizer.zero_grad();
			bh.bmodel.optimizer.zero_grad();
			for(auto &c: cs) {
				total++;
				if(c.winner == 1)
					winners++;
				total_uframes += c.umodel.get_frames();
				total_bframes += c.bmodel.get_frames();
				DEBUG("Update unit model\n");
				float ureward = update_model(c.winner, bh.umodel, c.umodel, ustats, u_reward_map, bh.avg_ureward, false);
				DEBUG("Update build model\n");
				float breward = update_model(c.winner, bh.bmodel, c.bmodel, bstats, b_reward_map, bh.avg_breward, false);
				// if(ureward != breward) {
				// 	std::cout << "REWARD IS NOT THE SAME" << std::endl;
				// }
				total_ureward += ureward;
				total_breward += breward;
				float avg_game_reward = 0.0f;
				if(c.umodel.get_frames() > 0) {
					float avg_game_ureward = ureward / c.umodel.get_frames();
					avg_game_reward += avg_game_ureward;
				}
				if(c.bmodel.get_frames() > 0) {
					float avg_game_breward = breward / c.bmodel.get_frames();
					avg_game_reward += avg_game_breward;
				}
				if(avg_game_reward > high_reward) {
					high_reward = avg_game_reward;
					// high_game = line;
				}
			}
			bh.umodel.optimizer.step();
			bh.bmodel.optimizer.step();
		}

		// saveModel(m, std::string(argv[4]));
		if(total_uframes > 0) {
			bh.avg_ureward = total_ureward / total_uframes;
		}
		if(total_bframes > 0) {
			bh.avg_breward = total_breward / total_bframes;
		}
		bh.umodel.avg_rewards.push_back(bh.avg_ureward);
		bh.bmodel.avg_rewards.push_back(bh.avg_breward);
		bh.save(argv[4]);
		std::cout << std::endl << total << " games with " << winners << " winners, avg. reward " << bh.avg_ureward << std::endl;
		std::cout << "Highest reward game is " << high_game << " with " << high_reward << std::endl;
		json["avg_ureward"] = bh.avg_ureward;
		json["avg_breward"] = bh.avg_breward;
		json["lr"] = LR;
		json["u_rewards"] = b_reward_map;
		json["b_rewards"] = b_reward_map;
		write_json(std::string(argv[4]) + "_stats.json", json);
		print_stats(ustats, total_uframes);
		print_stats(bstats, total_bframes);
		generate_plot(std::string(argv[4]) + "_rewards", bh);
	}
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
struct TestNet : torch::nn::Module {
 	TestNet() {
		 l1 = register_module("l1", torch::nn::Linear(2, 2));
		 torch::nn::init::ones_(l1->weight);
		 torch::nn::init::ones_(l1->bias);
	 }
	torch::Tensor forward(torch::Tensor input) {
		return l1->forward(input);
	}
	torch::nn::Linear l1{nullptr};
};

void torch_tests() {
	TestNet model1;
	TestNet model2;
	auto opt1 = torch::optim::Adam(model1.parameters(), 0.01);
	auto opt2 = torch::optim::Adam(model2.parameters(), 0.01);

	torch::Tensor data = torch::randn({10, 2});

	model1.forward(data)[0][0].backward();
	std::cout << model1.parameters()[0].grad() << std::endl;

	std::cout << "Start" << std::endl;
	for(auto &p: model1.parameters()) {
		std::cout << p << std::endl;
	}
	opt1.zero_grad();
	opt2.zero_grad();
	{
		torch::Tensor loss = torch::tensor(0.0f);
		for(int i = 0; i < 2; ++i) {
			auto out1 = model1.forward(data[i]);
			loss = torch::pow((out1.sum() - static_cast<float>(i)), 2.0);
			loss.backward();
		}
	}
	auto out21 = model2.forward(data[0]);
	auto out22 = model2.forward(data[1]);
	auto loss21 = torch::pow(out21.sum() - 0.0, 2.0);
	auto loss22 = torch::pow(out22.sum() - 1.0, 2.0);
	auto loss2 = loss21 + loss22;
	loss2.backward();
	opt1.step();
	opt2.step();
	std::cout << "End" << std::endl;
	for(auto &p: model1.parameters()) {
		std::cout << p << std::endl;
	}
	for(auto &p: model2.parameters()) {
		std::cout << p << std::endl;
	}

}


void test_load_save() {
	BrainHerder bh1(1.0);
	BrainHerder bh2(1.0);
	bh1.save("/tmp/model");
	bh2.load("/tmp/model");
	for(size_t i = 0; i < bh1.umodel.net->parameters().size(); ++i) {
		torch::Tensor diff = bh1.umodel.net->parameters()[i] - bh2.umodel.net->parameters()[i]; 
		std::cout << bh1.umodel.net->parameters()[i] << std::endl;
		std::cout << diff << std::endl;
	}
}


void test_grad() {
	TestNet model1;
	auto opt1 = torch::optim::SGD(model1.parameters(), 0.01);
	torch::Tensor data = torch::randn({10, 2});
	auto out1 = model1.forward(data);
	(-out1[0][0]).backward();
	opt1.step();
	auto out2 = model1.forward(data);
	std::cout << out2[0][0] / out1[0][0] << std::endl;

}