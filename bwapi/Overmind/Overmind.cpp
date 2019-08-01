// Overmind.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <unordered_map>
#include <iostream>
#include <chrono>
#include <ctime>
#include "AIStructs.h"
//#include <torch/torch.h>

const float LR = 0.1;
const size_t BATCH_SIZE = 100;

using stat_map = std::unordered_map<std::string, size_t>;
using state_reward_map = std::unordered_map<size_t, float>;

state_reward_map u_reward_map = {{Param::TEAM_MINERALS, 1.0}};
state_reward_map b_reward_map = {{BuildParam::MINERALS, 1.0}};

void torch_tests();

void generate_plot(std::string path, BrainHerder &bh) {
	std::ofstream outfile;
	outfile.open(path, std::ios::trunc);
	for(int i = 0; i < bh.umodel.avg_rewards.size(); ++i) {
		outfile << i << " " << bh.umodel.avg_rewards[i] << " " << bh.bmodel.avg_rewards[i] << std::endl;
	}
}

void print_stats(const stat_map &s, size_t total_frames) {
	if(total_frames) {
		for(auto it = s.begin(); s.end() != it; ++it) {
			std::cout << (it->first) << ": " << static_cast<int>(100.0 * static_cast<float>(it->second) / total_frames) << "%, ";
		}
		std::cout << std::endl;
	}
}

template<typename T, size_t N>
float update_model(bool winner, Model<T, N> &m, Model<T, N> &experience, stat_map &stats, state_reward_map &sr, const float avg_reward, bool debug) {
	float win_reward = winner ? (fabs(avg_reward) + 10) : (-fabs(avg_reward) - 1);
	float reward = win_reward;
	float total_reward = reward;
	torch::Tensor loss = torch::tensor({0.0f});
	uint32_t totalmicro = 0;
	std::vector<torch::Tensor> presoftmax;
	// if(debug) {
		//m.optimizer.zero_grad();
	// }
	std::vector<float> rewards;
	rewards.reserve(experience.get_frames());
	for (int frame = experience.get_frames() - 1; frame >= 0; --frame) {
		for(auto& reward_pair: sr) {
			reward += reward_pair.second * (experience.states[frame][reward_pair.first]);
		}
		reward *= 0.99;
		total_reward += reward;
		rewards.push_back(reward);
		stats[experience.actions[frame].name()]++;
	}

	for (int frame = experience.get_frames() - 1; frame >= BATCH_SIZE; frame-=BATCH_SIZE) {
		// float adv = (reward - avg_reward);
		float adv = (reward );
		auto batch = std::vector<State<N>>(&(experience.states[frame - BATCH_SIZE]), &(experience.states[frame]));
		auto batch_rewards = std::vector<float>(&rewards[frame - BATCH_SIZE], &rewards[frame]);
		auto batch_actions = std::vector<T>( &(experience.actions[frame - BATCH_SIZE]), &(experience.actions[frame]));
		// auto old_p = torch::softmax(experience.forward(experience.states[frame]), -1);

		auto p = torch::softmax(m.forward_batch(batch), -1);

		// auto r = p[0][experience.actions[frame]] / old_p[0][experience.actions[frame]];
		// loss += torch::min(r * adv, torch::clamp(r, 1 - 0.2, 1 + 0.2) * adv);
		// auto end = std::high_resolution_clock::now();
		// totalmicro += std::duration_cast<std::microseconds>(end - start).count();

		for(int i = 0; i < BATCH_SIZE; ++i) {
			loss += batch_rewards[i] * p[i][batch_actions[i]].log();
		}

		// if(debug) {
		// 	presoftmax.push_back(p);
		// 	std::cout << experience.actions[frame].name() << std::endl;
		// 	std::cout << "Softmax: " << p << std::endl;
		// 	std::cout << "reward: " << reward << std::endl;
		// 	std::cout << "advantage: " << adv << std::endl;
		// }
		// for(auto& t: m.net->parameters()) {
		// 	std::cout << t.grad() << std::endl;
		// }
		// auto logs_after = m.forward(experience.states[frame]);
		// std::cout << "State:" << std::endl;
		// for(auto& e: experience.states[frame]) {
		// 	std::cout << e << ", " << std::endl;
		// }
		// std::cout << "reward: " << reward << std::endl;
		// std::cout << "Action: " << experience.actions[frame] << std::endl;
		// std::cout << "LOGITS_DIFF: " << (logs_after - logs) << std::endl;

	}
	// std::cout << "avg forward " << totalmicro / experience.get_frames() << std::endl;
	// auto start = std::high_resolution_clock::now();
	loss.backward();
	// auto middle = std::high_resolution_clock::now();
	//m.optimizer.step();
	if(debug) {
		// for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
		// 	int action = experience.actions[frame];
		// 	auto logs = m.forward(experience.states[frame]);
		// 	auto np = torch::softmax(logs, -1);
		// 	auto nr = np[0][action] / presoftmax[frame-1][0][action];
		// 	std::cout << "r: " << nr << std::endl;
		// }
	}
	// auto end = std::high_resolution_clock::now();
	// auto tback = std::duration_cast<std::microseconds>(middle - start).count();
	// auto topt = std::duration_cast<std::microseconds>(end - middle).count();
	// std::cout << "back: " << tback << std::endl;
	// std::cout << "opt: " << topt << std::endl;
	return total_reward;
}

int main(int argc, char* argv[])
{
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
	else if (std::string(argv[1]) == "-debug") {
		// loadModel(m, argv[2]);
		BrainHerder c(0.0f);
		stat_map ustats;
		stat_map bstats;
		bh.load(argv[2]);
		c.load(argv[2]);
		update_model(bh.winner, bh.umodel, c.umodel, ustats, u_reward_map, bh.avg_ureward, true);
		update_model(bh.winner, bh.bmodel, c.bmodel, bstats, b_reward_map, bh.avg_breward, true);
		// for (int frame = 0; frame < bh.bmodel.get_frames(); ++frame) {
		// 	std::cout << std::endl << "#" << frame << " " << (bh.winner ? "WINNER" : "LOOSER") <<
		// 		" " << bh.bmodel.actions[frame].name() << " " << static_cast<int>(bh.bmodel.probs[frame] * 100.0) << "%"
		// 		<< std::endl;
		// 	for(int i = 0; i < MAX_BUILD; ++i) {
		// 		std::cout << build_param_to_string(static_cast<BuildParam>(i)) << " " << bh.bmodel.states[frame][i] << std::endl;
		// 	}
		// }
		// for (int frame = 0; frame < bh.umodel.get_frames(); ++frame) {
		// 	std::cout << std::endl << "#" << frame << " " << (bh.winner ? "WINNER" : "LOOSER") <<
		// 		" " << bh.umodel.actions[frame].name() << " " << static_cast<int>(bh.umodel.probs[frame] * 100.0) << "%"
		// 		<< std::endl;
		// 	for(int i = 0; i < MAX_PARAM; ++i) {
		// 		std::cout << param_to_string(static_cast<Param>(i)) << " " << bh.umodel.states[frame][i] << std::endl;
		// 	}
			// auto z = bh.umodel.forward(bh.umodel.states[frame]);
			// std::cout << "LogP" << std::endl;
			// std::cout << z << std::endl;
			// std::cout << "softmax" << std::endl;
			// std::cout << torch::softmax(z, -1) << std::endl;
			// auto effective_lr = bh.winner ? LR : (-LR);
			// std::cout << "LR: " << effective_lr << std::endl;
			// bh.umodel.descent(bh.umodel.saved_grads(frame), effective_lr);
			// auto zafter = bh.umodel.forward(bh.umodel.states[frame]);
			// std::cout << "grad diff " << std::endl << (zafter - z);
			// // std::cout << " => " << UnitActionS.at(argMax(z));
			// std::cout << std::endl;
		// }
	}
	else if (std::string(argv[1]) == "-show") {
		std::cout << "Showing " << argv[2] << std::endl;
		// loadModel(m, argv[2]);
		bh.load(argv[2]);
		std::cout << (bh.winner ? "WINNER" : "LOSER") << std::endl;
		std::cout << bh.umodel << std::endl;
	}
	else if (std::string(argv[1]) == "-update") {
		if (argc < 5) {
			std::cout << "-update model result_list_file model_out" << std::endl;
			exit(0);
		}
		bh.load(argv[2]);
		std::ifstream infile(argv[3]);
		std::string line;
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
		bh.umodel.optimizer.zero_grad();
		bh.bmodel.optimizer.zero_grad();
		while (std::getline(infile, line)) {
			BrainHerder c(0.0f);
			c.load(line);
			total++;
			if(c.winner)
				winners++;
			total_uframes += c.umodel.get_frames();
			total_bframes += c.bmodel.get_frames();
			float ureward = update_model(c.winner, bh.umodel, c.umodel, ustats, u_reward_map, bh.avg_ureward, false);
			float breward = update_model(c.winner, bh.bmodel, c.bmodel, bstats, b_reward_map, bh.avg_breward, false);
			// if(ureward != breward) {
			// 	std::cout << "REWARD IS NOT THE SAME" << std::endl;
			// }
			total_ureward += ureward;
			total_breward += breward;
			float avg_game_ureward = ureward / c.umodel.get_frames();
			float avg_game_breward = breward / c.bmodel.get_frames();
			if(avg_game_ureward + avg_game_breward > high_reward) {
				high_reward = avg_game_ureward + avg_game_breward;
				high_game = line;
			}
			std::cout << "#" << std::flush;	
		}
		bh.umodel.optimizer.step();
		bh.bmodel.optimizer.step();

		// saveModel(m, std::string(argv[4]));
		bh.avg_ureward = total_ureward / total_uframes;
		bh.avg_breward = total_breward / total_bframes;
		bh.umodel.avg_rewards.push_back(bh.avg_ureward);
		bh.bmodel.avg_rewards.push_back(bh.avg_breward);
		bh.save(argv[4]);
		std::cout << total << " games with " << winners << " winners, avg. reward " << bh.avg_ureward << std::endl;
		std::cout << "Highest reward game is " << high_game << " with " << high_reward << std::endl;
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