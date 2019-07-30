// Overmind.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <unordered_map>
#include <iostream>
#include <chrono>
#include <ctime>
#include "AIStructs.h"
//#include <torch/torch.h>

const float LR = 0.001;  

using stat_map = std::unordered_map<std::string, size_t>;

void print_stats(const stat_map &s, size_t total_frames) {
	if(total_frames) {
		for(auto it = s.begin(); s.end() != it; ++it) {
			std::cout << (it->first) << ": " << static_cast<int>(100.0 * static_cast<float>(it->second) / total_frames) << "%, ";
		}
		std::cout << std::endl;
	}
}

template<typename T, size_t N>
float update_model(bool winner, Model<T, N> &m, Model<T, N> &experience, stat_map &stats, const float lr, const float avg_reward) {
	float reward = 0;
	float win_reward = winner ? avg_reward : (-avg_reward);
	float total_reward = reward;
	torch::Tensor loss = torch::tensor({0.0f});
	uint32_t totalmicro = 0;
	m.optimizer.zero_grad();
	for (int frame = experience.get_frames() - 1; frame >= 0; --frame) {
			// reward += 100.0 * (experience.states[frame](Param::TEAM_COUNT) 
			// 		 - experience.states[frame - 1](Param::TEAM_COUNT));
		reward += experience.states[frame][Param::TEAM_COUNT];
		reward -= experience.states[frame][Param::ENEMY_COUNT];
		//reward += experience.states[frame][Param::TEAM_MINERALS];
		reward *= 0.99;
		total_reward += reward;
		//int distance_from_end = experience.get_frames() - frame;
		
		// r_i 
		//float discount = pow(0.99, distance_from_end);
		//auto grads = experience.saved_grads(frame);
		// if(descent) {
		//	m.descent(grads, (reward - avg_reward + win_reward) * lr);
		// }
		// auto start = std::high_resolution_clock::now();
		auto logs = m.forward(experience.states[frame]);
		// auto end = std::high_resolution_clock::now();
		// totalmicro += std::duration_cast<std::microseconds>(end - start).count();
		auto lsoftmax = torch::softmax(logs, -1);
		loss += (reward - avg_reward + win_reward) * lsoftmax[0][experience.actions[frame]].log();
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

		stats[experience.actions[frame].name()]++;
	}
	// std::cout << "avg forward " << totalmicro / experience.get_frames() << std::endl;
	// auto start = std::high_resolution_clock::now();
	loss.backward();
	// auto middle = std::high_resolution_clock::now();
	m.optimizer.step();
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
	else if (std::string(argv[1]) == "-debug") {
		// loadModel(m, argv[2]);
		bh.load(argv[2]);
		for (int frame = 0; frame < bh.umodel.get_frames(); ++frame) {
			//loadModel(m, argv[2]);
			std::cout << std::endl << "#" << frame << " " << (bh.winner ? "WINNER" : "LOOSER") <<
				" " << bh.umodel.actions[frame].name() << " " << static_cast<int>(bh.umodel.probs[frame] * 100.0) << "%"
				<< std::endl;
			for(const auto& e: bh.umodel.states[frame]) {
				std::cout << e << std::endl;
			}
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
		}
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
		while (std::getline(infile, line)) {
			BrainHerder c(0.0f);
			c.load(line);
			total++;
			if(c.winner)
				winners++;
			total_uframes += c.umodel.get_frames();
			total_bframes += c.bmodel.get_frames();
			float ureward = update_model(c.winner, bh.umodel, c.umodel, ustats, LR, bh.avg_ureward);
			float breward = update_model(c.winner, bh.bmodel, c.bmodel, bstats, 16 * LR, bh.avg_breward);
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

		// saveModel(m, std::string(argv[4]));
		bh.avg_ureward = total_ureward / total_uframes;
		bh.avg_breward = total_breward / total_bframes;
		bh.save(argv[4]);
		std::cout << total << " games with " << winners << " winners, avg. reward " << bh.avg_ureward << std::endl;
		std::cout << "Highest reward game is " << high_game << " with " << high_reward << std::endl;
		print_stats(ustats, total_uframes);
		print_stats(bstats, total_bframes);
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
