// Overmind.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <unordered_map>
#include <iostream>
#include <chrono>
#include <ctime>
#include "AIStructs.h"
//#include <torch/torch.h>

const float LR = 0.00001;//0.00000001;
const size_t BATCH_SIZE = 20;

using stat_map = std::unordered_map<std::string, size_t>;
using state_reward_map = std::unordered_map<size_t, float>;

state_reward_map u_reward_map = {{Param::TEAM_MINERALS, 1.0}};
state_reward_map b_reward_map = {{BuildParam::MINERALS, 1.0}};

void torch_tests();
void test_batch();
void test_load_save();

void generate_plot(std::string path, BrainHerder<BATCH_SIZE> &bh) {
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

struct Reward {
	std::vector<float> rewards;
	float total_reward;
};

template<typename T, size_t N, size_t BS>
Reward calculate_rewards(Model<T, N, BS> &experience, state_reward_map &sr) {
	Reward ret;
	ret.rewards.reserve(experience.get_frames());
	std::fill(std::begin(ret.rewards), std::end(ret.rewards), 0.0f);
	ret.total_reward = 0.0;
	float reward = 0.0;//win_reward;
	for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
		for(auto& reward_pair: sr) {
			reward += reward_pair.second * (experience.states[frame][reward_pair.first]) - reward_pair.second * (experience.states[frame - 1][reward_pair.first]);
			// reward += reward_pair.second * (experience.states[frame][reward_pair.first]);
		}
		reward *= 0.99;
		ret.total_reward += reward;
		ret.rewards[frame] = reward;
	}
	return ret;
}

template<typename T, size_t N, size_t BS>
float update_model(bool winner, Model<T, N, BS> &m, Model<T, N, BS> &experience, stat_map &stats, state_reward_map &sr, const float avg_reward, bool debug) {
	float win_reward = winner ? (fabs(avg_reward) + 10) : (-fabs(avg_reward) - 1);
	torch::Tensor loss = torch::tensor({0.0f});
	Reward reward = calculate_rewards(experience, sr);

	for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
		stats[experience.actions[frame].name()]++;
	}

	for (int frame = experience.get_frames() - 1; frame >= BATCH_SIZE; frame-=BATCH_SIZE) {
		// float adv = (reward - avg_reward);
		// float adv = (rewards[frame]);
		auto batch = std::vector<State<N>>(&(experience.states[frame - BATCH_SIZE]), &(experience.states[frame]));
		auto batch_rewards = std::vector<float>(&reward.rewards[frame - BATCH_SIZE], &reward.rewards[frame]);
		auto batch_actions = std::vector<T>( &(experience.actions[frame - BATCH_SIZE]), &(experience.actions[frame]));

		auto logp = m.forward_batch(batch);
		auto p = torch::softmax(logp, -1);
		auto old_p = torch::softmax(experience.forward_batch(batch), -1);

		for(int i = 0; i < BATCH_SIZE; ++i) {
			// float adv = (batch_rewards[i] - avg_reward);
			float adv = batch_rewards[i];
			auto r = p[i][experience.actions[frame]] / old_p[i][experience.actions[frame]];
			// auto re = adv * p[i][experience.actions[frame]].log();
			// loss += r;
			loss += torch::min(r * adv, torch::clamp(r, 1 - 0.2, 1 + 0.2) * adv);
		}
	}
	loss.backward();
	if(debug) {
		// for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
		// 	int action = experience.actions[frame];
		// 	auto logs = m.forward(experience.states[frame]);
		// 	auto np = torch::softmax(logs, -1);
		// 	auto nr = np[0][action] / presoftmax[frame-1][0][action];
		// 	std::cout << "r: " << nr << std::endl;
		// }
	}
	return reward.total_reward;
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
	BrainHerder<BATCH_SIZE> bh(LR);
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
		test_batch();
	}
	else if (std::string(argv[1]) == "-savetest") {
		//saveModel(m, argv[2]);
		test_load_save();
	}
	else if (std::string(argv[1]) == "-debug") {
		BrainHerder<1> prev(0.0f);
		BrainHerder<1> next(0.0f);
		stat_map ustats;
		stat_map bstats;
		prev.load(argv[2]);
		next.load(argv[3]);
		Reward reward = calculate_rewards(prev.umodel, u_reward_map);
		for(int frame = 0; frame < prev.umodel.get_frames(); ++frame) {
			auto p = torch::softmax(prev.umodel.forward(prev.umodel.states[frame]), -1);
			auto pnext = torch::softmax(next.umodel.forward(prev.umodel.states[frame]), -1);
			auto action = prev.umodel.actions[frame];
			float r = pnext[0][action].item<float>() / p[0][action].item<float>();
			float creward = reward.rewards[frame];
			float adv = reward.rewards[frame] - prev.avg_ureward;
			std::cout << action.name() << " @ " << p[0][action].item<float>() << " reward: " << creward << " adv: " << adv  << " r: " << r << std::endl;
			std::cout << prev.umodel.states[frame][Param::TEAM_MINERALS] * 2000.0 << std::endl;

		}
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
		std::cout << (bh.winner ? "WINNER" : "LOSER") << std::endl;
		std::cout << bh.umodel << std::endl;
	}
	else if (std::string(argv[1]) == "-update") {
		if (argc < 5) {
			std::cout << "-update model result_list_file model_out" << std::endl;
			exit(0);
		}
		bh.load(argv[2]);
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
		std::cout << (bh.umodel.net->is_training() ? "TRAINING" : "EVALUATING") << std::endl;
		std::cout << "LR: " << bh.umodel.optimizer.options.learning_rate() << std::endl;
		for(int epoch = 0; epoch < 6; epoch++) {
			bh.umodel.optimizer.zero_grad();
			bh.bmodel.optimizer.zero_grad();
			std::ifstream infile(argv[3]);
			while (std::getline(infile, line)) {
				BrainHerder<BATCH_SIZE> c(0.0f);
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
		}

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
	BrainHerder<BATCH_SIZE> bh1(1.0);
	BrainHerder<BATCH_SIZE> bh2(1.0);
	bh1.save("/tmp/model");
	bh2.load("/tmp/model");
	for(size_t i = 0; i < bh1.umodel.net->parameters().size(); ++i) {
		torch::Tensor diff = bh1.umodel.net->parameters()[i] - bh2.umodel.net->parameters()[i]; 
		std::cout << bh1.umodel.net->parameters()[i] << std::endl;
		std::cout << diff << std::endl;
	}
}


void test_batch() {
	BrainHerder<BATCH_SIZE> bh(1.0);
	torch::Tensor out;
	std::vector<float> data;
	const size_t N = 3;
	{
	data.reserve(N * Param::MAX_PARAM);
	for(int i = 0; i < N; ++i) {
		std::fill(&data[i * Param::MAX_PARAM], &data[i * Param::MAX_PARAM + Param::MAX_PARAM], 1.0);
		//std::copy(s[i].begin(), s[i].end(), &data[i * Param::MAX_PARAM]);
	}
	for(int i = 0; i < N; ++i) {
	for(int j = 0; j < Param::MAX_PARAM; ++j) {
		std::cout << data[i * Param::MAX_PARAM + j] << "|";
	}
	std::cout << std::endl;
	}
	auto ts = torch::from_blob(static_cast<void*>(data.data()), 
								{static_cast<long>(N), Param::MAX_PARAM}, torch::kFloat32);

	out = bh.umodel.net->forward(ts);
	}
	bh.umodel.optimizer.zero_grad();
	torch::Tensor loss = torch::tensor({0.0f});
	loss = out[0][0];
	loss.backward();
	std::cout << bh.umodel.net->parameters()[2].grad() << std::endl;
}