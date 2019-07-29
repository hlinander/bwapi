// Overmind.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <unordered_map>
#include <iostream>
#include <ctime>
#include "AIStructs.h"
//#include <torch/torch.h>

const float LR = 0.000001;  

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
void update_model(bool winner, Model<T, N> &m, Model<T, N> &experience, stat_map &stats) {
	for (int frame = 0; frame < experience.get_frames(); ++frame) {
		int distance_from_end = experience.get_frames() - frame;
		float discount = pow(0.99, distance_from_end);
		auto grads = experience.saved_grads(frame);
		m.descent(grads, discount * (winner ? LR : (-0.1*LR)));
		stats[experience.actions[frame].name()]++;
	}
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
	BrainHerder bh;
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
			std::cout << bh.umodel.states[frame] << std::endl;
			auto z = bh.umodel.forward(bh.umodel.states[frame]);
			std::cout << "LogP" << std::endl;
			std::cout << z << std::endl;
			std::cout << "softmax" << std::endl;
			std::cout << softmax(z) << std::endl;
			auto effective_lr = bh.winner ? LR : (-LR);
			std::cout << "LR: " << effective_lr << std::endl;
			bh.umodel.descent(bh.umodel.saved_grads(frame), effective_lr);
			auto zafter = bh.umodel.forward(bh.umodel.states[frame]);
			std::cout << "grad diff " << std::endl << (zafter - z);
			// std::cout << " => " << UnitActionS.at(argMax(z));
			std::cout << std::endl;
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
		while (std::getline(infile, line)) {
			// std::cout << "Updating from " << line << std::endl;
			BrainHerder c;
			c.load(line);
			total++;
			if(c.winner)
				winners++;
			total_uframes += c.umodel.get_frames();
			total_bframes += c.bmodel.get_frames();
			update_model(c.winner, bh.umodel, c.umodel, ustats);
			update_model(c.winner, bh.bmodel, c.bmodel, bstats);
		}

		// saveModel(m, std::string(argv[4]));
		bh.save(argv[4]);
		std::cout << total << " games with " << winners << " winners " << std::endl;
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
