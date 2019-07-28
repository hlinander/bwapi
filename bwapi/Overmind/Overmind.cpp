// Overmind.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <ctime>
#include "AIStructs.h"

const float LR = 0.001;  

int main(int argc, char* argv[])
{
	srand((unsigned int)time(0));
	if (argc < 3) {
		std::cout << "Overmind\n"; 
		std::cout << "-create name\n"; 
		exit(0);
	}
	Model m = {};
	if (std::string(argv[1]) == "-create") {
		saveModel(m, argv[2]);
	}
	else if (std::string(argv[1]) == "-debug") {
		loadModel(m, argv[2]);
		for (int frame = 0; frame < m.get_frames(); ++frame) {
			//loadModel(m, argv[2]);
			std::cout << std::endl << "#" << frame << " " << (m.winner ? "WINNER" : "LOOSER") <<
				" " << ActionS.at(m.actions[frame]) << " " << static_cast<int>(m.probs[frame] * 100.0) << "%"
				<< std::endl;
			std::cout << m.states[frame];
			auto z = m.forward(m.states[frame]);
			std::cout << z << std::endl;
			std::cout << softmax(z) << std::endl;
			auto effective_lr = m.winner ? LR : (-LR);
			std::cout << "LR: " << effective_lr << std::endl;
			m.descent(m.saved_grads(frame), effective_lr);
			auto zafter = m.forward(m.states[frame]);
			std::cout << "grad diff " << std::endl << (zafter - z);
			std::cout << " => " << ActionS.at(argMax(z));
			std::cout << std::endl;
		}
	}
	else if (std::string(argv[1]) == "-show") {
		std::cout << "Showing " << argv[2] << std::endl;
		loadModel(m, argv[2]);
		std::cout << m << std::endl;
	}
	else if (std::string(argv[1]) == "-update") {
		if (argc < 5) {
			std::cout << "-update model result_list_file model_out" << std::endl;
			exit(0);
		}
		Model m = Model();
		loadModel(m, argv[2]);
		std::ifstream infile(argv[3]);
		std::string line;
		int winners = 0;
		int total = 0;
		int total_frames = 0;
		int attack = 0;
		while (std::getline(infile, line)) {
			// std::cout << "Updating from " << line << std::endl;
			Model c = Model();
			loadModel(c, line);
			total++;
			if(c.winner)
				winners++;
			// std::cout << c.get_frames() << " actions" << std::endl;
			for (int frame = 0; frame < c.get_frames(); ++frame) {
				total_frames++;
				if(c.actions[frame] == Action::ATTACK)
					attack++;
				int distance_from_end = c.get_frames() - frame;
				float discount = pow(0.99, distance_from_end);
				auto grads = c.saved_grads(frame);
				m.descent(grads, discount * (c.winner ? LR : (-0.1 * LR)));
				if (frame == 0) {
					// std::cout << "Mean hidden " << m.hidden.mean() << std::endl;
					// std::cout << "Mean out " << m.out.mean() << std::endl;
					// std::cout << "[Frame 0] Mean grads * lr: ";
					// for (auto& grad : grads) {
					// 	std::cout << LR * grad.mean() << " : ";
					// }
					// std::cout << std::endl;
				}
			}
		}
		saveModel(m, std::string(argv[4]));
		int attack_percent = 0;
		if(total_frames > 0) {
			attack_percent = static_cast<int>(100.0 * static_cast<float>(attack) / static_cast<float>(total_frames));
		}
		std::cout << total << " games with " << winners << " winners " << attack_percent << "% attacks" << std::endl;
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
