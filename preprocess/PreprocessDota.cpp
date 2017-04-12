//============================================================================
// Name        : PreprocessDota.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "json.hpp"
#include <string>
#include <set>
#include <vector>

#define MAX_HEROES 120

std::set<int> VALID_LOBBY_TYPES {0, 2, 5, 6, 7};
std::set<int> VALID_MODES {1, 2, 3, 4, 5, 12, 16, 22};

bool isValidMatch(nlohmann::json &match) {
	return match["human_players"].get<int>() == 10 &&
		   match["duration"].get<int>() >= 600 &&
		   match["players"].size() == 10 &&
		   VALID_LOBBY_TYPES.count(match["lobby_type"].get<int>()) == 1 &&
		   VALID_MODES.count(match["game_mode"].get<int>());
}

int main(int argc, char** argv) {
	std::string line;
	while (getline(std::cin, line)) {
		nlohmann::json match = nlohmann::json::parse(line);
		if (isValidMatch(match)) {
			std::vector<bool> heroes(MAX_HEROES * 2);
			for (unsigned int i = 0; i < heroes.size(); i++) {
				heroes[i] = false;
			}
			for (int i = 0; i < 10; i++) {
				int hero = match["players"][i]["hero_id"].get<int>();
				bool radiant = match["players"][i]["player_slot"].get<int>() < 128;
				heroes[hero + (radiant ? 0 : MAX_HEROES)] = true;
			}
			for (unsigned int i = 0; i < heroes.size(); i++) {
				std::cout << (heroes[i] ? '1' : '0') << ',';
			}
			std :: cout << (match["radiant_win"].get<bool>() ? '1' : '0') << '\n';
		} else {
			// nothing
		}
	}
	return 0;
}
