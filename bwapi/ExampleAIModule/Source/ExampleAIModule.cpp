#include "ExampleAIModule.h"

#include <ctime>
#include <chrono>

#include <iostream>
#include <fstream>
#include "secret.stash"

#define MAX_FRAMES 3000

using namespace BWAPI;
using namespace Filter;

float get_seconds(std::chrono::high_resolution_clock::time_point start) {
	auto dur = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start);
	return dur.count();
}

static std::vector<std::string> get_tokens(std::string s) {
	size_t pos = 0;
	std::vector<std::string> tokens;
	if(s.empty()) {
		std::cout << "THE PANTS THE PANTS" << std::endl;
	}
	while ((pos = s.find("_")) != std::string::npos) {
		tokens.push_back(s.substr(0, pos));
		s.erase(0, pos + 1);
	}
	tokens.push_back(s);
	return tokens;
}

static std::string get_dbname() {
	std::stringstream ss;
	auto tokens = get_tokens(Broodwar->self()->getName());
	ss << "models/" << tokens[0] << tokens[2];
	return ss.str();
}

static int get_generation() {
	std::stringstream ss;
	auto tokens = get_tokens(Broodwar->self()->getName());
	return std::atoi(tokens[1].c_str());
}

static std::string get_resname() {
	std::stringstream ss;
	auto tokens = get_tokens(Broodwar->self()->getName());
	// std::cout << "Tokens: ";
	// for (auto& token : tokens) {
	// 	std::cout << token << " : ";
	// }
	ss << "results/" << tokens[0] << tokens[2];
	return ss.str();
}

template<typename T>
static constexpr float mf(T x) {
	return static_cast<float>(x);
}
static void writeUnitState(BWAPI::Unit u, UnitParam &up) {
	up.attacked = u->isUnderAttack();
	up.repaired = u->isBeingHealed();

	up.constructing = u->isConstructing();
	up.repairing = u->isRepairing();
	up.gathering = u->isGatheringGas() || u->isGatheringMinerals();
	up.attacking = u->isAttacking();
	up.moving = u->isMoving();
	up.hp = mf(u->getHitPoints()) / mf(u->getType().maxHitPoints());
	up.unit_types[SCV] = u->getType() == BWAPI::UnitTypes::Terran_SCV;
	up.unit_types[MARINE] = u->getType() == BWAPI::UnitTypes::Terran_Marine;
}
static StateParam createState() {
	float n_buildings = Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Buildings);
	float n_units = Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Terran_SCV) + Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Terran_Marine);
	StateParam p;
	p.minerals = Broodwar->self()->minerals() / 2000.0;
	p.gas = Broodwar->self()->gas() / 2000.0;
	p.n_supply_depots = Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Terran_Supply_Depot) / n_buildings;
	p.n_barracks = Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Terran_Barracks) / n_buildings;
	p.n_marines = Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Terran_Marine) / n_units;
	p.supply = static_cast<float>(Broodwar->self()->supplyUsed()) / static_cast<float>(Broodwar->self()->supplyTotal());
	size_t index = 0;
	for(auto& u: Broodwar->self()->getUnits()) {
		writeUnitState(u, p.friendly[index]);
		++index;
	}
	index = 0;
	for(auto& u: Broodwar->enemy()->getUnits()) {
		writeUnitState(u, p.enemy[index]);
		++index;
	}
	return p;
}

// template <typename T>
// BWAPI::Unit find_unit_in(BWAPI::Unitset us, T cb) {
// 	for (auto &x : us) {
// 		if (cb(x)) {
// 			return x;
// 		}
// 	}
// 	return nullptr;
// }

template <class TContainer, typename T>
BWAPI::Unit find_max(TContainer us, T cb) {
	if (0 == us.size()) {
		return nullptr;
	}
	else if (1 == us.size()) {
		return us[0];
	}
	auto win = us.begin();
	for (auto it = us.begin() + 1; us.end() != it; ++it) {
		if (cb(*it, *win)) {
			win = it;
		}
	}
	return *win;
}

// template <typename T>
// BWAPI::Unit find_max_friend(T cb) {
// 	return find_max(Broodwar->self()->getUnits(), cb);
// }

// template <typename T>
// BWAPI::Unit find_max_enemy(T cb) {
// 	return find_max(Broodwar->enemy()->getUnits(), cb);
// }

// template <typename T>
// BWAPI::Unit find_friend(T cb) {
// 	return find_unit_in(Broodwar->self()->getUnits(), cb);
// }

// template <typename T>
// BWAPI::Unit find_enemy(T cb) {
// 	return find_unit_in(Broodwar->enemy()->getUnits(), cb);
// }

template <typename T>
std::vector<BWAPI::Unit> filter_units(BWAPI::Unitset us, T cb) {
	std::vector<BWAPI::Unit> out;
	for (auto &ref : us) {
		if (cb(ref)) {
			out.emplace_back(ref);
		}
	}
	return out;
}

template <typename T>
std::vector<BWAPI::Unit> filter_friendly(T cb) {
	return filter_units(Broodwar->self()->getUnits(), cb);
}

template <typename T>
std::vector<BWAPI::Unit> filter_enemy(T cb) {
	return filter_units(Broodwar->enemy()->getUnits(), cb);
}

template<typename T>
bool filter(BWAPI::Unitset us, const BWAPI::UnitFilter &pred, T cb) {
	for(auto& u: us) {
		if(!pred.isValid() || pred(u)) {
			if(!cb(u)) {
				return false;
			}
		}
	}
	return true;
}

bool requirements(BWAPI::UnitType t) {
	if(Broodwar->self()->minerals() > t.mineralPrice() && 
	   Broodwar->self()->supplyTotal() - Broodwar->self()->supplyUsed() >= t.supplyRequired()) {
		   return true;
	   }
	return false;
}

static bool commitAction(BuildAction a) {
	bool didCommit = false;
	if (a == BuildAction::Type::BUILD_SCV) {
		filter(Broodwar->self()->getUnits(), BWAPI::Filter::GetType == BWAPI::UnitTypes::Terran_Command_Center, 
			[&didCommit](BWAPI::Unit u){
				if(requirements(BWAPI::UnitTypes::Terran_SCV)) {
					u->train(BWAPI::UnitTypes::Terran_SCV);
					didCommit = true;
				}
				return false;
				});
	}
	else if (a == BuildAction::Type::BUILD_MARINE) {
		filter(Broodwar->self()->getUnits(), BWAPI::Filter::GetType == BWAPI::UnitTypes::Terran_Barracks, 
			[&didCommit](BWAPI::Unit u){
				if(!u->isTraining()) {
					if(requirements(BWAPI::UnitTypes::Terran_Marine)) {
						u->train(BWAPI::UnitTypes::Terran_Marine);
						didCommit = true;
					}
					return false;
				}
				return true;
				});
	}
	else if (a == BuildAction::Type::BUILD_SUPPLY) {
		filter(Broodwar->self()->getUnits(), BWAPI::Filter::IsWorker, 
			[&didCommit](BWAPI::Unit u){
				if(!u->isConstructing()) {
					if(requirements(BWAPI::UnitTypes::Terran_Supply_Depot)) {
						auto target_loc = Broodwar->getBuildLocation(BWAPI::UnitTypes::Terran_Supply_Depot, u->getTilePosition());
						u->build(BWAPI::UnitTypes::Terran_Supply_Depot, target_loc);
						didCommit = true;
					}
					return false;
				}
				return true;
				});
	}
	else if (a == BuildAction::Type::BUILD_BARRACK) {
		filter(Broodwar->self()->getUnits(), BWAPI::Filter::IsWorker, 
			[&didCommit](BWAPI::Unit u){
				if(!u->isConstructing()) {
					if(requirements(BWAPI::UnitTypes::Terran_Barracks)) {
						auto target_loc = Broodwar->getBuildLocation(BWAPI::UnitTypes::Terran_Barracks, u->getTilePosition());
						u->build(BWAPI::UnitTypes::Terran_Barracks, target_loc);
						didCommit = true;
					}
					return false;
				}
				return true;
				});
	}
	else if (a == BuildAction::Type::IDLE) {
		didCommit = true;
	}
	return didCommit;
}

static bool commitAction(UnitAction a, Unit me, bool debug) {
	bool didCommit = false;
	if (a == UnitAction::Type::ATTACK) {
		if (debug)
			std::cout << "MY QUEST IS TO ATTACK" << std::endl;
		//auto enemyUnits{ filter_enemy([](auto u) {
		//	return !u->getType().isBuilding();
		//})};
		//auto target = find_max(enemyUnits, [](Unit left, Unit right) {
		//	return left->getHitPoints() < right->getHitPoints();
		//});
		auto target = me->getClosestUnit(IsEnemy && !IsBuilding);
		if (target/* && !me->getType().isWorker()*/) {
			me->attack(target);
			Broodwar->drawLineMap(me->getPosition(), target->getPosition(), Color(255, 0, 0));
			didCommit = true;
		}
	}
	else if (a == UnitAction::Type::REPAIR) {
		if (debug)
			std::cout << "MY QUEST IS TO REPAIR" << std::endl;
		auto friendlyUnits{ filter_friendly([](auto u) {
			return !u->getType().isBuilding();
		}) };
		auto target = find_max(friendlyUnits, [](Unit left, Unit right) {
			return left->getHitPoints() < right->getHitPoints();
		});
		if (target && me->getType() == BWAPI::UnitTypes::Terran_SCV
				   && target->getHitPoints() < target->getType().maxHitPoints()) {
			me->repair(target);
			Broodwar->drawLineMap(me->getPosition(), target->getPosition(), Color(0, 255, 0));
			didCommit = true;
		}
	}
	else if (a == UnitAction::Type::FLEE) {
		if (debug)
			std::cout << "MY QUEST IS TO RUN AWAAAAY!!!" << std::endl;
		auto target = me->getClosestUnit(IsEnemy && !IsBuilding);
		if (target) {
			auto delta = target->getPosition() - me->getPosition();
			auto move_target = me->getPosition() + delta * (-1);
			Broodwar->drawLineMap(me->getPosition(), move_target, Color(0, 0, 255));
			me->move(move_target);
			didCommit = true;
		}
	}
	else if (a == UnitAction::Type::MINE) {
		if (debug)
			std::cout << "MINE QUEST IS TO MINE!!!" << std::endl;
		if(me->isGatheringMinerals()) {
			didCommit = true;
		}
		if(me->getType().isWorker() && !me->isGatheringMinerals()) {
			if(me->isCarryingMinerals()) {
				me->returnCargo();
				didCommit = true;
			}
			else {
				auto target = me->getClosestUnit(IsMineralField);
				if (target) {
					me->gather(target);
					didCommit = true;
				}
			}
		}
	}
	return didCommit;
}


void ExampleAIModule::onStart()
{
	Benchmark onstart{"onStart"};
	debug = false;
	char buf[255] = {};
	if (nullptr != getenv("DEBUG")) {
		debug = true;
	}
	// std::cout << "Loading model at " << get_dbname() << std::endl;
	{
		Benchmark load{"load"};
		if (!bh.load(get_dbname())) {
			Broodwar->sendText("My pants are on my head!");
		}
	}
	generation = get_generation();
	bh.umodel.net->eval();
	bh.bmodel.net->eval();
	// std::cout << "Model leaded" << std::endl;
	// Hello World!
	// Broodwar->sendText("Hello world!");

	// Print the map name.
	// BWAPI returns std::string when retrieving a string, don't forget to add .c_str() when printing!
	// Broodwar << "The map is " << Broodwar->mapName() << "!" << std::endl;

	// Enable the UserInput flag, which allows us to control the bot and type messages.
	//Broodwar->enableFlag(Flag::UserInput);

	// Uncomment the following line and the bot will know about everything through the fog of war (cheat).
	Broodwar->enableFlag(Flag::CompleteMapInformation);

	// Set the command optimization level so that common commands can be grouped
	// and reduce the bot's APM (Actions Per Minute).
	Broodwar->setCommandOptimizationLevel(0);
	Broodwar->setLocalSpeed(0);

	// Check if this is a replay
	if (Broodwar->isReplay())
	{

		// Announce the players in the replay
		Broodwar << "The following players are in this replay:" << std::endl;

		// Iterate all the players in the game using a std:: iterator
		Playerset players = Broodwar->getPlayers();
		for (auto p : players)
		{
			// Only print the player if they are not an observer
			// if (!p->isObserver())
				// Broodwar << p->getName() << ", playing as " << p->getRace() << std::endl;
		}
	}
	else // if this is not a replay
	{
		// Retrieve you and your enemy's races. enemy() will just return the first enemy.
		// If you wish to deal with multiple enemies then you must use enemies().
		// if (Broodwar->enemy()) // First make sure there is an enemy
		// 	Broodwar << "The matchup is " << Broodwar->self()->getRace() << " vs " << Broodwar->enemy()->getRace() << std::endl;
	}
	start_time = time(NULL);
	start_chrono = std::chrono::high_resolution_clock::now();
	force_lose = false;
	// std::cout << "SCV build time " << BWAPI::UnitTypes::Terran_SCV.buildTime() << std::endl;
	// std::cout << "Starting" << std::endl; 
}

void ExampleAIModule::onEnd(bool isWinner)
{
	//char buf[MAX_PATH] = {};
	//if (0 != GetEnvironmentVariableA("resultfile", buf, sizeof(buf)))
	//{
	//	std::ofstream out(buf);
	//	out << (isWinner ? "1" : "0");
	//}
	if(force_lose) {
		bh.winner = 0; // Draw from timeout
	}
	else {
		bh.winner = isWinner ? 1 : (-1);
	}
	// std::cout << "THE END! I AM " << (isWinner ? "WINNER" : "LOOSER") << std::endl;
	std::cout << "Trying to save at " << get_resname() << std::endl;
	bh.save(get_resname());
	std::cout << "(" << bh.umodel.get_frames() << ", " <<  bh.bmodel.get_frames() << ")" << std::endl;
	cb.print();
	fflush(stdout);

//   std::cout << "THEOTHERSIDE!" << std::endl;
}

static uint32_t frames = -1;
static uint32_t nexttick = 0;

static bool action_allowed(int frame, int generation, int max_freq) {
	float full_grown_generation = 100;
	float min_freq = 30 * 10;
	if(generation >= full_grown_generation && frame % max_freq == 0) {
		return true;
	}
	// generation = full_grown_generation => freq = max_freq 
	// min_freq - (min_freq - C) = max_frax
	int freq = (min_freq - (min_freq - max_freq) * (float)generation / full_grown_generation);
	if(0 == frame % freq) {
		return true;
	}
	return false;
}

void ExampleAIModule::onFrame()
{
	// Called once every game frame

	// Return if the game is a replay or is paused
	if (Broodwar->isReplay() || Broodwar->isPaused() || !Broodwar->self())
		return;

	// Prevent spamming by only running our onFrame once every number of latency frames.
	// Latency frames are the number of frames before commands are processed.
	if (Broodwar->getFrameCount() % Broodwar->getLatencyFrames() != 0)
		return;

	bool did_lose = true;

	++frames;
	if (frames == (MAX_FRAMES - 100)) {
		// Broodwar->sendText("timeout");
		force_lose = true;
	}
	else if(frames > MAX_FRAMES) {
		// std::cout << "@@@ Timeout!" << std::endl;
		//onEnd(false);
		Broodwar->leaveGame();
	}
	Benchmark bmaction{cb, "buildAction"};
	auto state{ createState() };
	if(action_allowed(frames, generation, 15))
	{
		Benchmark bmfullaction{cb, "buildFullAction"};
		auto baction{ bh.bmodel.get_action(state) };
		Benchmark bmcaction{cb, "commitBuildAction"};
		if(commitAction(baction)) {
			if(baction == BA::IDLE) {
				bh.bmodel.record_action(state, baction, -0.1, frames);
			}
			else {
				bh.bmodel.record_action(state, baction, 0.0, frames);
			}
		}
		else {
			bh.bmodel.record_action(state, baction, -0.1, frames);
		}
	}
	bmaction.stop();

	// Iterate through all the units that we own
	for (auto &u : Broodwar->self()->getUnits())
	{
		// Ignore the unit if it no longer exists
		// Make sure to include this block when handling any Unit pointer!
		if (!u->exists())
			continue;

		if (!u->getType().isBuilding())
		{
			did_lose = false;
		}

		// Ignore the unit if it has one of the following status ailments
		if (u->isLockedDown() || u->isMaelstrommed() || u->isStasised())
			continue;

		// Ignore the unit if it is in one of the following states
		if (u->isLoaded() || !u->isPowered() || u->isStuck())
			continue;

		// Ignore the unit if it is incomplete or busy constructing
		if (!u->isCompleted() || u->isConstructing())
			continue;


		// Finally make the unit do some stuff!


		// If the unit is a worker unit
		if (!u->getType().isBuilding())
		{
			if(action_allowed(frames, generation, 4))
			{
				Benchmark bmuaction{cb, "unitAction"};
				writeUnitState(u, state.me);
				auto uaction{ bh.umodel.get_action(state) };
				Benchmark bmucaction{cb, "commitUnitAction"};
				if(commitAction(uaction, u, debug)) {
					if(generation < 10) {
						Broodwar->sendText("My function is to %s", uaction.name());
					}
					bh.umodel.record_action(state, uaction, 0.0f, frames);
				}
				else {
					bh.umodel.record_action(state, uaction, -0.1f, frames);
				}
			} // closure: if idle
		}
	} // closure: unit iterator

	if (did_lose) {
		Broodwar->leaveGame();
	}
	// double elapsed = difftime(time(NULL), start_time);
	// if (debug)
	// 	std::cout << "@ ELAPSED " << elapsed << std::endl;

}

void ExampleAIModule::onSendText(std::string text)
{

	// Send the text to the game if it is not being processed.
	Broodwar->sendText("%s", text.c_str());


	// Make sure to use %s and pass the text as a parameter,
	// otherwise you may run into problems when you use the %(percent) character!

}

void ExampleAIModule::onReceiveText(BWAPI::Player player, std::string text)
{
	// Parse the received text
	Broodwar << player->getName() << " said \"" << text << "\"" << std::endl;
	if (text == "timeout") {
		//onEnd(false);
		Broodwar->leaveGame();
	}
	if (text == "win") {
		//onEnd(true);
	}
}

void ExampleAIModule::onPlayerLeft(BWAPI::Player player)
{
	// Interact verbally with the other players in the game by
	// announcing that the other player has left.
	// Broodwar->sendText("Goodbye %s!", player->getName().c_str());
}

void ExampleAIModule::onNukeDetect(BWAPI::Position target)
{

	// Check if the target is a valid position
	if (target)
	{
		// if so, print the location of the nuclear strike target
		Broodwar << "Nuclear Launch Detected at " << target << std::endl;
	}
	else
	{
		// Otherwise, ask other players where the nuke is!
		Broodwar->sendText("Where's the nuke?");
	}

	// You can also retrieve all the nuclear missile targets using Broodwar->getNukeDots()!
}

void ExampleAIModule::onUnitDiscover(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitEvade(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitShow(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitHide(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitCreate(BWAPI::Unit unit)
{
	if (Broodwar->isReplay())
	{
		// if we are in a replay, then we will print out the build order of the structures
		if (unit->getType().isBuilding() && !unit->getPlayer()->isNeutral())
		{
			int seconds = Broodwar->getFrameCount() / 24;
			int minutes = seconds / 60;
			seconds %= 60;
			Broodwar->sendText("%.2d:%.2d: %s creates a %s", minutes, seconds, unit->getPlayer()->getName().c_str(), unit->getType().c_str());
		}
	}
}

void ExampleAIModule::onUnitDestroy(BWAPI::Unit unit)
{
}

void ExampleAIModule::onUnitMorph(BWAPI::Unit unit)
{
	if (Broodwar->isReplay())
	{
		// if we are in a replay, then we will print out the build order of the structures
		if (unit->getType().isBuilding() && !unit->getPlayer()->isNeutral())
		{
			int seconds = Broodwar->getFrameCount() / 24;
			int minutes = seconds / 60;
			seconds %= 60;
			Broodwar->sendText("%.2d:%.2d: %s morphs a %s", minutes, seconds, unit->getPlayer()->getName().c_str(), unit->getType().c_str());
		}
	}
}

void ExampleAIModule::onUnitRenegade(BWAPI::Unit unit)
{
}

void ExampleAIModule::onSaveGame(std::string gameName)
{
	Broodwar << "The game was saved to \"" << gameName << "\"" << std::endl;
}

void ExampleAIModule::onUnitComplete(BWAPI::Unit unit)
{
}
