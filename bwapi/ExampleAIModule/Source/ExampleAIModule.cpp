#include "ExampleAIModule.h"

#include <ctime>

#include <iostream>
#include <fstream>

#define DEBUG(...) printf(__VA_ARGS__)


using namespace BWAPI;
using namespace Filter;

static std::vector<std::string> get_tokens(std::string s) {
	size_t pos = 0;
	std::vector<std::string> tokens;
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

	ss << "models/" << tokens[0];
	return ss.str();
}

static std::string get_resname() {
	std::stringstream ss;
	auto tokens = get_tokens(Broodwar->self()->getName());
	// std::cout << "Tokens: ";
	// for (auto& token : tokens) {
	// 	std::cout << token << " : ";
	// }
	ss << "results/" << tokens[0] << "_result_" << tokens[1];
	return ss.str();
}


static BuildState createBuildState() {
	BuildState bs;
	std::fill(std::begin(bs), std::end(bs), 0.0f);
	bs[MINERALS] = Broodwar->self()->minerals() / 2000.0;
	bs[GAS] = Broodwar->self()->gas() / 2000.0;
	bs[N_SCVS] = Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Terran_SCV) / 200.0; 
	bs[N_MARINES] = Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Terran_Marine) / 200.0;
	bs[N_SUPPLY_DEPOTS] = Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Terran_Supply_Depot) / 200.0;
	bs[N_BARRACKS] = Broodwar->self()->allUnitCount(BWAPI::UnitTypes::Terran_Barracks) / 200.0;
	return bs;
}

static UnitState createUnitState(Unit me) {
	UnitState s;
	std::fill(std::begin(s), std::end(s), 0.0f);
	for (auto u : Broodwar->enemy()->getUnits()) {
		if (!u->getType().isBuilding()) {
			s[Param::ENEMY_COUNT]++;
			s[Param::ENEMY_HP] += u->getHitPoints();
		}
	}
	s[Param::ENEMY_COUNT] *= (1.0 / 200.0);
	s[Param::ENEMY_HP] *= (1.0 / 2000.0);
	for (auto u : Broodwar->self()->getUnits()) {
		if (!u->getType().isBuilding()) {
			s[Param::TEAM_COUNT]++;
			s[Param::TEAM_HP] += u->getHitPoints();
		}
	}
	s[Param::TEAM_COUNT] *= (1.0 / 200.0);
	s[Param::TEAM_HP] *= (1.0 / 2000.0);
	s[Param::ME_HP] = static_cast<float>(me->getHitPoints()) / 200.0;
	auto closest = me->getClosestUnit(IsEnemy && !IsBuilding);
	s[Param::ENEMY_DISTANCE] = (closest ? closest->getPosition().getDistance(me->getPosition()) : 64.0f) / 3000.0;
	closest = me->getClosestUnit(!IsEnemy && !IsBuilding);
	s[Param::TEAM_DISTANCE] = (closest ? closest->getPosition().getDistance(me->getPosition()) : 64.0f) / 3000.0;
	s[Param::ME_ATTACKED] = static_cast<float>(me->isUnderAttack());
	s[Param::ME_REPAIRED] = static_cast<float>(me->isBeingHealed());
	s[Param::TEAM_MINERALS] = Broodwar->self()->minerals() / 2000.0;
	s[Param::ME_SCV] = static_cast<float>(me->getType() == BWAPI::UnitTypes::Terran_SCV);
	s[Param::ME_MARINE] = static_cast<float>(me->getType() == BWAPI::UnitTypes::Terran_Marine);
	return s;
}

template <typename T>
BWAPI::Unit find_unit_in(BWAPI::Unitset us, T cb) {
	for (auto &x : us) {
		if (cb(x)) {
			return x;
		}
	}
	return nullptr;
}

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

template <typename T>
BWAPI::Unit find_max_friend(T cb) {
	return find_max(Broodwar->self()->getUnits(), cb);
}

template <typename T>
BWAPI::Unit find_max_enemy(T cb) {
	return find_max(Broodwar->enemy()->getUnits(), cb);
}

template <typename T>
BWAPI::Unit find_friend(T cb) {
	return find_unit_in(Broodwar->self()->getUnits(), cb);
}

template <typename T>
BWAPI::Unit find_enemy(T cb) {
	return find_unit_in(Broodwar->enemy()->getUnits(), cb);
}

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

static void commitAction(BuildAction a) {
	if (a == BuildAction::Type::BUILD_SCV) {
		filter(Broodwar->self()->getUnits(), BWAPI::Filter::GetType == BWAPI::UnitTypes::Terran_Command_Center, 
			[](BWAPI::Unit u){
				u->train(BWAPI::UnitTypes::Terran_SCV);
				return false;
				});
	}
	else if (a == BuildAction::Type::BUILD_MARINE) {
		filter(Broodwar->self()->getUnits(), BWAPI::Filter::GetType == BWAPI::UnitTypes::Terran_Barracks, 
			[](BWAPI::Unit u){
				if(!u->isTraining()) {
					u->train(BWAPI::UnitTypes::Terran_Marine);
					return false;
				}
				return true;
				});
	}
	else if (a == BuildAction::Type::BUILD_SUPPLY) {
		filter(Broodwar->self()->getUnits(), BWAPI::Filter::IsWorker, 
			[](BWAPI::Unit u){
				if(!u->isConstructing()) {
					auto target_loc = Broodwar->getBuildLocation(BWAPI::UnitTypes::Terran_Supply_Depot, u->getTilePosition());
					u->build(BWAPI::UnitTypes::Terran_Supply_Depot, target_loc);
					return false;
				}
				return true;
				});
	}
	else if (a == BuildAction::Type::BUILD_BARRACK) {
		filter(Broodwar->self()->getUnits(), BWAPI::Filter::IsWorker, 
			[](BWAPI::Unit u){
				if(!u->isConstructing()) {
					auto target_loc = Broodwar->getBuildLocation(BWAPI::UnitTypes::Terran_Barracks, u->getTilePosition());
					u->build(BWAPI::UnitTypes::Terran_Barracks, target_loc);
					return false;
				}
				return true;
				});
	}
}

static void commitAction(UnitAction a, Unit me, bool debug) {
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
		if (target) {
			me->attack(target);
			Broodwar->drawLineMap(me->getPosition(), target->getPosition(), Color(255, 0, 0));
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
		if (target) {
			me->repair(target);
			Broodwar->drawLineMap(me->getPosition(), target->getPosition(), Color(0, 255, 0));
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
		}
	}
	else if (a == UnitAction::Type::MINE) {
		if (debug)
			std::cout << "MINE QUEST IS TO MINE!!!" << std::endl;
		if(!me->isGatheringMinerals()) {
			if(me->isCarryingMinerals()) {
				me->returnCargo();
			}
			else {
				auto target = me->getClosestUnit(IsMineralField);
				if (target) {
					me->gather(target);
				}
			}
		}
	}

}


void ExampleAIModule::onStart()
{
	debug = false;
	char buf[255] = {};
	if (nullptr != getenv("DEBUG")) {
		debug = true;
	}
	// std::cout << "Loading model at " << get_dbname() << std::endl;
	if (!bh.load(get_dbname())) {
		Broodwar->sendText("My pants are on my head!");
	}
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
			if (!p->isObserver())
				Broodwar << p->getName() << ", playing as " << p->getRace() << std::endl;
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
	force_lose = false;
}

void ExampleAIModule::onEnd(bool isWinner)
{
	//char buf[MAX_PATH] = {};
	//if (0 != GetEnvironmentVariableA("resultfile", buf, sizeof(buf)))
	//{
	//	std::ofstream out(buf);
	//	out << (isWinner ? "1" : "0");
	//}
	bh.winner = (!force_lose) && isWinner;
	// std::cout << "THE END! I AM " << (isWinner ? "WINNER" : "LOOSER") << std::endl;
	bh.save(get_resname());
//   std::cout << "THEOTHERSIDE!" << std::endl;
}

static uint32_t frames = 0;
static uint32_t nexttick = 0;

void ExampleAIModule::onFrame()
{
	// Called once every game frame
	++frames;

	// Return if the game is a replay or is paused
	if (Broodwar->isReplay() || Broodwar->isPaused() || !Broodwar->self())
		return;

	// Prevent spamming by only running our onFrame once every number of latency frames.
	// Latency frames are the number of frames before commands are processed.
	if (Broodwar->getFrameCount() % Broodwar->getLatencyFrames() != 0)
		return;

	bool did_lose = true;

	if((frames & 0xf) == 0) {
		auto bs{ createBuildState() };
		auto baction{ bh.bmodel.get_action(bs) };
		commitAction(baction);
	}

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
			if((frames & 1) == 0)
			{
				auto s{ createUnitState(u) };
				auto uaction{ bh.umodel.get_action(s) };
				commitAction(uaction, u, debug);
			} // closure: if idle
		}
	} // closure: unit iterator

	if (did_lose) {
		Broodwar->leaveGame();
	}
	// double elapsed = difftime(time(NULL), start_time);
	// if (debug)
	// 	std::cout << "@ ELAPSED " << elapsed << std::endl;
	if (frames == 9900) {
		// Broodwar->sendText("timeout");
		force_lose = true;
	}
	else if(frames > 10000) {
		//std::cout << "@@@ Timeout!" << std::endl;
		//onEnd(false);
		Broodwar->leaveGame();
	}

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
