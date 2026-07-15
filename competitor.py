from vgc2.agent import BattlePolicy, SelectionPolicy, TeamBuildPolicy
from vgc2.competition import Competitor

# Policies
from src.battle.policy import DongimonBattlePolicy
from src.selection.policy import DongimonSelectionPolicy
from src.teambuild.policy import HesfTeamBuildPolicy


class DongimonCompetitor(Competitor):
    def __init__(self, name: str = "Dongimon", custom_weights: dict[str, float] | None = None):
        self.__name = name
        battle_policy = DongimonBattlePolicy(custom_weights=custom_weights) if custom_weights else DongimonBattlePolicy()
        self.__battle_policy_instance = battle_policy
        self.__selection_policy_instance = DongimonSelectionPolicy()
        self.__team_build_policy_instance = HesfTeamBuildPolicy()

    @property
    def name(self) -> str:
        return self.__name

    @property
    def battlepolicy(self) -> BattlePolicy | None:
        return self.__battle_policy_instance

    @property
    def selectionpolicy(self) -> SelectionPolicy | None:
        return self.__selection_policy_instance

    @property
    def teambuildpolicy(self) -> TeamBuildPolicy | None:
        return self.__team_build_policy_instance
