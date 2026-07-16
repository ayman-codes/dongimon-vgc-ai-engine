from vgc2.competition import Competitor

from src.battle.policy import DongimonBattlePolicy
from src.selection.policy import DongimonSelectionPolicy
from src.teambuild.policy import HesfTeamBuildPolicy


class DongimonCompetitor(Competitor):
    def __init__(self, name: str = "Dongimon", custom_weights: dict[str, float] | None = None):
        self.__name = name
        bp = DongimonBattlePolicy(custom_weights=custom_weights) if custom_weights else DongimonBattlePolicy()
        self.__battle_policy_instance = bp
        self.__selection_policy_instance = DongimonSelectionPolicy()
        self.__team_build_policy_instance = HesfTeamBuildPolicy()

    @property
    def name(self) -> str:
        return self.__name

    @property
    def battlepolicy(self):
        return self.__battle_policy_instance

    @property
    def selectionpolicy(self):
        return self.__selection_policy_instance

    @property
    def teambuildpolicy(self):
        return self.__team_build_policy_instance
