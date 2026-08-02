from pathlib import Path
from typing import Any

from vgc2.competition import Competitor

from src.battle.greedy_dongi import GreedyDongiPolicy
from src.selection.policy import DongimonSelectionPolicy
from src.teambuild.policy import HesfTeamBuildPolicy


class DongimonCompetitor(Competitor):  # type: ignore[misc]
    def __init__(
        self,
        name: str = "Dongimon",
        custom_weights: dict[str, float] | None = None,
        selection_mode: str = "mp_only",
        mp_model_path: Path | None = None,
        n_top_candidates: int | None = None,
    ):
        self.__name = name
        bp = GreedyDongiPolicy()
        self.__battle_policy_instance = bp
        sel_kwargs: dict[str, Any] = {
            "selection_mode": selection_mode,
            "mp_model_path": mp_model_path,
        }
        if n_top_candidates is not None:
            sel_kwargs["n_top_candidates"] = n_top_candidates
        self.__selection_policy_instance = DongimonSelectionPolicy(**sel_kwargs)
        self.__team_build_policy_instance = HesfTeamBuildPolicy()

    @property
    def name(self) -> str:
        return self.__name

    @property
    def battlepolicy(self) -> Any:
        return self.__battle_policy_instance

    @property
    def selectionpolicy(self) -> Any:
        return self.__selection_policy_instance

    @property
    def teambuildpolicy(self) -> Any:
        return self.__team_build_policy_instance
