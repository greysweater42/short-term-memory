from typing import List, Dict

import pandas as pd

from .events import EVENTS
from src.phase import Phase, PHASES
from pydantic import ValidationError


class InvalidTrialDF(Exception):
    pass


class Trial:
    def __init__(self, person_id: int, eeg: pd.DataFrame, trial_df: pd.DataFrame, trial_id: int) -> None:
        self._check_is_trial_df_valid(trial_df)

        self.trial_df: pd.DataFrame = trial_df
        self.trial_id: int = trial_id
        self.eeg: pd.DataFrame = eeg
        self.person_id: int = person_id

    @staticmethod
    def _check_is_trial_df_valid(trial_df: pd.DataFrame) -> bool:
        """checks if a given raw trial dataframe has errors, e.g. wrong number or order of events"""
        if len(trial_df) != len(EVENTS):
            raise InvalidTrialDF
        for event_data, event_expected in zip(trial_df["trial_type"], EVENTS.values()):
            if not event_data.startswith(event_expected):
                raise InvalidTrialDF

    def extract_phases(self) -> None:
        rows = self._change_trial_df_to_dict_of_rows()
        num_letters = rows["baseline"]["trial_type"].iloc[0][-3]
        type_ = rows["baseline"]["trial_type"].iloc[0][-2]
        response_type = self.get_trial_response(rows["response"]["trial_type"].iloc[0])

        phases: List[Phase] = []
        for phase_name, (event_start, event_end) in PHASES.items():
            start = rows[event_start].index[0]
            end = rows[event_end].index[0]
            phase = Phase(
                name=phase_name,
                trial_id=self.trial_id,
                person_id=self.person_id,
                num_letters=num_letters,
                type_=type_,
                response_type=response_type,
                eeg=self.eeg[self.eeg.index.to_series().between(start, end, inclusive="left")],
            )
            phases.append(phase)
        self.phases = phases

    # TODO does this function return a dict of series or dataframes?
    def _change_trial_df_to_dict_of_rows(self) -> Dict[str, pd.Series]:
        """for convenience: it will be easier to access specific rows by their "name", not .startswith"""
        rows = {}
        for name, startswith in EVENTS.items():
            rows[name] = self.trial_df[self.trial_df["trial_type"].str.startswith(startswith)]
        return rows

    @staticmethod
    def get_trial_response(response_type: str) -> bool:
        """dataset has response types as "...correct..." or "...error..." -> this function binarizes them:
        correct -> True, error -> False"""
        if "correct" in response_type:
            return True
        if "error" in response_type:
            return False
        raise InvalidTrialDF

    @classmethod
    def extract_phases_from_args(
        cls, person_id: int, eeg: pd.DataFrame, trial_df: pd.DataFrame, trial_id: int
    ) -> List[Phase]:
        try:
            trial = cls(person_id=person_id, eeg=eeg, trial_df=trial_df, trial_id=trial_id)
            trial.extract_phases()
            return trial.phases
        except (ValidationError, InvalidTrialDF):
            # some trials have incorrect data, so we do not extract phases from them at all
            return []
