from collections import OrderedDict
from typing import List, Dict, Tuple


EVENTS: OrderedDict = OrderedDict(
    start="start of the trial",
    baseline="start of the baseline period",
    presentation="presentation",
    encoding="encoding",
    delay="delay",
    probe="probe",
    response="Response",
)

trial_events_names = list(EVENTS.keys())
phases_names = list(zip(trial_events_names[:-1], trial_events_names[1:]))
PHASES: Dict[str, Tuple[str, str]] = {name: phase for name, phase in zip(trial_events_names[:-1], phases_names)}

ELECTRODES = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
    "EOGv",
    "EOGh",
]


class DatasetInfo:
    """possible values for each of the dataset parameters"""

    # TODO this class can be confused with Dataset. it should be called SurveyInfo or MetaInfo, maybe
    experiment_types: List[int] = ["M", "R"]
    num_letters: List[int] = [5, 6, 7]
    response_types: List[str] = ["correct", "wrong"]
    phases: Dict[str, Tuple[str, str]] = PHASES
    electrodes: List[str] = ELECTRODES
    events: OrderedDict[str, str] = EVENTS

    @classmethod
    def get(cls, key: str):
        return getattr(cls, key)


# WAVES = {
#     "theta": {"min": 4, "mean": 6, "max": 8},
#     "alpha": {"min": 8, "mean": 10.5, "max": 13},
#     "beta": {"min": 16, "mean": 19, "max": 22},
#     "gamma": {"min": 35, "mean": 42.5, "max": 50},
# }
