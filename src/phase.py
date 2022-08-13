from pathlib import Path
from pydantic import BaseModel
import pandas as pd
from .trial import EVENTS


trial_events_names = list(EVENTS.keys())
phases = list(zip(trial_events_names[:-1], trial_events_names[1:]))
PHASES = {name: phase for name, phase in zip(trial_events_names[:-1], phases)}


class Phase(BaseModel):
    # TODO validate if name in EVENTS
    # TODO validate num_letters, type_
    name: str
    trial_id: int
    person_id: int
    num_letters: int
    type_: str
    response_type: bool

    eeg: pd.DataFrame

    @property
    def path(self) -> Path:
        return (
            Path()
            / str(self.person_id)
            / str(self.num_letters)
            / self.type_
            / str(int(self.response_type))
            / str(self.trial_id)
            / self.name
        )

    @staticmethod
    def path_for_searching(_, num_letters: int, type_: str, response_type: bool, name: str, person: int = None):
        """used for finding trials which meet specific criterium, e.g. num_letters=6 and person="*" (all persons)"""
        # can't use cls because of built-in pydantic validation, which would not accept "*" for int fields; validation
        # is crucial for creating flawless Trials; "*" near the end - always search for all trials
        return Path() / str(person if person else "*") / str(num_letters) / type_ / str(int(response_type)) / "*" / name

    @classmethod
    def from_path(cls, path: Path):
        # TODO maybe data should be stored also for separate electrodes
        *_, person_id, num_letters, type_, response_type, trial_id, name = path.parts
        eeg = pd.read_csv(path)
        return cls(
            trial_id=trial_id,
            person_id=person_id,
            num_letters=num_letters,
            type_=type_,
            response_type=response_type,
            name=name,
            eeg=eeg,
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # TODO I could use some better format, maybe even mongo db with ODM
        self.eeg.to_csv(self.path)
