from pathlib import Path
from pydantic import BaseModel, validator
import pandas as pd
from .trial import EVENTS
from config import DATA_CACHE_PATH


trial_events_names = list(EVENTS.keys())
phases_names = list(zip(trial_events_names[:-1], trial_events_names[1:]))
PHASES = {name: phase for name, phase in zip(trial_events_names[:-1], phases_names)}


class Phase(BaseModel):
    # TODO validate num_letters, type_
    name: str
    trial_id: int
    person_id: int
    num_letters: int
    type_: str
    response_type: bool

    eeg: pd.DataFrame

    class Config:
        # pydantic does not allow exotic types like pd.DataFrame: it need to be allowed in this Config class
        arbitrary_types_allowed = True

    @validator("name")
    def check_name(cls, v):
        if v not in EVENTS:
            # TODO a custom error, so RawPerson could catch it easily
            raise ValueError(f"name should be one of {EVENTS}; {v} is none of these")
        return v

    @validator("num_letters")
    def check_num_letters(cls, v):
        if v not in [5, 6, 7]:
            raise ValueError(f"num_letters should be one of [5, 6, 7]; {v} is none of these")
        return v

    @property
    def path(self) -> Path:
        return (
            DATA_CACHE_PATH
            / str(self.person_id)
            / str(self.num_letters)
            / self.type_
            / str(int(self.response_type))
            / str(self.trial_id)
            / self.name
        ).with_suffix(".feather")

    @staticmethod
    def path_for_searching(
        _,
        person_id: int = None,
        num_letters: int = None,
        type_: str = None,
        response_type: bool = None,
        name: str = None,
    ):
        """used for finding trials which meet specific criterium, e.g. num_letters=6 and person="*" (all persons)"""
        # can't use cls because of built-in pydantic validation, which would not accept "*" for int fields; validation
        # is crucial for creating flawless Phases. this could be overriden by Union[int, str], but path_for_searching is
        # actually written for different purpose than creating an instance
        return (
            DATA_CACHE_PATH
            / str(person_id) if person_id else "*"
            / str(num_letters) if num_letters else "*"
            / type_ if type_ else "*"
            / str(int(response_type)) if response_type else "*"
            / "*"  # we always take all the trials, since their ID is artificial
            / name if name else "*"
        ).with_suffix(".feather")

    @classmethod
    def from_path(cls, path: Path):
        # TODO maybe data should be stored also for separate electrodes, as a list of observations, even without index:
        # the index is np.arange(0, len(x), 0.002) for all the cases anyway
        *_, person_id, num_letters, type_, response_type, trial_id, name = path.parts
        eeg = pd.read_feather(path)
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
        # TODO I could use some better format, maybe even mongo db with ODM< where every document would be searchable 
        # by all the parameters available in self.path
        self.eeg.reset_index().to_feather(self.path)
