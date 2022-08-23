from src.raw_person import RawPerson
import pandas as pd
import numpy as np


def test_load():
    # TODO fixture with example data
    # raw_person = RawPerson(person_id=1)
    # raw_person.load()
    # assert len(raw_person.phases) == 672
    assert True


def test_save():
    # TODO
    assert True


def test_dataset():
    # TODO
    assert True


def test_apply_filters():
    # TODO
    raw_person = RawPerson(person_id=1)
    np.random.seed(0)
    raw_person.eeg = pd.DataFrame(dict(Fz=np.abs(np.random.rand(100))))
    raw_person._apply_filters()
