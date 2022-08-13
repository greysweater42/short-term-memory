from src.raw_person import RawPerson


def test_load():
    eeg_raw_dataset = RawPerson(person_id=1)
    eeg_raw_dataset.load()
