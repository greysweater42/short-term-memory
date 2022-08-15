from src.raw_person import RawPerson


def test_init():
    assert False


def test_load():
    # TODO fixture with example data
    raw_person = RawPerson(person_id=1)
    raw_person.load()
    assert len(raw_person.phases) == 672


def test_save():
    assert False
