import pandas as pd


def get_trial_response(response_type: str) -> bool:
    """dataset has response types as "...correct..." or "...error..." -> this function binarizes them:
    correct -> True, error -> False"""
    if "correct" in response_type:
        return True
    elif "error" in response_type:
        return False
    else:
        return None


def check_is_raw_trial_valid(trial_df: pd.DataFrame) -> bool:
    """checks if a given raw trial dataframe has errors, e.g. wrong number or order of events"""
    if len(trial_df) != len(EVENTS):
        return False
    for event_data, event_expected in zip(trial_df["trial_type"], EVENTS.values()):
        if not event_data.startswith(event_expected):
            return False
    return True
