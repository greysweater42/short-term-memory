from pathlib import Path
import mne
import pandas as pd
import numpy as np


mne.set_config("MNE_LOGGING_LEVEL", "ERROR")
PATH = Path("/home/tomek/doktorat/ds003655-download/")


class EEGData:
    def __init__(self, person):
        self.eeg = None
        self.events = None
        self.person = int(person[-3:])
        person_path = PATH / person / "eeg"
        channels_file = f"{person}_task-VerbalWorkingMemory_channels.tsv"
        self.channels_path = person_path / channels_file
        self.data_path = person_path / f"{person}_task-VerbalWorkingMemory_eeg.set"
        events_file = f"{person}_task-VerbalWorkingMemory_events.tsv"
        self.events_path = person_path / events_file
        self._load_data()

    def _load_data(self):
        channels = pd.read_csv(self.channels_path, sep="\t")
        eog = channels.loc[channels.type == "EOG", "name"].tolist()
        data = mne.io.read_raw_eeglab(input_fname=self.data_path, eog=eog, preload=True)
        eeg = data.get_data()
        self.eeg = pd.DataFrame(eeg.transpose(), columns=channels["name"])
        self.events = pd.read_csv(self.events_path, sep="\t")

    @staticmethod
    def _filter_func(start, end):
        return lambda x: x.startswith(start) and x.endswith(end)

    def prepare_dataset(self, experiment):
        # ["5R", "5M", "6R", "6M", "7R", "7M"]
        assert experiment in [str(x) + y for x in [5, 6, 7] for y in ["R", "M"]]
        # iterating over dataset is a very primitive method, but the data is so dirty it
        # must be done that way
        trials = []
        trial = {}
        for _, row in self.events.iterrows():
            if row["trial_type"] == "start of the trial":
                trial = {"trial_start": row["onset"]}
            elif row["trial_type"].startswith("encoding"):
                if not trial:
                    continue
                trial["encoding_start"] = row["onset"]
            elif row["trial_type"].startswith("delay"):
                if len(trial) != 2:
                    trial = {}
                    continue
                trial["encoding_end"] = row["onset"]
                trial["experiment"] = row["trial_type"][14:16]
                assert trial["trial_start"] < trial["encoding_start"]
                assert trial["encoding_start"] < trial["encoding_end"]
                trials.append(trial)
                trial = {}

        responses = self.events.loc[
            self.events["trial_type"].map(lambda x: x.startswith("Response")),
            ["trial_type", "onset"],
        ]
        trials = [x for x in trials if len(x) == 4]
        for trial in trials:
            cond_min = responses["onset"] > 10 + trial["trial_start"]
            cond_max = responses["onset"] < 30 + trial["trial_start"]
            all_responses = responses.loc[cond_min & cond_max, ["trial_type", "onset"]]
            try:
                response = all_responses.iloc[0]  # the nearest response
            except IndexError:  # no response for a trial
                continue
            assert response["trial_type"][10:12] == trial["experiment"]
            trial["label"] = 1 if "correct" in response["trial_type"] else 0
            trial["response_time"] = response["onset"]

        trials = [x for x in trials if len(x) == 6]
        trials = [x for x in trials if x["experiment"] == experiment]
        bins = [(x["encoding_start"], x["encoding_end"]) for x in trials]
        intervals = pd.IntervalIndex.from_tuples(bins)
        continuous_time = (np.arange(len(self.eeg)) + 1) / 500
        cuts = pd.cut(continuous_time, intervals)
        cuts.categories = range(len(trials))
        # TODO trial is probably not the best name, as it is the same as for trial_type
        self.eeg["trial"] = cuts
        self.eeg["person"] = self.person

        labels = pd.DataFrame(
            {
                "label": [x["label"] for x in trials],
                "trial": range(len(trials)),
                "person": self.person,
            }
        )
        return labels, self.eeg[self.eeg["trial"].notna()]
