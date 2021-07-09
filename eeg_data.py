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

    def prepare_dataset(self, experiment, phases):
        events = self._extract_events(experiment)
        labels, data = self._extract_eeg(events, phases)
        return labels, data

    def _extract_events(self, experiments):
        # ["5R", "5M", "6R", "6M", "7R", "7M"]
        possible_experiments = set(f"{x}{y}" for x in [5, 6, 7] for y in ["R", "M"])
        assert set(experiments).issubset(possible_experiments)
        # iterating over dataset is a very primitive method, but the data is so dirty it
        # must be done this way
        trials = []
        trial = {}
        for _, row in self.events.iterrows():
            if row["trial_type"] == "start of the trial":
                trial = {"start": row["onset"]}
            elif row["trial_type"].startswith("encoding"):
                if not len(trial) == 1:
                    trial == {}
                    continue
                trial["encoding"] = row["onset"]
            elif row["trial_type"].startswith("delay"):
                if len(trial) != 2:
                    trial = {}
                    continue
                trial["delay"] = row["onset"]
            elif row["trial_type"].startswith("probe"):
                if len(trial) != 3:
                    trial = {}
                    continue
                trial["probe"] = row["onset"]
            elif row["trial_type"].startswith("Response"):
                if len(trial) != 4:
                    trial = {}
                    continue
                trial["end"] = row["onset"]
                trial["experiment"] = row["trial_type"][10:12]
                trial["label"] = 1 if "correct" in row["trial_type"] else 0
                assert trial["start"] < trial["encoding"]
                assert trial["encoding"] < trial["delay"]
                assert trial["delay"] < trial["probe"]
                assert trial["probe"] < trial["end"]
                if len(trial) == 7:
                    trials.append(trial)
                trial = {}

        trials = [x for x in trials if x["experiment"] in experiments]
        return trials

    def _extract_eeg(self, events, phases):
        bins = [
            [
                (x["start"], x["encoding"]),
                (x["encoding"], x["delay"]),
                (x["delay"], x["probe"]),
                (x["probe"], x["end"]),
            ]
            for x in events
        ]
        bins = [i for x in bins for i in x]  # flatten the list
        intervals = pd.IntervalIndex.from_tuples(bins)
        continuous_time = (np.arange(len(self.eeg)) + 1) / 500
        cuts = pd.cut(continuous_time, intervals)
        types = ["start", "encoding", "delay", "probe"]
        cuts.categories = [f"{t}_{i}" for i in range(len(events)) for t in types]
        # TODO trial is probably not the best name, as it is the same as for trial_type
        self.eeg["trial"] = cuts
        self.eeg["phase"] = self.eeg["trial"].map(lambda x: x.split("_")[0])
        self.eeg["trial"] = self.eeg["trial"].map(lambda x: x.split("_")[1])
        self.eeg["person"] = self.person

        labels = pd.DataFrame(
            {
                "label": [x["label"] for x in events],
                "trial": range(len(events)),
                "person": self.person,
            }
        )
        data = self.eeg[self.eeg["trial"].notna() & self.eeg["phase"].isin(phases)]
        trials = [int(i) for i in data["trial"].tolist()]
        data = data.drop("trial", axis=1)
        data["trial"] = trials
        return labels, data
