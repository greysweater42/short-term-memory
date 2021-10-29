from oct2py import octave
from pathlib import Path

eeglab_path = Path("/home/tomek/doktorat/eeglab")

funcs = ["guifunc", "popfunc", "adminfunc", "sigprocfunc", "miscfunc"]
for func in funcs:
    path = eeglab_path / "functions" / func
    octave.addpath(str(path))

EEG = octave.pop_loadset(str(eeglab_path / "sample_data/eeglab_data_epochs_ica.set"))

EEG.data.shape

octave.addpath("/home/tomek/doktorat/amica/")
w, s, m = octave.runamica15(
    EEG.data[0],
    # numprocs=1,  # this tells that amica is run locally
    # max_threads=1,
    # num_models=1,
    # max_iter=1000,
    outdir="octave/fun.amica",
)

octave.addpath("/home/tomek/doktorat/code/learning")
a = octave.fn_arearect(x=10, y=12)
type(a)
a

octave.loadmodout15("/home/tomek/doktorat/code/tmpdata77443.fdt")
