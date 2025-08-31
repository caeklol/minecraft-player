# minecraft-player

play any sound in vanilla Minecraft without texture packs

> [!NOTE]
> This project is incomplete

## usage
you can use `--help`, but if you like reading:
##### `-i, --input`
specifies input file. this should be in mono. automatically resampled to \
48kHz sampling rate, so it may be faster to do that beforehand
##### `-o, --output`
mcfunction files are directly saved here, named by index, starting by 0. \
each following sound is scheduled via `audio:_/{}`. you should structure your \
functions folder as such.

##### `--reconstruction`
optionally, you can create an audio reconstruction using this parameter. this saves \
under the WAV format, but `.wav` is not automatically appended to the filename.

##### `--local, -l` / `--refetch, -r`
this specifies whether to refetch from remote (mojang) or use locally saved assets. \
this can save a lot of time in dev

##### `--verbosity`
the only possible verbosity levels are: `problems-only`, `normal`, `debug` and `everything`

## methodology
#### NNLS (current)
this is what is currently being used. intitially it was per-column but it was too slow \
and not very accurate. using a global solution[^1] made it faster, but accuracy with \
greedy solutions was much better

#### Gradient descent
this is the next method i will write when i get back to this project. ideally, pitch and volume \
for each sample will be solved by writing some cost function that is computed when needed

[^1]: NNLS PGD as detailed [here](https://angms.science/doc/NMF/nnls_pgd.pdf)
