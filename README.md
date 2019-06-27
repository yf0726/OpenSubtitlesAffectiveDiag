# Affect-rich Dialogue Generation using OpenSubtitles 2018
Semester project in human interaction group (https://hci-test.epfl.ch/). Topic on dialogue generation.

## Goal of this project

First we extract multi-turn dialogues from OpenSubtitles 2018 (segmentation is based on sentence similarity), then we adopt the affect-rich approach and MMI objective function to improve the basic Seq2Seq model.

## Results
First we create our own corpus:
1. `OpenSubtitles 2018`: we clean and parse the original OpenSubtitles 2018, and save lines with timestamps in .txt file. Not uploaded in Github for its huge size. Not labelled with characters, scenes or dialogue boundaries.
2. `Scripts data set`: created from 985 scripts and save in './dataset/scripts/script_data_set.csv'. Well labelled with characters, dialogue boundaries and movie names.

For dialogue segmentation part, we validate our method on Cornell Movie Dialog data set and test it on our own scripts data set, and finally we reach $p_k$ 0.262 and 0.295 respectively.

For affect-rich dialogue generation, since I did not train the model very well (small training set+few epochs) the predictions are not very perfect, but the effect of MMI+VAD embedding is still obvious. The predictions are saved in 'results/predictions.csv'.

## Structure of this repo
