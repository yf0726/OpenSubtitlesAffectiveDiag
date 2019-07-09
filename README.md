# Affect-rich Dialogue Generation using OpenSubtitles 2018
Semester project in human interaction group (https://hci-test.epfl.ch/). Topic on dialogue generation.

## Goal of this project

First we extract multi-turn dialogues from OpenSubtitles 2018 (segmentation is based on sentence similarity), then we adopt the affect-rich approach and MMI objective function to improve the basic Seq2Seq model.

## Results

### Parsing the OpenSubtitles:
1. `OpenSubtitles 2018`: we clean and parse the original OpenSubtitles 2018, and save lines with timestamps in .txt file. Only few samples are uploaded in Github for huge size of whole data (For whole processed OpenSubtitles 2018 please download from: https://drive.google.com/open?id=1ZUJ2J8ukuXhMKXd0SVG1pCZ5ZnJZ9Csc). Not labelled with characters, scenes or dialogue boundaries.
2. `Scripts data set`: created from 985 scripts and save in './dataset/scripts/script_data_set.csv'. Well labelled with characters, dialogue boundaries and movie names.

### Dialogue segmentation
For dialogue segmentation part, we validate our method on Cornell Movie Dialog data set and test it on our own scripts data set, and finally we reach p_k 0.262 and 0.295 respectively.
The results are saved in Google drive: https://drive.google.com/open?id=1niLIVTBJkhxRAUdLk67uthcDwxA2Pj0M
In total we have more than 350,000 files in `OpenSubtitles 2018`, and it's hard to process all 350,000 files at once. So I splited the 350,000 files into seven parts, each part has 50,000 files. To segment every 50,000 files, it will need more than 7 hours.
And currently we have session_segmentation_0.txt (more than 4,000,000 extracted dialogue).


For affect-rich dialogue generation, since I did not train the model very well (small training set+few epochs) the predictions are not very perfect, but the effect of MMI+VAD embedding is still obvious. The predictions are saved in 'results/predictions.csv'.

## Structure of this repo

1. code: all codes are saved in this folder. In code/jupyternb there are notebooks showing the whole process as mentioned in the report; in code/py there are .py files for users to parse and segment OpenSubtitles data; and in code/affect-rich there are codes for affect rich dialogue generation.
2. datasets: in this folder examples of each data set are saved.
3. papers
4. results: samples of results of segmentation and dialogue generation.

For more details please check the report.
