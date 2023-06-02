# Bridging the Gap: Gaze Events as Interpretable Concepts to Explain Deep Neural Sequence Models


Before running any experiments, please adjust the file source/config/basepaths.py to your own paths.

To run the experiments please first write the datasets into npy files and detect all events with the following command:

```
python -m datasets.write_gazebase
python -m datasets.write_judo
python -m datasets.write_potec
```


To train the models on the datasets run the following commands:

```
python -m train --model=eky2 --data=gazebase
python -m train --model=eky2 --data=judo
python -m train --model=eky2 --data=potec
```


To evaluate concept influences for fixations and saccades run:

```
python -m evaluate.segmentations --model=eky2 --data=gazebase --metric=etra23 --segmentation=etra23
python -m evaluate.segmentations --model=eky2 --data=judo1000 --metric=etra23 --segmentation=etra23
python -m evaluate.segmentations --model=eky2 --data=potec --metric=etra23 --segmentation=etra23
```


To evaluate concept influneces for saccade sub-events run:

```
python -m evaluate.segmentations --model=eky2 --data=gazebase --metric=etra23 --segmentation=engbert.saccade_dissection_vpeak80
python -m evaluate.segmentations --model=eky2 --data=judo1000 --metric=etra23 --segmentation=engbert.saccade_dissection_vpeak80
python -m evaluate.segmentations --model=eky2 --data=potec --metric=etra23 --segmentation=engbert.saccade_dissection_vpeak80
```


To evaluate concept influneces for saccade sub-events run:

```
python -m evaluate.segmentations --model=eky2 --data=gazebase --metric=etra23 --segmentation=engbert.saccade_binning_duration_n100
python -m evaluate.segmentations --model=eky2 --data=judo1000 --metric=etra23 --segmentation=engbert.saccade_binning_amplitude_n100
python -m evaluate.segmentations --model=eky2 --data=gazebase --metric=etra23 --segmentation=ivt.fixation_binning_dispersion_n100
python -m evaluate.segmentations --model=eky2 --data=judo1000 --metric=etra23 --segmentation=ivt.fixation_binning_v_std_n100
```
