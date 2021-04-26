# EMG_Gesture_Recognition

> Gati Aher  
> SP2021: Neurotechnology: Brains and Machines

This notebook performs generalized human gesture classification on a balanced dataset of three gestures (rock, paper, scissors) from 10 subjects. Each subject performed 18 trials of duration 3 seconds. Fouier transform was used to select a filter frequency, a fast and simple feature set (mean absolute value, root mean squared, slope sign change, waveform length, Hjorth parameters) was extracted from overlapping sliding windows of 200 ms with stride length of 4 ms over 3 channels of sEMG data. The resulting feature vectors were fed into a neural net with one hidden layer which achieved 80% test set classification accuracy, 53% validation set classification accuracy.

[Please read notebook here](./EMG_Gesture_Recognition.html)

