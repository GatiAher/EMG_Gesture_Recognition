# EMG_Gesture_Recognition

> Gati Aher  
> SP2021: Neurotechnology: Brains and Machines

## Overview:

This project analyzes features extracted from integrated EMGs from a balanced dataset of three gestures (rock, paper, scissors) from a single subject who performed 6 trials of each of three gestures, each gesture taking a duration of 3 seconds. The raw signal was rectified and integrated before being saved to data files. 20 percent of the data was held out as a test set. Both the train and test data was normalized by channel, by subtracting off the train set's average value over all time points and trials, and dividing by the train set's standard deviation.

This project followed the topological data analysis steps laid out in [1], and used sklearn's topological data analysis tools with the Kepler Mapper and Ward's minimum variance method as the criterion for hierarchial clustering.

43 common features were extracted from the dataset. A comparison of feature redundancy was performed with topological data analysis. A comparison of feature class separability was analyzed by calculating Davies-Bouldin index(DBI), Fisher's linear discriminant index (FLDI), and measuring misclassification rates using 5-fold cross validation classification. Linear Discriminant Analysis and Support Vector Machine were employed as classifiers.

Correlation plots and dendrogram plots were also made to see the overlap between features.

Feature extraction was performed on segmented data, where windows were broken into lengths of 200 ms to the full 3000 ms, and overlapped by 100 ms. Classification was first performed on individual segements alone (approximating a real time system). This performed poorly (barely above 33% cross-validation accuracy). Papers performing real time classification generally have much more than 150 samples of data and collect at a higher frequency rate.

Classification on feature vectors where the segments' features were concatenated performed better (reaching up to 45% cross-validation accuracy classification for some features). However, overall performance on features extracted from the full 3000s performed the best. I am currently investigating whether there may be a bug in the windowed feature extraction.

See results folder for diagrams and visualizations.

## Discussion of Analysis

The dendrogram chart and topological data analysis both use hierarchial clustering to create subgroups. Both charts show that many of the frequency-domain features convey the same information. This is confirmed by correlation plots.

The real strength of topological data analysis appears when choosing a good node visualization function. Some papers [1, 2] use coloring to convery cluster density (k-nearest distance) and size to convey number of cluster members. Others use coloring to visualize how features capture variance in classes. This [talk by Gunner Carlsson](https://www.youtube.com/watch?v=8nUBqawu41k) has some very nice examples to that effect. I colored nodes by how they captured the mean value of each gesture class by channel, but there was no strong difference that I could pick up visually. This may be in part due to the limitations of coloring nodes in Kepler Mapper. There is no easy way to deviate from the default yellow-green-blue-purple gradients color map used by Kepler Mapper's visualization software, but I saved the graph information in a json and can use other programs to visualize a custom colored map.

I also colored nodes by feature classes [1] of signal amplitude and power, frequency information, non-linear complexity, time-series modeling, and unique. Depending on the hyperparameters, this frequently showed that topological maps clustered groups of the same feature class together. However, the topological analysis is muddied by multi-value features (autoregressive coefficients and their differencing version, cepstrum coefficients, sample entropy, histogram, interquartile range), as differences between the clustering pattern of say Histogrambin 1 and bin 2 are not readily interpretable. The sheer number of multi-value time series modeling features overshadow coloring patterns from other feature correlations. A good next step is using PCA to convert multi-feature values into a single feature as done by [2]. This will enable the creation of more interpretable topological maps.

## Takeaway Lessons

Topological data analysis is very cool, I am glad this project gave me an opportunity to use new tools and play around with interpretation. Some key lessons:

### On Designing Data Pipelines 

0. Makefiles are very helpful for making command shortcuts.
1. Understand the shape of data and do not preprocess out important information. For example, using preprocessing designed for raw EMG signals will eliminate the important curvature of an integrated EMG graph.
2. Be very careful about the sizes of matrixes that are being passes into functions. I spent a long time debugging an issue that was causing feature extraction to only be performed on the last 50 time points.
3. Remove constant features to avoid crypic machine learning / collinear errors.
4. Plot resulting features in order to identify problems. Right now, my sliding window code has might have a bug causing it to create an artificial slope in the the extracted features (this slope does not exist when extracting features from the full window).
5. Know what information will be available during testing and real evaluation, as this will inform the preprocessing. For example, if performing real time classification, the whole sample will not be available during classification, so normalization cannot mean-center data by overall mean value of the trial. A good practice is to mean center testing data based on values calculated from the train data.
6. Saving extracted features before performing analysis saves time. Decoupling the two allows for rapid visualization prototyping. 
7. Using a standard naming scheme to differentiate different feature extraction / normalization is a must. A good practice might be to use ID numbers to refer to each trial and maintaining a spreadsheet to match ID numbers to analysis.

### On Making Effective Data Visualizations

1. Use seaborn to make more complex data visualizations (with seaborn, heatmaps and dendrograms are just a few lines of code. Doing it in matplotlib is many many lines of custom code)
2. Leverage color and shape to add a dimension. Choose colors to be interpretable (especially relevant in TDA where sub-group nodes the the mean color of member nodes)
3. Identify when features should be summarized to allow for more interpretable analysis (relevant for comparing features that return multiple values with features that return singular values) and account for the bias that might appear when summarizing.
3. Make sure plots are appropriately sized and readable. Use interactive notebooks to quickly prototype graphs.

## Future Steps

I spent this project learning new tools and understanding how they can be leveraged for interesting data analysis. Over the summer, I will use insights gained in this project to practice making a flexible experimental data analysis pipeline with appropriate data visualizations.

Once I have the pipeline, I will analyze the heatmaps, dendrograms, topological maps, and other data visualization in depth, make a short list of promising feature sets, and test them out on the held out test data.

If I make my pipeline general enough, I can also test it on other data sets. This would allow me to practice comparing features across datasets and subjects.

## References:

- [1] Phinyomark Angkoon, Khushaba Rami N., Ibáñez-Marcelo Esther, Patania Alice, Scheme Erik and Petri Giovanni 2017 Navigating features: a topologically informed chart of electromyographic features spaceJ. R. Soc. Interface.142017073420170734 http://doi.org/10.1098/rsif.2017.0734
- [2] Côté-Allard, Ulysse et al. “Interpreting Deep Learning Features for Myoelectric Control: A Comparison With Handcrafted Features.” Frontiers in bioengineering and biotechnology vol. 8 158. 3 Mar. 2020, http://doi:10.3389/fbioe.2020.00158
    * code: https://github.com/UlysseCoteAllard/sEMG_handCraftedVsLearnedFeatures



