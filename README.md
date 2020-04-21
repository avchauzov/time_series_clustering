# FMCG Selling Volumes Trajectories Clustering
Our goal:
 - to build the model that can efficiently cluster trajectories

## Project plan
1. Data processing and initial cleaning
2. Clustering algorithm

## Data processing
### 1.dataReading.ipynb

## Main algorithm
### 2.2.clusteringWeekly.py
File contains several steps:
1. We work on duplicated (union by '&') and really bad columns (that have only single value)
2. Exponential smoothing algorithm (our data is very sparse)
3. Time Series to the equal size conversion
4. Building a distance matrix based on Spearman correlation metric
5. An efficient step-by-step clustering algorithm
6. Plotting

As a result, we get:
 - *.csv file with column names union in groups for the same cluster (results.csv)
 - 3 pictures for each cluster (that contain more than 1 column): smoothed and interpolated TS, smoothed TS, original data
 - several outputs for different hyperparameters

### 2.3.clusteringDaily.py
Same for daily data.

### 3.1.hyperParametersSelection.ipynb
Visual outputs for optimal hyperparameters selection.