# Human-Activity-Classifier 2020
# Computational Data Science, CMPT 353, SFU

* Collected accelerometer data from phone sensor to classify between different human activities using different machine learning models.
* Activities like walking, jogging, running and climbing up or down stairs were classified with an accuracy of 80%.
* Used Pandas to manage and clean the data, applied a lowpass butterworth filter to remove noise and did feature engineering to produce velocity, displacement and frequency of steps.
* Used KNeighborClassifier, SVC and RandomForestClassifier from sklearn library to classify between different activities. Tried to determine velocity of walks using KNeighborRegressor.
