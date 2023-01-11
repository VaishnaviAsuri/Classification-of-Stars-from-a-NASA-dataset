# Classification-of-Stars-from-a-NASA-dataset
This project deals with Star type classification using Machine Learning on Python. NASA
has a Star and Exoplanet database that consists of tremendous amounts of raw data. This
project aims at classifying the data using various Machine learning techniques. The
classification of data creates pathways for predictive analysis of planetary systems based on
stellar features.
The project consists of two main parts. In the first part, we import the data and check for any
missing values or inaccurate data. We then perform class distribution to make sure the data is
balanced. This is followed by plotting various characteristics against each other to find the
right features to train our model. We do so by using data visualization techniques involving
libraries like Pandas, Numpy, Seaborn and Matplotlib.
The next part of the code is aimed at training our model. To do this, we prepare the data by
scaling it and splitting it into two parts- The Descriptive/Train set and the Target/Test set. We
pick up features from the train set that have significant differences in values to give a
maximum description of our training set. The machine learning model is then implemented
onto the train set, and the results are tested.
While the initial part of the code displays various machine learning techniques that can be
used to categorize data, the second half displays the scope of the trained set in building a
machine learning model. The potential of this project extends to comparative analysis of
various machine learning models. It also leaves scope for the prediction of Exoplanet
habitability based on unsupervised learning models. Overall, the project exponentially
reduces the human effort required to classify large datasets.


The project entails a dataset that consists of 250 stars that are to be classified into 6 different categories. This classification is done based on 6 features namely Temperature, Spectral Class, Color, Absolute Magnitude, Luminosity and Radius. The crux of the project is to find the relation between these features in such a way so that a Machine Learning Model can differentiate them into Red Dwarfs, Brown Dwarfs, White Dwarfs, Main Sequence stars, SuperGiants and Hyper Giants. We used Random forest classification method to train the model. An accuracy of 98 Percent was observed and more importantly, quality metrics of the model were calculated to find the origin and cause behind errors of the model. The method used for classification, albeit simple, holds many advantages over its contenders. Since the project is done on python, the code used is simplified due to an array of inbuilt funtions and libraries that enable easy execution. Basic Statistics such as correlation matrix, harmonic mean, etc are implemented using libraries such as matplotlib however a prior knowledge might be required to write the code efficiently.
