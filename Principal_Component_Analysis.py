#Note: All of the print statements in this code are commented. In order to run one of the print 
#statements, remove the "#" before it and then run the code.
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA

dataset = pd.read_csv('song_data.csv')

stddevs = np.empty(dataset.columns.size, dtype = float)
means = [dataset[column].mean() for column in dataset.columns]
np.set_printoptions(precision=3)

#Calculating the standard deviation of each feature
i = 0
for column in dataset.columns:
    for entry in dataset[column]:
        stddevs[i] += (entry - means[i])**2
    stddevs[i] = math.sqrt(stddevs[i]/100)
    i += 1

#Standardizing the data
std_dataset = dataset.copy()
i = 0
for column in dataset.columns:
    for j in range(dataset[column].size):
        std_dataset[column].iloc[j] = (dataset[column].iloc[j] - means[i]) / stddevs[i]
    i += 1

#Creating the covariance matrix and calculating eigenvalues
covar = std_dataset.cov().to_numpy()

#Remove the "#" in the next line to print the covariance matrix
#print(covar)

eigenvals, eigvecs = np.linalg.eig(covar)

#Remove the "#" in the next two line to print the eigenvalues
#print("Eigenvalues: ")
#print(eigenvals)

#Remove the "#" in the next two line to print the eigenvectors
#print("Eigenvectors: ")
#print(eigvecs)

#Remove the "#" in the next line to print the EVR for Feature 1
#print("EVR for Feature 1: " + str(round(eigenvals[0] / sum(eigenvals), 3)))

#Remove the "#" in the next line to print the EVR for Feature 2
#print("EVR for Feature 2: " + str(round(eigenvals[1] / sum(eigenvals), 3)))

#Remove the "#" in the next line to print the total information retention by selecting Features 1 and 2
#print("Total information retention: " + str(round((eigenvals[1] + eigenvals[0]) / sum(eigenvals), 3)))

featurenums = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11"]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(featurenums, eigenvals)
ax.set_ylabel('Magnitude of Eigenvalue')
ax.set_xlabel('Feature')
ax.set_title('Plot of Eigenvalues for Features')
ax.set_xticks(featurenums)
ax.set_yticks(np.arange(0, 7, 1))
#Remove the "#" in the next line to display the plot
#plt.show()


feature_matrix = np.vstack((eigvecs[:, 0], eigvecs[:, 1]))
tpdata = np.transpose(dataset)
finaldata = np.dot(feature_matrix, tpdata)

#Remove the "#" in the next two line to print the finaldata matrix  
#print("Final data matrix: ")                                               
#print(finaldata)
    

#Checking ratio with Python's PCA functions
x = dataset.loc[:, dataset.columns].values
x = sklearn.preprocessing.StandardScaler().fit_transform(x)
pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

#Remove the "#" in the next line to see the EVR for both selected components
#print("EVR for each component from Python's PCA model: " + str(pca.explained_variance_ratio_))