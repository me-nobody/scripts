# -*- coding: utf-8 -*-
"""Machine_Learning_classifier_nuclei_GFP_status.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Cbd-iPtWx56rXTZ28FRH12blavhFUx8F

mouse is the only tumor where we can measure blood vessel of normal, tumor and infiltrating zone
The strategy for doing this
- StarDist to segment nuclei
- HistomicsTK for feature extraction
- SKlearn for Machine Learning
"""

# Commented out IPython magic to ensure Python compatibility.
import sys,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from skimage import io
import seaborn as sns

# normal_image_path = "/content/drive/MyDrive/100um_images/"

image_dict = {}
for i,image in enumerate(os.listdir(normal_image_path)):
  name ="name_"+str(i)
  image = io.imread(os.path.join(normal_image_path,image))
  image_dict[name] = image

"""## Machine Learning

earlier we had trained a adaBoost classifier on both tumor and normal images, now we are training a 1 class SVM for the normal brain and tumor nuclei will be outlier

these normal features have been taken from GFP negative images from Val Lab
"""

normal_features = pd.read_csv('/content/gfp_negative_nuclei_features_val_40X.csv')

normal_features.head()

normal_features.drop('Label',axis=1,inplace=True)

normal_features.head()

import seaborn as sns

normal_features.shape

plt.figure(figsize=(24,80))
try:
    for i, col in enumerate(normal_features.columns.to_list()):
        plt.subplot(20, 4, i + 1)
        plt.hist(normal_features[col], label=col,color='blue')
        plt.legend()
        plt.title(col)
        plt.tight_layout()
except Exception as e:
    print(col,e)

# Compute the correlation matrix
corr = normal_features.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

normal_features.isna().sum()[normal_features.isna().sum()>0]

normal_features = normal_features[normal_features.columns[~normal_features.isnull().any()]]

normal_features.shape

tumor_features = pd.read_csv("/content/gfp_positive_nuclei_features_val_40X.csv")

tumor_features.head()

tumor_features.drop('Label',inplace=True,axis=1)
tumor_features = tumor_features[normal_features.columns]

tumor_features.head()

"""We have to scale the dataset"""



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

combined_df = pd.concat([normal_features,tumor_features],axis=0)

normal_features.shape, tumor_features.shape,combined_df.shape

combined_df.head()

# set the targets: normal=0, tumor=1
y_normal = np.zeros(6485)
y_tumor = np.ones(1348)
combined_targets = np.concatenate([y_normal,y_tumor],axis=0)
combined_targets.shape

combined_targets = pd.DataFrame(combined_targets)
combined_targets.columns = ["target"]

combined_targets.value_counts()

combined_targets = combined_targets.astype(int)

final_train_df = combined_df.copy()

final_train_df['target'] = np.concatenate([y_normal,y_tumor],axis=0)

final_train_df.head(n=2)

final_train_df['target'].value_counts()

final_train_df.to_csv("final_GFP_train_df.csv",index=False)



# Create training and validation sets
X_train, X_test, train_targets, test_targets = train_test_split(
    combined_df, combined_targets, test_size=0.25, random_state=42)

combined_df.shape

# Impute and scale numeric columns
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

np.unique(train_targets,return_counts=True)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

X_train.shape

X_train.columns = normal_features.columns
X_test.columns = normal_features.columns

from sklearn.svm import OneClassSVM

clf = OneClassSVM(kernel="rbf",gamma='auto',nu=0.001).fit(normal_features)

predictions_normal = clf.predict(normal_features)

unique, frequency=np.unique(predictions_normal, return_counts=True)
count=np.asarray((unique, frequency))
count

predictions_tumor = clf.predict(tumor_features)

unique, frequency=np.unique(predictions_tumor, return_counts=True)
count=np.asarray((unique, frequency))
count

!pip install lazypredict --upgrade --quiet

from lazypredict.Supervised import LazyClassifier

# Running the Lazypredict library and fit multiple regression libraries
# for the same dataset
clf = LazyClassifier(verbose=1,ignore_warnings=False, custom_metric=None)

models,predictions = clf.fit(X_train, X_test, train_targets,test_targets)

# Calculate performance of all models on test dataset
model_dictionary = clf.provide_models(X_train,X_test,train_targets,test_targets)
models

"""We are trying to map the different clusters using tSNE and uMAP"""

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

# Impute and scale numeric columns
scaler = MinMaxScaler()

scaled_normal_features = scaler.fit_transform(normal_features)

scaled_normal_features = pd.DataFrame(scaled_normal_features)

scaled_normal_features.columns = normal_features.columns

scaled_normal_features.head()

"""NORMAL FEATURES"""

pca = PCA(n_components= 3)

pca_result = pca.fit_transform(scaled_normal_features.values)

pca_result.shape

normal_features_color = np.zeros(6485)

pca_normal_features = pd.DataFrame()

pca_normal_features['PCA-1'] = pca_result[:,0]
pca_normal_features['PCA-2'] = pca_result[:,1]
pca_normal_features['PCA-3'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(5,5));
sns.scatterplot(
    x="PCA-1", y="PCA-2",
    palette=sns.color_palette("hls", 10),
    data=pca_normal_features,
    legend="full",
    alpha=0.3
);

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_normal_features)

tsne_results = pd.DataFrame(tsne_results)
tsne_results.columns=["first_dim","second_dim"]

plt.figure(figsize=(5,5))
sns.scatterplot(
    x="first_dim", y="second_dim",hue=normal_features_color,
    palette=sns.color_palette("hls", 10),
    data= tsne_results,
    legend="full",
    alpha=0.3
)

"""NORMAL and TUMOR FEATURES"""

pca_result = pca.fit_transform(X_train.values)

pca_result.shape

pca_normal_features = pd.DataFrame()

pca_normal_features['PCA-1'] = pca_result[:,0]
pca_normal_features['PCA-2'] = pca_result[:,1]
pca_normal_features['PCA-3'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(5,5));
sns.scatterplot(
    x="PCA-1", y="PCA-2",
    palette=sns.color_palette("hls", 10),
    data=pca_normal_features,
    legend="full",
    alpha=0.3
);

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_normal_features)

tsne_results = pd.DataFrame(tsne_results)
tsne_results.columns=["first_dim","second_dim"]

tsne_results.shape

tsne_results.shape,X_train.shape,train_targets.shape

plt.figure(figsize=(5,5))
sns.scatterplot(
    x="first_dim", y="second_dim",hue=train_targets,
    palette=sns.color_palette("hls", 10),
    data= tsne_results,
    legend="full",
    alpha=0.3
)

