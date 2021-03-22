"""
    Same caclulations as was used for bert embeddings, but using the baseline featureset.
    Author: Antonio Laverghetta Jr.
    alaverghett@usf.edu
"""

import pandas as pd

from scipy.stats import pearsonr, spearmanr, uniform
from scipy.spatial.distance import cosine
from os import listdir
from os.path import isfile, join
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS, TSNE, SpectralEmbedding

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from statistics import mean

class AoATrials():

    def __init__(self,wordbank=True):
      if wordbank:
        self.dataset = pd.read_csv("./Wordbank_psycholinguistic_features_classification.csv")
        self.dataset = self.dataset.sample(frac=1)
        self.dataset_embeddings = self.dataset[['noun','Nletters','frequency','synsets']].to_numpy()
        # scale features before running any algorithms
        scaler = StandardScaler()
        self.dataset_embeddings = scaler.fit_transform(self.dataset_embeddings)
      else:
        self.dataset = pd.read_csv("./kuperman_psycholinguistic_features.csv")
        self.dataset = self.dataset.sample(frac=1)
        self.dataset_embeddings = self.dataset[['noun','Nletters','frequency','synsets']].to_numpy()
        # scale features before running any algorithms
        scaler = StandardScaler()
        self.dataset_embeddings = scaler.fit_transform(self.dataset_embeddings)
    
    # classification trials for the wordbank dataset
    def run_classification_experiments(self):
        print("Logisitc Regression: ")
        y = self.dataset['labels']
        
        kf = KFold(n_splits=10)
        all_cors = []
        for train_index, test_index in kf.split(self.dataset_embeddings):
            X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
            y_train, y_test = y[train_index], y[test_index]
            log = LogisticRegression(n_jobs=-1,multi_class='multinomial',class_weight='balanced',max_iter=5000,C=1.0, penalty='l2', solver='sag')
            
            search = log.fit(X_train,y_train)

            y_pred = search.predict(X_test)
            print(pearsonr(y_test, y_pred))
            all_cors.append(pearsonr(y_test, y_pred)[0])
            print("\n")
        
        print(f"Mean:{mean(all_cors)}")


        print("Decision Tree")
        all_cors = []
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(self.dataset_embeddings):
            X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
            y_train, y_test = y[train_index], y[test_index]
            tree = DecisionTreeClassifier(class_weight='balanced',criterion='entropy', max_depth=15, max_features='log2', splitter='best')
            
            search = tree.fit(X_train,y_train)

            y_pred = search.predict(X_test)
            print(pearsonr(y_test, y_pred))
            all_cors.append(pearsonr(y_test, y_pred)[0])
            print("\n")
        
        print(f"Mean:{mean(all_cors)}")

        print("SVC")
        all_cors = []
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(self.dataset_embeddings):
            X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
            y_train, y_test = y[train_index], y[test_index]
            svc = SVC(class_weight='balanced',C=5.0,gamma='scale',kernel='rbf')
            
            search = svc.fit(X_train,y_train)

            y_pred = search.predict(X_test)
            print(pearsonr(y_test, y_pred))
            all_cors.append(pearsonr(y_test, y_pred)[0])
            print("\n")
        
        print(f"Mean:{mean(all_cors)}")

        print("KNN")
        all_cors = []
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(self.dataset_embeddings):
            X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
            y_train, y_test = y[train_index], y[test_index]
            knn = KNeighborsClassifier(n_jobs=-1,metric='manhattan',n_neighbors=15)
            
            search = knn.fit(X_train,y_train)
            y_pred = search.predict(X_test)
            print(pearsonr(y_test, y_pred))
            all_cors.append(pearsonr(y_test, y_pred)[0])
            print("\n")
        
        print(f"Mean:{mean(all_cors)}")

    
    def run_regression_experiments(self):
        print("Linear Regression:\n")
        self.LinearRegression()
        print("Decision Tree:\n")
        self.DecisionTreeRegression()
        print("Ridge Regression:\n")
        self.RidgeRegression()
        print("Nearest Neighbors:\n")
        self.NearestNeighborsRegression()
        print("Support Vector Machine:\n")
        self.SupportVectorRegression()
        print("SGD:\n")
        self.GradientDescent()
    
    # get and plot embedding visualizations
    @staticmethod
    def visualize(neighbors=5, method='isomap', lle_method='standard', perplexity=30.0, learning_rate=200.0, affinity='nearest_neighbors'):
        assert method in ['isomap','lle','mds','t-sne','spectral']
        print("Running visualization")
        dataset = pd.read_csv("./AoA.csv")
        scaler = StandardScaler()
        y = dataset['AoA'].to_numpy()
        word_embeddings = pd.read_csv("/content/Wordbank_psycholinguistic_features.csv")
        X = word_embeddings[['noun','Nletters','frequency','synsets']].to_numpy()
        X = scaler.fit_transform(X)
        if method == 'isomap':
          mds = Isomap(n_components=2, n_jobs=-1, n_neighbors=neighbors)
          X = mds.fit_transform(X)
          plt.scatter(X[:,0], X[:,1])
          plt.title("Isomap BERT Embedding visualization")
          plt.savefig("isomap_test.png")
          plt.show()
        
        elif method == 'lle':
          mds = LocallyLinearEmbedding(n_components=2, n_jobs=-1, n_neighbors=neighbors, method=lle_method)
          X = mds.fit_transform(X)
          plt.scatter(X[:,0], X[:,1])
          plt.title("LLE BERT Embedding visualization")
          plt.savefig("lle_test.png")
          plt.show()
        
        elif method == 'mds':
          mds = MDS(n_components=2, n_jobs=-1, n_init=100, max_iter=600)
          X = mds.fit_transform(X)
          plt.scatter(X[:,0], X[:,1])
          plt.title("MDS BERT Embedding visualization")
          plt.savefig("mds_test.png")
          plt.show()
        
        elif method == 't-sne':
          mds = TSNE(init='pca', method='exact', n_jobs=-1, perplexity=perplexity, n_iter=2000, learning_rate=learning_rate)
          X = mds.fit_transform(X)
          plt.scatter(X[:,0], X[:,1])
          plt.title("t-sne BERT Embedding visualization")
          plt.savefig("tsne_test.png")
          plt.show()
        
        elif method == 'spectral':
          mds = SpectralEmbedding(affinity=affinity, n_neighbors=neighbors, n_jobs=-1)
          X = mds.fit_transform(X)
          plt.scatter(X[:,0], X[:,1])
          plt.title("Spectral BERT Embedding visualization")
          plt.savefig("spectral_test.png")
          plt.show()
    
    def GradientDescent(self):
      y = self.dataset['labels'].to_numpy()
      kf = KFold(n_splits=10)
      for train_index, test_index in kf.split(self.dataset_embeddings):
        X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sgd = SGDRegressor(early_stopping=True,n_iter_no_change=10, penalty='l2', loss='huber',
                           learning_rate='optimal', eta0=0.1, alpha=0.01)
        
        search = sgd.fit(X_train,y_train)

        AoA_estimates = search.predict(X_test)
        print(pearsonr(AoA_estimates,y_test))
        print(spearmanr(AoA_estimates,y_test))
        print("\n")
    
    def NearestNeighborsRegression(self):
        kf = KFold(n_splits=10)
        y = self.dataset['labels'].to_numpy()
        
        for train_index, test_index in kf.split(self.dataset_embeddings):
          X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
          y_train, y_test = y[train_index], y[test_index]
          svr = KNeighborsRegressor(metric="manhattan", n_jobs=-1, algorithm='auto', n_neighbors=25)
          search = svr.fit(X_train,y_train)

          AoA_estimates = search.predict(X_test)
          print(pearsonr(AoA_estimates,y_test))
          print(spearmanr(AoA_estimates,y_test))
          print("\n")
    
    def RidgeRegression(self):
        kf = KFold(n_splits=10)
        y = self.dataset['labels'].to_numpy()

        for train_index, test_index in kf.split(self.dataset_embeddings):
            X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
            y_train, y_test = y[train_index], y[test_index]

            reg = Ridge().fit(X_train, y_train)
            AoA_estimates = reg.predict(X_test)
            print(pearsonr(AoA_estimates,y_test))
            print(spearmanr(AoA_estimates,y_test))
            print("\n")
        

    def LinearRegression(self):
        kf = KFold(n_splits=10)
        y = self.dataset['labels'].to_numpy()

        for train_index, test_index in kf.split(self.dataset_embeddings):
            X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
            y_train, y_test = y[train_index], y[test_index]

            reg = LinearRegression().fit(X_train, y_train)
            AoA_estimates = reg.predict(X_test)
            print(pearsonr(AoA_estimates,y_test))
            print(spearmanr(AoA_estimates,y_test))
            print("\n")

    def SupportVectorRegression(self):
        kf = KFold(n_splits=10)
        y = self.dataset['labels'].to_numpy()
        
        for train_index, test_index in kf.split(self.dataset_embeddings):
          X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
          y_train, y_test = y[train_index], y[test_index]
          svr = SVR(kernel='rbf',gamma='scale', C=3.769, epsilon=0.844)
          search = svr.fit(X_train,y_train)

          AoA_estimates = search.predict(X_test)
          print(pearsonr(AoA_estimates,y_test))
          print(spearmanr(AoA_estimates,y_test))
          print("\n")


    def DecisionTreeRegression(self):
        kf = KFold(n_splits=10)
        y = self.dataset['labels'].to_numpy()
        for train_index, test_index in kf.split(self.dataset_embeddings):
          X_train, X_test = self.dataset_embeddings[train_index], self.dataset_embeddings[test_index]
          y_train, y_test = y[train_index], y[test_index]
        
          tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, min_impurity_decrease=0.0)
          search = tree.fit(X_train,y_train)

          AoA_estimates = search.predict(X_test)
          print(pearsonr(AoA_estimates,y_test))
          print(spearmanr(AoA_estimates,y_test))
          print("\n")


if __name__ == "__main__":
  experiments = AoATrials(wordbank=True)
  print("Starting Wordbank trials")
  experiments.run_classification_experiments()

  # use this code to obtain the word embedding visualizations for wordbank
  # KupermanRegressionTrials.visualize()

  print("Starting Kuperman trials")
  experiments = AoATrials(wordbank=False)
  experiments.run_regression_experiments()