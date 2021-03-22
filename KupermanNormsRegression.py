"""
    Predict AoA on the kuperman norms using transformer embeddings
    Author: Antonio Laverghetta Jr.
    alaverghett@usf.edu
"""

import pandas as pd
import numpy as np
import matplotlib as mpl

from scipy.stats import pearsonr, spearmanr, uniform
from scipy.spatial.distance import cosine
from os import listdir
from os.path import isfile, join
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split, GridSearchCV
from sklearn.manifold import MDS
from GetEmbeedings import GetEmbeedings
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.dummy import DummyClassifier, DummyRegressor
from statistics import mean





def PrepareDataset(worbank=False):
    if worbank:
        kuperman = pd.read_csv("./Wordbank_psycholinguistic_features_classification.csv")
        kuperman = kuperman.sample(frac=1)
        print(len(kuperman))
        return kuperman
    else:

        kuperman = pd.read_csv("./kuperman_psycholinguistic_features.csv")
        
        # only necessary if all the rows from the kuperman norms have been kept
        # in the provided csv these transformations have already been made

        # kuperman.drop(['Freq_pm','Nletters','Nphon','Nsyll','AoA_Kup','Perc_known','Perc_known_lem','AoA_Bird_lem','AoA_Bristol_lem','AoA_Cort_lem','AoA_Schock','Word','Alternative.spelling'],axis=1,inplace=True)
        # kuperman.dropna(inplace=True)
        # kuperman.drop_duplicates(inplace=True,subset=['Lemma_highest_PoS'])
        # kuperman.rename({"Lemma_highest_PoS":"text","AoA_Kup_lem":"labels"},axis=1,inplace=True)


        kuperman = kuperman.sample(frac=1)

        print(len(kuperman))

        return kuperman


# define various models to test algorithm on
class AoATrials():

    def __init__(self, model, dim, roberta, wordbank):
        self.dataset = PrepareDataset(wordbank)
        _, self.dataset_embeddings = GetEmbeedings(model, model, self.dataset, word_bank_norms=wordbank, dim=dim, roberta=roberta)

        # sanity check to ensure that all words were embedded
        assert self.dataset_embeddings.shape == (len(self.dataset), dim)

    # all trials use optimal hyperparameters found using a grid search
    # this is for the Wordbank dataset
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
            # print(balanced_accuracy_score(y_test, y_pred))
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
            # print(balanced_accuracy_score(y_test, y_pred))
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
            # print(balanced_accuracy_score(y_test, y_pred))
            print(pearsonr(y_test, y_pred))
            all_cors.append(pearsonr(y_test, y_pred)[0])
            print("\n")
        
        print(f"Mean:{mean(all_cors)}")
        

    # this is for the kuperman dataset
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
    # this was used to generate the figure for the wordbank norms presented in the paper
    # we specifically used the isomap algorithm, which is the default
    @staticmethod
    def visualize(neighbors=5, method='isomap', lle_method='standard', perplexity=30.0, learning_rate=200.0, affinity='nearest_neighbors'):
        print("Running visualization")
        dataset = pd.read_csv("/home/antonio/from_source/bert-word-prediction/AoA.csv")
        word_embedding_mapping, word_embeddings = GetEmbeedings('bert-base-cased', dataset, word_bank_norms=True, bs=2048, dim=768)

        if method == 'mds':
            mds = MDS(n_components=2, n_jobs=1)
            X = mds.fit_transform(X)
            plt.scatter(X[:,0], X[:,1])
            plt.title("MDS BERT Embedding visualization")
            plt.savefig("mds_test.png")

        if method == 'isomap':
            mds = Isomap(n_components=2, n_jobs=-1, n_neighbors=neighbors)
            X = mds.fit_transform(X)
            df = pd.DataFrame({'x':X[:,0],'y':X[:,1],'c':dataset['category']})
            groups = df.groupby('c')
            for name, group in groups:
                plt.plot(group.x, group.y, marker='o', linestyle='', markersize=12, label=name)
                
                plt.title("Isomap BERT Embedding visualization")
                plt.legend(bbox_to_anchor=(1,1), loc="upper left")
                plt.savefig("isomap_test.png")
                plt.show()
        
        elif method == 'lle':
            mds = LocallyLinearEmbedding(n_components=2, n_jobs=-1, n_neighbors=neighbors, method=lle_method)
            X = mds.fit_transform(X)
            df = pd.DataFrame({'x':X[:,0],'y':X[:,1],'c':dataset['category']})
            groups = df.groupby('c')
            for name, group in groups:
                plt.plot(group.x, group.y, marker='o', linestyle='', markersize=12, label=name)
                plt.title("LLE BERT Embedding visualization")
                plt.legend(bbox_to_anchor=(1,1), loc="upper left")
                plt.savefig("lle_test.png")
                plt.show()
        
        elif method == 'mds':
            mds = MDS(n_components=2, n_jobs=-1, n_init=100, max_iter=600)
            X = mds.fit_transform(X)
            df = pd.DataFrame({'x':X[:,0],'y':X[:,1],'c':dataset['category']})
            groups = df.groupby('c')
            for name, group in groups:
                plt.plot(group.x, group.y, marker='o', linestyle='', markersize=12, label=name)
                plt.title("MDS BERT Embedding visualization")
                plt.legend(bbox_to_anchor=(1,1), loc="upper left")
                plt.savefig("mds_test.png")
                plt.show()
        
        elif method == 't-sne':
            mds = TSNE(init='pca', method='exact', n_jobs=-1, perplexity=perplexity, n_iter=2000, learning_rate=learning_rate)
            X = mds.fit_transform(X)
            df = pd.DataFrame({'x':X[:,0],'y':X[:,1],'c':dataset['category']})
            groups = df.groupby('c')
            for name, group in groups:
                plt.plot(group.x, group.y, marker='o', linestyle='', markersize=12, label=name)
                plt.title("t-sne BERT Embedding visualization")
                plt.legend(bbox_to_anchor=(1,1), loc="upper left")
                plt.savefig("tsne_test.png")
                plt.show()
        
        elif method == 'spectral':
            mds = SpectralEmbedding(affinity=affinity, n_neighbors=neighbors, n_jobs=-1)
            X = mds.fit_transform(X)
            df = pd.DataFrame({'x':X[:,0],'y':X[:,1],'c':dataset['category']})
            groups = df.groupby('c')
            for name, group in groups:
                plt.plot(group.x, group.y, marker='o', linestyle='', markersize=12, label=name)
                plt.title("Spectral BERT Embedding visualization")
                plt.legend(bbox_to_anchor=(1,1), loc="upper left")
                plt.savefig("spectral_test.png")
                plt.show()
        
    
    # these functions are all for the kupeprman regression trials
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



# get results for the random baseline
def RandomBaseline(trial='wordbank'):
    all_cors = []
    insignificant_ps = 0
    kf = KFold(n_splits=10)
    if trial == 'wordbank':
        kuperman = pd.read_csv("./Wordbank_psycholinguistic_features_classification.csv")
        X = kuperman[['noun','Nletters','frequency','synsets']].to_numpy()
        y = kuperman[['labels']].to_numpy()
        kuperman = kuperman.sample(frac=1)

        for train_index, test_index in kf.split(X,y):
            dummy = DummyClassifier(strategy='uniform')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            dummy.fit(X_train, y_train)
            y_pred = dummy.predict(X_test)
            print(matthews_corrcoef(y_test, y_pred))
            all_cors.append(matthews_corrcoef(y_test, y_pred))
            print("\n")
        
        print(f"Mean:{mean(all_cors)}")

    else:
        kf = KFold(n_splits=10)
        kuperman = pd.read_csv("./kuperman_psycholinguistic_features.csv")
        X = kuperman[['text','noun','Nletters','frequency','synsets']].to_numpy()
        y = kuperman[['labels']].to_numpy()
        kuperman = kuperman.sample(frac=1)

        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_pred = 20 * np.random.random_sample((len(X_test),1)) + 1
            y_pred = y_pred.flatten()
            y_test = y_test.flatten()
            print(pearsonr(y_test, y_pred))
            all_cors.append(pearsonr(y_test, y_pred)[0])
            print("\n")
        
        print(f"Mean:{mean(all_cors)}")
        print(insignificant_ps)





if __name__ == "__main__":
    # for getting the results using bert embeddings
    print("Beginning WordBank Trials")
    for i in [('bert-base-cased', 768, False), ('bert-large-cased', 1024, False), ('roberta-base', 768, True), ('roberta-large', 1024, True)]:
        print(f"Starting {i[0]}")
        experiments = AoATrials(i[0], i[1], roberta=i[2], wordbank=True)
        experiments.run_classification_experiments()
    
    # RANDOM BASELINE
    RandomBaseline('wordbank')

    print("Beginning Kuperman Trials")
    for i in [('bert-base-cased', 768, False), ('bert-large-cased', 1024, False), ('roberta-base', 768, True), ('roberta-large', 1024, True)]:
        print(f"Starting {i[0]}")
        experiments = AoATrials(i[0], i[1], roberta=i[2], wordbank=False)
        experiments.run_regression_experimentss()

    RandomBaseline('kuperman')
    
    # use this code to obtain the word embedding visualizations for wordbank
    # KupermanRegressionTrials.visualize()