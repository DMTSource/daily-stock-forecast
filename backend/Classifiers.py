
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, ExpSineSquared

from scipy.stats import randint as sp_randint
from scipy.stats import expon

# we random search on these clfs, can set defaults here
classifier_names = [
    "Nearest Neighbors", 
    "Linear SVM", 
    "RBF SVM", 
    "Gaussian Process",
    "Decision Tree", 
    "Random Forest", 
    #"Neural Net", 
    "AdaBoost",
    "Naive Bayes", 
    "QDA"
]
classifiers = {

    "Nearest Neighbors":KNeighborsClassifier(n_jobs=1),
    "Linear SVM":LinearSVC(),
    "RBF SVM":SVC(cache_size=10000),
    "Gaussian Process":GaussianProcessClassifier(n_jobs=1),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(n_jobs=1),
    #"Neural Net":MLPClassifier(alpha=1),
    "AdaBoost":AdaBoostClassifier(),
    "Naive Bayes":GaussianNB(),
    "QDA":QuadraticDiscriminantAnalysis()
}

ker_rbf     = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
#ker_rq      = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)
#ker_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
kernel_list = [ker_rbf]#ker_rq causes very long train time, ker_expsine cuses linalg error

param_dict = {
    "Nearest Neighbors":{"pca__n_components": [None],
                         "kneighborsclassifier__n_neighbors":sp_randint(2, 20),
                         "kneighborsclassifier__metric":["euclidean", "cityblock"],
                         "kneighborsclassifier__algorithm":["auto","ball_tree","kd_tree", "brute"],
                         "kneighborsclassifier__leaf_size":sp_randint(5, 50),
                         },
    "Linear SVM":{"pca__n_components": [None],
                  'linearsvc__C': expon(scale=10), 
                  'linearsvc__loss': ['squared_hinge'],
                  'linearsvc__tol': [1e-3,1e-4,1e-5,1e-6], 
                  'linearsvc__dual': [False, True], 
                  'linearsvc__class_weight':['balanced', None],
                  'linearsvc__max_iter':[100,500,1000,5000]
                  },
    "RBF SVM":{"pca__n_components": [None],
               'svc__C': expon(scale=10), 
               'svc__gamma': expon(scale=.1),
               'svc__kernel': ['rbf'], 
               'svc__class_weight':['balanced', None],
               'svc__max_iter':[-1,100,500,1000,5000]
              },
    "Gaussian Process":{"pca__n_components": [None],
                        "gaussianprocessclassifier__kernel": kernel_list,
                        "gaussianprocessclassifier__optimizer": ["fmin_l_bfgs_b"],
                        "gaussianprocessclassifier__n_restarts_optimizer": [0, 1, 2, 3],
                        "gaussianprocessclassifier__copy_X_train": [True],
                        "gaussianprocessclassifier__warm_start": [False, True],
                        "gaussianprocessclassifier__max_iter_predict": sp_randint(50, 200),
                        "gaussianprocessclassifier__random_state": [0]
                        },
    "Decision Tree":{"pca__n_components": [None],
                     'decisiontreeclassifier__min_samples_split': range(2, 403, 10)
                    },
    "Random Forest":{"pca__n_components": [None],
                     "randomforestclassifier__n_estimators": sp_randint(5, 50),
                     "randomforestclassifier__max_depth": sp_randint(2, 10),
                     "randomforestclassifier__max_features": ['auto', None, 'sqrt', 'log2'],
                     "randomforestclassifier__min_samples_split": sp_randint(2, 11),
                     "randomforestclassifier__min_samples_leaf": sp_randint(1, 11),
                     "randomforestclassifier__bootstrap": [True, False],
                     "randomforestclassifier__criterion": ["gini", "entropy"]
                     },
    "Neural Net":{"pca__n_components": [None],
                  "mlpclassifier__hidden_layer_sizes":sp_randint(10, 200),
                  "mlpclassifier__activation":["identity", "logistic", "tanh", "relu"],
                  "mlpclassifier__solver":['lbfgs','sgd','adam'],
                  "mlpclassifier__alpha":expon(scale=0.001),
                  "mlpclassifier__batch_size":[sp_randint(50, 200), 'auto'],
                  "mlpclassifier__learning_rate":['constant','invscaling','adaptive'],
                  "mlpclassifier__learning_rate_init": [1e-2,1e-3,1e-4,1e-5],
                  "mlpclassifier__max_iter":sp_randint(10, 1000),
                  "mlpclassifier__tol": [1e-3,1e-4,1e-5,1e-6]
                 },
    "AdaBoost":{"pca__n_components": [None],
                "adaboostclassifier__n_estimators":  sp_randint(25, 150),
                "adaboostclassifier__learning_rate": [1e-2,1e-3,1e-4,1e-5],
                "adaboostclassifier__algorithm": ["SAMME", "SAMME.R"]
               },
    "Naive Bayes":{"pca__n_components": [None]
                  },
    "QDA":{"pca__n_components": [None],
           "quadraticdiscriminantanalysis__tol":[1e-3,1e-4,1e-5,1e-6]
           }
}