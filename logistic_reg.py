import numpy as np
import pandas as pd
import time

#Prétraitement MNIST

train_features = pd.read_csv('train.csv').to_numpy()[:,:1568]
train_labels = pd.read_csv('train_result.csv').to_numpy()[:,-1]
test_features = pd.read_csv('test.csv').to_numpy()[:,:1568]
n_classes = len(np.unique(train_labels))
print("Dimensions du jeu de données original:")
print("train_features: ", train_features.shape)
print("train_labels: ", train_labels.shape)
print("test_features: ",test_features.shape)
print("Les labels vont de ", np.min(train_labels), " à ", np.max(train_labels))

#Permuter les deux chiffres pour chaque exemple et rajouter au dataset
b = train_features.reshape(50000,28,56)
train_features2 = np.concatenate((b[:,:,28:], b[:,:,:28]),axis=2).reshape(50000,1568)
train_features = np.vstack((train_features,train_features2))
train_labels = np.hstack((train_labels,train_labels))

#Mélanger aléatoirement le jeu de données
inds = np.arange(train_features.shape[0])
np.random.shuffle(inds)
train_features = train_features[inds]
train_labels = train_labels[inds]

x_train=train_features[:90000,:]
y_train=train_labels[:90000]
x_val=train_features[90000:,:]
y_val=train_labels[90000:]

#Normaliser pour les colonnes qui ne sont pas juste des zéros:
mu = x_train.mean(axis=0)
sigma = x_train.std(axis=0)
x_train = x_train - mu
x_train = np.divide(x_train, sigma, out=np.zeros_like(x_train), where = sigma != 0)
x_val = x_val - mu
x_val = np.divide(x_val, sigma, out=np.zeros_like(x_val), where = sigma != 0)
test_features = test_features - mu
test_features = np.divide(test_features, sigma, out=np.zeros_like(test_features), where = sigma != 0)

#Rajouter une colonne de "1"
x_train_1 = np.hstack((x_train,np.ones((x_train.shape[0],1))))
x_val_1 = np.hstack((x_val,np.ones((x_val.shape[0],1))))
train_features_1= np.hstack((train_features,np.ones((train_features.shape[0],1))))
test_features_1 = np.hstack((test_features,np.ones((test_features.shape[0],1))))


# CLASSIFIEUR DE RÉGRESSION LOGISTIQUE MULTINOMIALE (avec mini-batch)

class RegLogistiqueMiniBatch:
    def __init__(self, w):
        self.w = w.astype('float128')  # Matrice (n_classes x (features+1))
        self.n_classes = self.w.shape[0]
        self.losses = np.array([]).astype('float128')
        self.accuracies = np.array([]).astype('float128')
        self.train_data = np.array([]).astype('float128')
        self.test_data = np.array([]).astype('float128')

    def predict(self, X):  # Retourne l'entier correspondant à la classe
        # Comme on prend l'argmax, on n'a pas besoin d'utiliser exp ou de normaliser
        return np.argmax(np.dot(self.w, np.transpose(X)), axis=0).astype('float128')

    def test_accuracy(self, X, y):
        # Taux de réussite
        accuracy = (y == self.predict(X)).mean()
        preds = self.predict(X)
        return accuracy

    def loss(self, X, y):
        # Pour chaque prédiction (sur l'ensemble de test), on calcule le coût, et on prend la moyenne
        # Astuce pour éviter les trop gros exp: soustraire maxX à l'intérieur du EXP, qui s'annule lors de la division
        maxX = np.max(X)
        expWX = np.exp(np.dot(self.w, np.transpose(X)) - maxX)
        probs = expWX / expWX.sum(axis=0)
        prob_correct = probs[y, np.arange(y.shape[0])]
        loss = np.nan_to_num(-np.log(prob_correct, out=-np.zeros_like(prob_correct),
                                     where=prob_correct != 0)).mean() + 0.5 * self.reg * (self.w ** 2).sum()
        return loss

    def gradient(self, X, y, batchsize, stepsize, epoch):

        n_batches = np.ceil(X.shape[0] / batchsize).astype('int')

        for i in range(n_batches):
            # Découper X et y selon la taille de la batch
            mini_X = X[i * batchsize:(i + 1) * batchsize]
            mini_y = y[i * batchsize:(i + 1) * batchsize]

            # matrice (n_classes x n_exemples) avec 0 ou 1
            y_onehot = np.zeros((self.n_classes, mini_y.shape[0]))
            y_onehot[mini_y, np.arange(mini_y.shape[0])] = 1

            # Astuce pour que le exp ne soit pas trop gros: soustraire valeur MAX,
            # Revient au même car ça s'annule au numérateur/dénominateur lors de la division
            maxX = np.max(mini_X)
            expWX = np.exp(np.dot(self.w, np.transpose(mini_X)) - maxX)

            # Le gradient peut devenir nan lorsqu'il y a overflow et division
            gradient = np.nan_to_num(np.dot(expWX / expWX.sum(axis=0) - y_onehot, mini_X) + self.reg * self.w)

            # Mettre à jour la matrice des poids W
            self.w = self.w - stepsize * gradient

    # Epoch = 1 passage au travers de tout le dataset
    def train(self, x_train, y_train, batchsize=50, stepsize=0.1, n_epochs=1, reg=.1):

        start = time.time()
        self.reg = reg
        self.train_data = x_train
        y = y_train.astype('int')

        for i in range(n_epochs):
            self.gradient(x_train, y, batchsize, stepsize, i)

        end = time.time()
        self.train_duration = end - start
        return self.train_duration


