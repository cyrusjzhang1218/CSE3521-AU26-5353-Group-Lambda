import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import copy

featureRep = "TF-IDF"


#Load data
trainDataFile = pd.read_excel("data/UsuableData_80%.xlsx")
testDataFile = pd.read_excel("data/UsuableData_20%.xlsx")

def prepareBinaryData(dataFile):
    dataFile = dataFile[["Summary", "Text", "Score"]].copy()
    dataFile = dataFile.dropna(subset = ["Text", "Score"])
    dataFile = dataFile[dataFile["Score"] != 3].copy()
    dataFile["combinedText"] = dataFile["Summary"].fillna("").astype(str) + " " + dataFile["Summary"].fillna("").astype(str) + " " + dataFile["Text"].astype(str)
    dataFile["Sentiment"] = (dataFile["Score"] >= 4).astype(int)
    return dataFile

trainingData = prepareBinaryData(trainDataFile)
testingData = prepareBinaryData(testDataFile)

#Vectorize features
vectorizer = TfidfVectorizer(
    ngram_range = (1, 2), #captures 2-word phrases like "not good"
    max_features = 15000, #keeps top 10k most informative unigrams + bigrams
    sublinear_tf = True #scales word frequencies logarithmically
)
xTrain = vectorizer.fit_transform(trainingData["combinedText"])
yTrain = trainingData["Sentiment"]

xTest = vectorizer.transform(testingData["combinedText"])
yTest = testingData["Sentiment"]


###Training
xTrainSub, xVal, yTrainSub, yVal = train_test_split(xTrain, yTrain, test_size = 0.1, random_state = 42, stratify = yTrain)

plotModel = MLPClassifier(
    hidden_layer_sizes = (64,),
    activation = "relu",
    solver = "adam",
    alpha = 1e-3,
    batch_size = 32,
    learning_rate_init = 1e-3,
    random_state = 42
)

trainLosses, valLosses = [], []
for epoch in range(1, 7):  #epochs 1 through 6
    plotModel.partial_fit(xTrainSub, yTrainSub, classes = np.array([0, 1]))
    trainLosses.append(plotModel.loss_)
    valLosses.append(log_loss(yVal, plotModel.predict_proba(xVal)))
    if epoch == 2:
        epoch2Model = copy.deepcopy(plotModel)

predsEpoch2 = epoch2Model.predict(xTest)
predsEpoch6 = plotModel.predict(xTest)

print(f"Epoch 2 -> Acc: {accuracy_score(yTest, predsEpoch2):.4f} | F1: {f1_score(yTest, predsEpoch2):.4f}")
print(f"Epoch 6 -> Acc: {accuracy_score(yTest, predsEpoch6):.4f} | F1: {f1_score(yTest, predsEpoch6):.4f}")

###Graphing
"""
plt.figure(figsize=(8, 5))
plt.plot(range(1, 7), trainLosses, "b-o", label="Training Loss")
plt.plot(range(1, 7), valLosses, "r--s", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Log-Loss")
plt.title("FNN Training Loss vs. Validation Loss (TF-IDF)")
plt.legend()
plt.tight_layout()
plt.savefig("fnn_loss_curves.png", dpi=300)
"""