import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

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

#Configure and Train FNN
fnnModel = MLPClassifier(
    hidden_layer_sizes = (64,), #single hidden layer with 64 neurons
    activation = "relu", #ReLU activation function
    solver = "adam", #Adam optimizer
    alpha = 1e-3, #L2 regularization to prevent overfitting
    batch_size = 32, #mini-batch size for gradient descent, 32 is best accuracy but takes longer to train
    learning_rate_init = 1e-3, #initial learning rate for Adam optimizer
    max_iter = 20, #max epochs, early stopping will usually halt sooner,
    early_stopping = True, #reverts to best model if validation score doesn't improve for n_iter_no_change epochs
    validation_fraction = 0.1, #automatically reserves 10% of xTrain as validation set
    n_iter_no_change = 3, #stop if validation score doesn't improve for 3 epochs
    random_state = 42, #seed to reproduce results
    verbose = True #prints epoch-by-epoch training loss and validation score
)

print(f"\n--- Training Feedforward Neural Network ({featureRep}) ---")
startTime = time.time()
fnnModel.fit(xTrain, yTrain)
trainTime = time.time() - startTime
print(f"Training completed in {trainTime:.2f} seconds ({fnnModel.n_iter_} epochs).")

###Testing model and report

#Testing FNN model
yPred = fnnModel.predict(xTest)

#Showing Results
print(f"\n--- FNN ({featureRep}) Results (20% Test Set) ---")
print(f"Training Time: {trainTime:.2f}s")
print(f"Accuracy: {accuracy_score(yTest, yPred):.4f}")
print(f"Precision: {precision_score(yTest, yPred):.4f}")
print(f"Recall: {recall_score(yTest, yPred):.4f}")
print(f"F1-Score: {f1_score(yTest, yPred):.4f}")

labels = [0, 1]
labelNames = ["Negative (0)", "Positive (1)"]
cm = confusion_matrix(yTest, yPred, labels = labels)
print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
print(pd.DataFrame(cm, index = labelNames, columns = labelNames))

print("\nDetailed Classification Report:")
print(classification_report(yTest, yPred, target_names = labelNames))

#Plotting Training Loss & Validation Score Curves
fig, ax1 = plt.subplots(figsize = (8, 5))

epochs = range(1, len(fnnModel.loss_curve_) + 1)
ax1.plot(epochs, fnnModel.loss_curve_, "b-o", label = "Training Loss (Log-Loss)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss", color = "b")
ax1.tick_params(axis = "y", labelcolor = "b")

#MLPClassifier tracks validation accuracy instead of validation loss
ax2 = ax1.twinx() #second y axis
ax2.plot(epochs, fnnModel.validation_scores_, "r--s", label = "Validation Accuracy")
ax2.set_ylabel("Validation Accuracy", color = "r")
ax2.tick_params(axis = "y", labelcolor = "r")

plt.title(f"FNN Training Loss & Validation Accuracy ({featureRep})")
fig.tight_layout()
plt.savefig("fnn_learning_curves.png", dpi = 300)
print("\nSaved learning curve plot to 'fnn_learning_curves.png'")



#Export Model & Vectorizer for the Streamlit App
export = "n"
if export.lower() == "y":
    joblib.dump(fnnModel, "models/fnn_model.joblib")
    joblib.dump(vectorizer, f"models/{featureRep.lower()}_vectorizer.joblib")
    print("Saved model and vectorizer to models/ directory.")

