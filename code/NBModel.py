import pandas as pd
import numpy as np
import joblib
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

#Uncomment Representation:
featureRep = "BoW"
#featureRep = "TF-IDF"

###Training model

#Loading vocab
vocabDataFile = pd.read_excel("data/BoW_0.1%Removed_filtered.xlsx")
trainDataFile = pd.read_excel("data/UsuableData_80%.xlsx")
testDataFile = pd.read_excel("data/UsuableData_20%.xlsx")

filteredVocab = vocabDataFile["Word"].dropna().astype(str).str.lower().unique().tolist()


#Drop 3-star review, map binary labels (1 = pos, 0 = neg)
def prepareBinaryData(dataFile):
    #Cleaning data file
    dataFile = dataFile[["Summary", "Text", "Score"]].copy() #take useful data
    dataFile = dataFile.dropna(subset = ["Text", "Score"])
    dataFile = dataFile[dataFile["Score"] != 3].copy() #remove 3 star reviews, not needed but a good safeguard

    #Augmenting data file
    dataFile["combinedText"] = dataFile["Summary"].fillna("").astype(str) + " " + dataFile["Summary"].fillna("").astype(str) + " " + dataFile["Text"].astype(str)
    dataFile["Sentiment"] = (dataFile["Score"] >= 4).astype(int)
    return dataFile


trainingData = prepareBinaryData(trainDataFile)
testingData = prepareBinaryData(testDataFile)

#Building BoW matrix using filtered vocab
vectorizer = CountVectorizer(vocabulary=filteredVocab) if featureRep == "BoW" else TfidfVectorizer(vocabulary=filteredVocab)
xTrain = vectorizer.transform(trainingData["combinedText"]) if featureRep == "BoW" else vectorizer.fit_transform(trainingData["combinedText"])
yTrain = trainingData["Sentiment"]

xTest = vectorizer.transform(testingData["combinedText"])
yTest = testingData["Sentiment"]


#Training multinomial naive bayes
nbModel = MultinomialNB(alpha = 1.0) #Laplace smoothing

startTime = time.time()
nbModel.fit(xTrain, yTrain)
trainTime = time.time() - startTime
print(f"Training completed in {trainTime:.2f} seconds")



###Testing model

#This creates a baseline
#Finds what is the most frequent outcome, then blindly guesses
#With our data, will guess positive all the time
majorityBaseline = DummyClassifier(strategy = "most_frequent")
majorityBaseline.fit(xTrain, yTrain)
baselinePred = majorityBaseline.predict(xTest)

print("--- Majority-Class Baseline (20% Test Set) ---")
print(f"Baseline Accuracy: {accuracy_score(yTest, baselinePred):.4f}")

#Testing NB model
yPred = nbModel.predict(xTest)


#Showing Results
print(f"\n--- Multinomial Naive Bayes ({featureRep}) Results (20% Test Set) ---")
print(f"Test Reviews Count: {xTest.shape[0]}")
print(f"Accuracy:  {accuracy_score(yTest, yPred):.4f}")
print(f"Precision: {precision_score(yTest, yPred):.4f}")
print(f"Recall:    {recall_score(yTest, yPred):.4f}")
print(f"F1-Score:  {f1_score(yTest, yPred):.4f}")

labels = [0, 1]
labelNames = ["Negative (0)", "Positive (1)"]
cm = confusion_matrix(yTest, yPred, labels = labels)
print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
print(pd.DataFrame(cm, index = labelNames, columns = labelNames))

print("\nDetailed Classification Report:")
print(classification_report(yTest, yPred, target_names = labelNames))



###Report Info

featureNames = np.array(vectorizer.get_feature_names_out()) #takes all words model trained on and puts them in an array
logOdds = nbModel.feature_log_prob_[1] - nbModel.feature_log_prob_[0] #finds how often each word appears in positive vs negative reviews, then takes the log of that ratio
#subtracts the log of the probability of a word appearing in a negative review from the log of the probability of it appearing in a positive review. This gives us the log-odds ratio for each word, which tells us how strongly each word is associated with positive or negative sentiment.

sortedIdx = np.argsort(logOdds) #sorts the log-odds ratios in ascending order
top10PosIdx = sortedIdx[-10:][::-1] #last 10 words in the array, most positive
top10NegIdx = sortedIdx[:10] #first 10 words in the array, most negative

print("\n--- Top-10 Most Informative Words ---")
print("Positive:", ", ".join(featureNames[top10PosIdx]))
print("Negative:", ", ".join(featureNames[top10NegIdx]))

#Error Analysis


testingData = testingData.reset_index(drop = True) #resets testing index to allign with predictions, since dropping 3 stars can leave gaps
testingData["Predicted"] = yPred
misclassified = testingData[testingData["Sentiment"] != testingData["Predicted"]] #get all incorrect predictions

print(f"\n--- Sample Misclassified Reviews ({len(misclassified)} total errors out of {len(testingData)}) ---")
for index, row in misclassified.head(5).iterrows(): #get 5 misclassified reviews
    #rename to words
    actualStr = "Positive" if row["Sentiment"] == 1 else "Negative" 
    predStr = "Positive" if row["Predicted"] == 1 else "Negative"
    #display results
    print(f"[Actual: {actualStr} ({int(row['Score'])} stars) | Predicted: {predStr}]")
    print(f"Review: {row['combinedText'][:250]}...\n") #truncate review to 250 characters for display


#Prevent NotFittedError in Streamlit
vectorizer.fit(trainingData["combinedText"])

#Export Model & Vectorizer for the Streamlit App
joblib.dump(nbModel, "models/nb_model.joblib")
joblib.dump(vectorizer, f"models/{featureRep.lower()}_vectorizer.joblib")
print("Saved model and vectorizer to models/ directory.")