

import sys
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Drop 3-star review, map binary labels (1 = pos, 0 = neg)
# Copied from NBModel.py due to no main execution point
def prepare_data(data_file):
	#Cleaning data file
	data_file = data_file[["Summary", "Text", "Score"]].copy() #take useful data
	data_file = data_file.dropna(subset = ["Text", "Score"])
	data_file = data_file[data_file["Score"] != 3].copy() #remove 3 star reviews, not needed but a good safeguard

	#Augmenting data file
	data_file["CombinedText"] = data_file["Summary"].fillna("").astype(str) + " " + data_file["Text"].astype(str) #combine summary and text
	data_file["Sentiment"] = (data_file["Score"] >= 4).astype(int)
	return data_file

# Converts training and test text into BoW or TF-IDF number matrices using the shared vocabulary
def build_features(train_data, test_data, vocab, feature_rep):
	if feature_rep == "BoW": vectorizer = CountVectorizer(vocabulary=vocab)
	else: vectorizer = TfidfVectorizer(vocabulary=vocab)
	
	train_features = vectorizer.fit_transform(train_data["CombinedText"])
	test_features = vectorizer.transform(test_data["CombinedText"])
	return vectorizer, train_features, test_features

def main():
	# Read argument to decide feature representation
	# 1 for BoW and 2 for TF-IDF
	if len(sys.argv)<2 or sys.argv[1] not in ["1", "2"]:
		print("Give 1 for BoW or 2 for TF-IDF")
		sys.exit(1)
	feature_rep = "BoW" if sys.argv[1] == "1" else "TF-IDF"
	
	# Loading vocab
	vocab_file = pd.read_excel("data/BoW_0.1%Removed_filtered.xlsx")
	train_file = pd.read_excel("data/UsuableData_80%.xlsx")
	test_file = pd.read_excel("data/UsuableData_20%.xlsx")
	
	# Cleaning and preparing the data
	vocab = vocab_file["Word"].dropna().astype(str).str.lower().unique().tolist()
	train_data = prepare_data(train_file)
	test_data = prepare_data(test_file)
	
	# Build features and train the model
	vectorizer, train_features, test_features = build_features(train_data, test_data, vocab, feature_rep)

	#Hyperparameters Tuning
	param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
	base_model = LogisticRegression(max_iter=1000)
	grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring="f1", n_jobs=-1)
	grid_search.fit(train_features,train_data["Sentiment"])
	print("\nBest Hyperparameters:", grid_search.best_params_)

	# Timing a single training model run using the winning hyperparameters
	model = LogisticRegression(**grid_search.best_params_, max_iter=1000)
	time_passed = time.perf_counter()
	model.fit(train_features, train_data["Sentiment"])
	time_passed = time.perf_counter() - time_passed
	print(f"Training Time: {round(time_passed, 4)} seconds")

	# Predict sentiment on the unseen 20% test data
	predictions = model.predict(test_features)
	print(f"\n--- {feature_rep} Logistic Regression Results ---")
	print("Accuracy:", round(accuracy_score(test_data["Sentiment"], predictions), 4))
	print(classification_report(test_data["Sentiment"], predictions, target_names=["Negative (0)", "Positive (1)"]))
	print("Confusion Matrix:")
	print(confusion_matrix(test_data["Sentiment"], predictions))

if __name__ == "__main__":
	main()
