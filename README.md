# AI-Powered Sentiment Analysis of Amazon Food Reviews

**CSE 3521 — Introduction to Artificial Intelligence I**  
**Autumn 2026 | Group Lambda**

## Project Overview

This project uses machine learning to classify customer sentiment in Amazon food reviews. The system analyzes written customer feedback and classify reviews as either positive or negative through the use of three model families (Naive Bayes, Logistic Regression, and a Feedforward Neural Network) on two text representations (Bag of Words and TF-IDF), serving the live predictions on a Streamlit wep app.

The goal is to help e-commerce vendors and product managers understand customer satisfaction, identify negative feedback, and prioritize reviews for further investigation.

A Streamlit web application (TODO: Deploy publicly on Streamlit Community Cloud in M4, and include URL) will allow users to submit review text, optionally include a review summary, and compare sentiment predictions from trained machine learning models. Currently, the app is deployed locally for testing and refinement purposes (see Quick Start).

## Team Members

| Name | Role |
|---|---|
| Jian Zhang | Project Lead |
| Bimal Timsina | Analyst / Writer |
| Dmitry Volf | App Engineer |
| Raynor He | Data Engineer |
| Jayden Vidican | ML Engineer |

## Quick Start

These steps run the Streamlit app locally. Trained models are located in 'models/', so you do not need to retrain anything.

1. Clone the repository and switch to your desired branch with 'git checkout -b'

2. Create and activate a python 3.13 virtual environment.

 Windows (PowerShell):
```
python -m venv venv
.\venv\Scripts\Activate.ps1
```
 
   macOS or Linux:
 
```
python3 -m venv venv
source venv/bin/activate
```
 
3. Install dependencies:
```
pip install -r requirements.txt
```
 
   The saved models were created with scikit-learn 1.9.1. Use the pinned versions in `requirements.txt`, because a different scikit-learn version can fail to load them or change predictions.
 
4. Start the app from the repository root:
```
streamlit run app/app.py
```
 
   Then open http://localhost:8501. If the `streamlit` command is not found on Windows, use `python -m streamlit run app/app.py`. Stop the app with Ctrl+C.

## Web Application
 
The app is in `app/app.py`. Enter a review and, optionally, a review summary, then select **Analyze sentiment**.
 
| Tab | What it shows |
|---|---|
| Predictions | For each model: the predicted label (positive or negative), the probability of positive sentiment, and the confidence in the predicted label |
| Model comparison | The list of available models and the summary setting each one uses |
| About | A short explanation of how the models process a review |
 
The app currently predicts with two models side by side: the Feedforward Neural Network (TF-IDF) and Logistic Regression (TF-IDF). The Naive Bayes models and Logistic Regression (BoW) are also saved in `models/` and configured in `app.py`, but they are not used for predictions yet.
 
The review summary is optional. If it is left empty, the models classify the review text alone.

### How the app builds each model's input
 
Each model must receive text in the same format it was trained on, and must be paired with its own vectorizer. The files in `models/` are named `<key>_model.joblib` and `<key>_vectorizer.joblib`.
 
| Key | Model | Review summary is included |
|---|---|---|
| `nb_bow` | Naive Bayes, Bag of Words | twice, then the review text |
| `nb_tfidf` | Naive Bayes, TF-IDF | twice, then the review text |
| `lr_bow` | Logistic Regression, Bag of Words | once, then the review text |
| `lr_tfidf` | Logistic Regression, TF-IDF | once, then the review text |
| `fnn_tfidf` | Feedforward Neural Network, TF-IDF | twice, then the review text |
 
The `MODELS` list at the top of `app/app.py` stores these settings. If a model is retrained with a different input format, update it there.

## Dataset

The project uses the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), containing 568,454 customer reviews.

The primary attributes are

- **Text** — Written customer reviews
- **Summary** — Optional short review summaries
- **Score** — Five-star customer ratings used for sentiment labeling

The project uses binary sentiment classification. 

Group Lambda uses a 50,000-review sample generated from the original dataset using the instructor-assigned random seed of 45. This sample will be used consistently throughout model development and evaluation. Preliminary analysis showed that five-star reviews account for 63.83% of the sample, indicating a substantial imbalance toward higher ratings.

**Labels:** reviews with 4 or 5 stars are positive (1), and reviews with 1 or 2 stars are negative (0). Three-star reviews are excluded. After removing three-star reviews and duplicate review texts, 42,893 labeled reviews remain (36,107 positive and 6,786 negative). They are split 80% for training (34,314 reviews) and 20% for testing (8,579 reviews). All models are evaluated on the same test set.

**Files in `data/`** that the training scripts read:
 
- `UsuableData_80%.xlsx`: the training split
- `UsuableData_20%.xlsx`: the held-out test split
- `BoW_0.1%Removed_filtered.xlsx`: the filtered vocabulary used by the Naive Bayes and Logistic Regression models


## Models
 
**Feature representations**
 
- Bag of Words (BoW): word counts over a shared vocabulary of 3,716 words, built by removing infrequent words and common stopwords. Used by Naive Bayes and Logistic Regression.
- TF-IDF: the same shared vocabulary, with words weighted by how distinctive they are across reviews. Used by Naive Bayes and Logistic Regression.
- TF-IDF for the neural network: its own vectorizer with unigrams and bigrams, up to 15,000 features, and sublinear term frequency.

**Models**
 
- Naive Bayes: multinomial Naive Bayes with alpha = 1.0
- Logistic Regression: the regularization strength C was chosen with 5-fold grid search over 0.01, 0.1, 1.0 and 10.0, scored by F1. The best value was C = 0.1 for BoW and C = 10.0 for TF-IDF.
- Feedforward Neural Network: scikit-learn `MLPClassifier` with one hidden layer of 64 ReLU units, Adam optimizer, L2 penalty 1e-3, batch size 32, up to 20 epochs with early stopping (10% validation split, stop after 3 epochs without improvement), and random seed 42.

## Results
 
All scores are on the 8,579-review held-out test set. Precision, recall and F1 are for the positive class. A baseline that always predicts positive reaches 0.8459 accuracy.
 
| Model | Accuracy | Precision | Recall | F1 | Training time |
|---|---|---|---|---|---|
| Naive Bayes (BoW) | 0.9193 | 0.9581 | 0.9460 | 0.9520 | 0.1104 s |
| Naive Bayes (TF-IDF) | 0.8875 | 0.8854 | 0.9960 | 0.9374 | 0.0310 s |
| Logistic Regression (BoW) | 0.9278 | 0.9396 | 0.9775 | 0.9582 | 0.1629 s |
| Logistic Regression (TF-IDF) | 0.9291 | 0.9442 | 0.9738 | 0.9588 | 0.1912 s |
| Feedforward NN (TF-IDF) | 0.9524 | 0.9651 | 0.9792 | 0.9721 | 198.25 s |
 
Training times were measured on one team member's computer. The Feedforward NN time is the mean of 4 runs, because it varies with hardware load. The others are single runs. The Logistic Regression time covers only the final fit with the chosen C, not the grid search.

**Confusion matrices** (rows are actual, columns are predicted; order is negative, positive):
 
| Model | Matrix |
|---|---|
| Naive Bayes (BoW) | [[1022, 300], [392, 6865]] |
| Naive Bayes (TF-IDF) | [[386, 936], [29, 7228]] |
| Logistic Regression (BoW) | [[866, 456], [163, 7094]] |
| Logistic Regression (TF-IDF) | [[904, 418], [190, 7067]] |
| Feedforward NN (TF-IDF) | [[1065, 257], [151, 7106]] |
 
**Known limitations**
 
- Negative reviews are harder to catch than positive ones. Recall on negative reviews is 0.81 for the Feedforward NN and 0.66 to 0.68 for Logistic Regression, because only about 16% of labeled reviews are negative.
- Reviews with mixed sentiment and reviews that use negation were the main error patterns found in the Milestone 2 error analysis.
- Naive Bayes with TF-IDF predicts almost every review as positive, so it misses most negative reviews.

## Retraining the Models
 
Run all commands from the repository root, because the scripts use relative paths. The `models/` folder must exist. Each script prints its test results and saves the model and vectorizer to `models/`.
 
- Naive Bayes: set `featureRep` at the top of `code/NBModel.py` to `"BoW"` or `"TF-IDF"`, then run `python code/NBModel.py`.
- Logistic Regression: run `python code/LRModel.py 1` for BoW or `python code/LRModel.py 2` for TF-IDF.
- Feedforward NN: run `python code/FNNModel.py`. It also writes `fnn_learning_curves.png` in the current folder. Expect about 3 to 4 minutes, which varies by machine.
Retrained files overwrite the ones in `models/`. If you change how a model's input text is built, update the `MODELS` list in `app/app.py` to match.
 
## Repository Structure
 
| Path | Contents |
|---|---|
| `app/app.py` | Streamlit application |
| `code/` | Training and evaluation scripts for the three models |
| `data/` | Cleaned train and test splits and the filtered vocabulary |
| `models/` | Saved model and vectorizer files (`.joblib`) |
| `requirements.txt` | Python dependencies with pinned versions |
| `fnn_learning_curves.png` | Feedforward NN training curves |

## Project Milestones

| Milestone | Target Date | Deliverable |
|---|---|---|
| M1 | November 10, 2026 | Project charter, dataset exploration, and technical plan |
| M2 | November 18, 2026 | Initial feature pipeline, classical model, and evaluation |
| M3 | November 25, 2026 | Complete model comparison and Streamlit prototype |
| M4 | November 30, 2026 | Final report, presentation, and deployed application |
| Final Preparation | December 2, 2026 | Testing, demonstration rehearsal, and Q&A preparation |

## Project Status

As of M3: all five model variants are trained and evaluated, and the Streamlit prototype runs locally with two models. Remaining work for M4 includes deploying the app to Streamlit Community Cloud, adding the remaining models and the metrics table to the app, further preprocessing and feature improvements, and the final report and presentation.
 
