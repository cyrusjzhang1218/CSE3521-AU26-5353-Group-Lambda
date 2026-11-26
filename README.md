# AI-Powered Sentiment Analysis of Amazon Food Reviews

**CSE 3521 — Survey of Artificial Intelligence I**  
**Autumn 2026 | Group Lambda**

## Project Overview

This project uses machine learning to classify customer sentiment in Amazon food reviews. The system analyzes written customer feedback and classifies reviews as positive or negative using three model families: Naive Bayes, Logistic Regression, and a Feedforward Neural Network (FNN). The models use Bag of Words or TF-IDF features. The Streamlit application compares predictions from all five trained models.

The goal is to help e-commerce vendors and product managers understand customer satisfaction, identify negative feedback, and prioritize reviews for further investigation.

The Streamlit application allows users to submit review text, optionally include a review summary, and compare sentiment predictions. It is deployed on Streamlit Community Cloud:

**Live app:** https://group-lambda-sentiment.streamlit.app

## Team Members

| Name | Role |
|---|---|
| Jian Zhang | Project Lead |
| Bimal Timsina | Analyst / Writer |
| Dmitry Volf | App Engineer |
| Raynor He | Data Engineer |
| Jayden Vidican | ML Engineer |

## Quick Start

These steps run the Streamlit app locally. Trained models are located in `models/`, so retraining is not required to run the app.

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/cyrusjzhang1218/CSE3521-AU26-5353-Group-Lambda.git
   cd CSE3521-AU26-5353-Group-Lambda
   ```

2. Install dependencies:
```
python -m pip install -r requirements.txt
```

   The saved models were created with scikit-learn 1.9.1. Use the pinned versions in `requirements.txt`, because a different scikit-learn version can fail to load them or change predictions.

3. Start the app from the repository root:
```
streamlit run app/app.py
```

   Then open http://localhost:8501. If the `streamlit` command is not found on Windows, use `python -m streamlit run app/app.py`. Stop the app with Ctrl+C.

Before deploying changed models or app code, run the held-out test-set check from the repository root:

```bash
python -m scripts.check_model_metrics
```

The check evaluates every saved model against `data/model_metrics.csv` and fails if a metric differs at four decimal places. Confirm that all five models pass before deploying.

## Web Application

The app is in `app/app.py`. Enter a review and, optionally, a review summary, then select **Analyze sentiment**. The four sample-review buttons load example reviews.

| Tab | What it shows |
|---|---|
| Predictions | Results grouped by model family, with the predicted label (positive or negative), probability of positive sentiment, prediction confidence, and a one-line agreement summary |
| Model comparison | The list of available models and the summary setting each one uses |
| About | A short explanation of how the models process a review |

All five saved models are used for each prediction. The review summary is optional; when provided, each model applies the summary repetition count used during training.

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

Group Lambda uses a 50,000-review sample generated from the original dataset using the instructor-assigned random seed of 45. This sample is used consistently throughout model development and evaluation. Preliminary analysis showed that five-star reviews account for 63.83% of the sample, indicating a substantial imbalance toward higher ratings.

**Labels:** reviews with 4 or 5 stars are positive (1), and reviews with 1 or 2 stars are negative (0). Three-star reviews are excluded. After removing 3,728 neutral reviews and 3,379 duplicate review texts, 42,893 labeled reviews remain (36,107 positive and 6,786 negative). The fixed split reserves 80% for model development (34,314 reviews) and 20% for testing (8,579 reviews). The FNN reserves 10% of the development partition internally for validation. All models are evaluated on the same test set.

**Files in `data/`** that the training scripts read:

- `UsuableData_80%.xlsx`: the fixed model development partition
- `UsuableData_20%.xlsx`: the held-out test split
- `BoW_0.1%Removed_filtered.xlsx`: the filtered vocabulary used by the Naive Bayes and Logistic Regression models

The existing processed Excel files are sufficient to reproduce the M3 training runs. The original Kaggle dataset is needed only to reconstruct the sample and preprocessing from scratch.


## Models

**Feature representations**

- Bag of Words (BoW): word counts over a shared vocabulary of 3,716 words, built by removing infrequent words and common stopwords. Used by Naive Bayes and Logistic Regression.
- TF-IDF: the same shared vocabulary, with words weighted by how distinctive they are across reviews. Used by Naive Bayes and Logistic Regression.
- TF-IDF for the neural network: its own vectorizer with unigrams and bigrams, a maximum of 15,000 features (15,000 in the reported experiment), and sublinear term frequency.

**Models**

- Naive Bayes: multinomial Naive Bayes with alpha = 1.0
- Logistic Regression: the regularization strength C was chosen with 5-fold grid search over 0.01, 0.1, 1.0 and 10.0, scored by F1. The best value was C = 0.1 for BoW and C = 10.0 for TF-IDF.
- Feedforward Neural Network: scikit-learn `MLPClassifier` with one hidden layer of 64 ReLU units, Adam optimizer, L2 regularization alpha = 0.001, learning rate = 0.001, batch size 32, and a maximum of 20 epochs. Early stopping uses validation accuracy with a patience setting of 3 and restores the best validation accuracy weights. The validation fraction is 0.10 and the random state is 42. The reported run completed six epochs.

## Results

All scores are on the 8,579-review held-out test set. Precision, recall and F1 are for the positive class. A baseline that always predicts positive reaches 0.8459 accuracy.

| Model | Accuracy | Precision | Recall | F1 | Training time |
|---|---|---|---|---|---|
| Naive Bayes (BoW) | 0.9193 | 0.9581 | 0.9460 | 0.9520 | 0.023 s |
| Naive Bayes (TF-IDF) | 0.8875 | 0.8854 | 0.9960 | 0.9374 | 0.020 s |
| Logistic Regression (BoW) | 0.9278 | 0.9396 | 0.9775 | 0.9582 | 0.159 s |
| Logistic Regression (TF-IDF) | 0.9291 | 0.9442 | 0.9738 | 0.9588 | 0.124 s |
| Feedforward NN (TF-IDF) | 0.9524 | 0.9651 | 0.9792 | 0.9721 | 190.50 s |

Training times are the M3 report measurements. They measure model fitting only, excluding data loading, feature extraction, and hyperparameter tuning. Times vary depending on hardware and system workload.

The expected test metrics are also recorded in `data/model_metrics.csv`. Run `python -m scripts.check_model_metrics` before each deployment to compare the saved models against these reported values.

The FNN achieves the highest test accuracy (95.24%) and positive sentiment F1 score (0.9721). It also correctly identifies 1,065 of 1,322 negative test reviews (80.56% negative recall).

**Confusion matrices** (rows are actual, columns are predicted; order is negative, positive):

| Model | Matrix |
|---|---|
| Naive Bayes (BoW) | [[1022, 300], [392, 6865]] |
| Naive Bayes (TF-IDF) | [[386, 936], [29, 7228]] |
| Logistic Regression (BoW) | [[866, 456], [163, 7094]] |
| Logistic Regression (TF-IDF) | [[904, 418], [190, 7067]] |
| Feedforward NN (TF-IDF) | [[1065, 257], [151, 7106]] |

**Known limitations**

- Negative reviews are harder to catch than positive ones. Recall on negative reviews is about 0.81 for the FNN and 0.66 to 0.68 for Logistic Regression. Only about 16% of labeled reviews are negative, so overall accuracy alone does not fully describe performance.
- Reviews with mixed sentiment and reviews that use negation were the main error patterns found in the Milestone 2 error analysis.
- Naive Bayes with TF-IDF predicts almost every review as positive, so it misses most negative reviews.
- The FNN requires longer training than the classical models. Its increasing validation loss suggests signs of overfitting, motivating further tuning in M4.

## Retraining the Models

Run all commands from the repository root, because the scripts use relative paths. The `models/` folder must exist. Each script prints its test results and saves the model and vectorizer to `models/`.

- Naive Bayes: set `featureRep` at the top of `code/NBModel.py` to `"BoW"` or `"TF-IDF"`, then run `python code/NBModel.py`.
- Logistic Regression: run `python code/LRModel.py 1` for BoW or `python code/LRModel.py 2` for TF-IDF.
- Feedforward NN: run `python code/FNNModel.py`. It also writes `fnn_learning_curves.png`, which shows training loss and validation accuracy. Training duration varies by machine.
Retrained files overwrite the corresponding files in `models/`. If you change how a model's input text is built, update the `MODELS` list in `app/app.py` to match.

### FNN Training and Validation Loss Experiment

To reproduce the loss curves used in the M3 report, run:

```bash
python code/FNNLossExperiment.py
```

This separate experiment uses the same FNN configuration and accuracy based early stopping as `FNNModel.py`. It records validation cross entropy loss after each epoch and saves the training and validation loss graph as `fnn_loss_experiment.png`.

The experiment does not overwrite the saved model or vectorizer when its `export` setting remains `"n"`. Keep this setting unchanged when reproducing the report graph. The script uses an internal scikit-learn method, so use the pinned scikit-learn version in `requirements.txt`.

Training loss decreased while validation loss increased after the early epochs, suggesting signs of overfitting. Validation accuracy and validation loss measure different aspects of performance. Training loss also includes a regularization penalty, unlike the recorded validation loss. In M4, the team plans to investigate whether tuning regularization, network size, learning rate, and early stopping can reduce overfitting while maintaining or improving classification performance.

## Repository Structure

| Path | Contents |
|---|---|
| `app/app.py` | Streamlit application |
| `scripts/check_model_metrics.py` | Checks saved-model metrics against the held-out test split and metrics CSV |
| `data/model_metrics.csv` | Expected held-out test metrics for all five models |
| `code/NBModel.py` | Naive Bayes training and evaluation |
| `code/LRModel.py` | Logistic Regression training and evaluation |
| `code/FNNModel.py` | FNN training, evaluation, and model export |
| `code/FNNLossExperiment.py` | FNN training and validation loss experiment |
| `data/` | Cleaned train and test splits and the filtered vocabulary |
| `models/` | Saved model and vectorizer files (`.joblib`) |
| `requirements.txt` | Python dependencies with pinned versions |
| `fnn_learning_curves.png` | FNN training loss and validation accuracy |
| `fnn_loss_experiment.png` | FNN training and validation loss curves |

## Project Milestones

The following dates are the team's existing planning targets. Refer to Carmen for official submission and presentation deadlines.

| Milestone | Target Date | Deliverable |
|---|---|---|
| M1 | November 10, 2026 | Project charter, dataset exploration, and technical plan |
| M2 | November 18, 2026 | Initial feature pipeline, classical model, and evaluation |
| M3 | November 25, 2026 | Complete model comparison and Streamlit prototype |
| M4 | November 30, 2026 | Final report, presentation, and deployed application |
| Final Preparation | December 2, 2026 | Testing, demonstration rehearsal, and Q&A preparation |

## Project Status

As of M4: all five model variants are available in the deployed Streamlit app. The interface includes sample reviews and grouped model comparisons. Run the held-out metrics check before deploying changes; sample inputs can be checked manually in the app.
