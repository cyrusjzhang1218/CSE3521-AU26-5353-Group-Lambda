# AI-Powered Sentiment Analysis of Amazon Food Reviews

**CSE 3521 — Introduction to Artificial Intelligence I**  
**Autumn 2026 | Group Lambda**

## Project Overview

This project explores the use of machine learning to automatically classify customer sentiment in Amazon food reviews. The system will analyze written customer feedback and classify reviews as either positive or negative.

The goal is to help e-commerce vendors and product managers understand customer satisfaction, identify negative feedback, and prioritize reviews for further investigation.

A Streamlit web application will allow users to submit review text, optionally include a review summary, and compare sentiment predictions from trained machine learning models.

## Team Members

| Name | Role |
|---|---|
| Jian Zhang | Project Lead |
| Bimal Timsina | Analyst / Writer |
| Dmitry Volf | App Engineer |
| Raynor He | Data Engineer |
| Jayden Vidican | ML Engineer |

## Dataset

The project uses the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), containing 568,454 customer reviews.

The primary attributes are

- **Text** — Written customer reviews
- **Summary** — Optional short review summaries
- **Score** — Five-star customer ratings used for sentiment labeling

The project uses binary sentiment classification. Preliminary analysis uses the full dataset while the required 50,000-review sample is being clarified.

Raw dataset files will not be committed to this repository.

## Planned Technical Approach

**Feature Representations**
- Bag of Words (BoW), including unigrams and bigrams
- Term Frequency–Inverse Document Frequency (TF-IDF)

**Machine Learning Models**
- Naive Bayes
- Logistic Regression
- Feedforward Neural Network (FNN)

**Model Evaluation**

Models will be evaluated using a shared 20% held-out test set and compared against a majority-class baseline.

Evaluation metrics will include accuracy, precision, recall, F1-score, and confusion matrices.

## Web Application

The planned Streamlit application will provide

- User input for review text and optional summary
- Live positive or negative sentiment predictions
- Predictions from at least two trained models
- A simple interface for comparing model outputs

## Project Milestones

| Milestone | Target Date | Deliverable |
|---|---|---|
| M1 | November 10, 2026 | Project charter, dataset exploration, and technical plan |
| M2 | November 15, 2026 | Initial feature pipeline, classical model, and evaluation |
| M3 | November 22, 2026 | Complete model comparison and Streamlit prototype |
| M4 | November 29, 2026 | Final report, presentation, and deployed application |
| Final Preparation | December 2, 2026 | Testing, demonstration rehearsal, and Q&A preparation |

## Project Status

The project is currently in the planning and initial dataset exploration phase. Model implementation, evaluation results, and application instructions will be added as development progresses.
