# Cyberbullying Role Identification NLP Model

## Overview
This repository contains an NLP model designed to identify roles associated with cyberbullying posts. The system was developed as part of my project with The University of Adelaide. It analyzes social media text to detect different roles typically involved in cyberbullying. The roles for detection are:
- Harasser
- Victim

## Summary of the two models:
Two performant models have been provided.
These were the better performing systems out of the experiemnts of models on the oversampled data.

## Model 1: Logistic Regression Classifier paired with FastText word embeddings:
This model has the followint features:
- Preprocessing: removal of special characters, removal of single characters, lower-case conversion, lemmatization, stop-word removal
- Embeddings: FastText word embeddings
- Classification: Logistic Regression classifier
This system achieved a weighted F1 score of approximately 0.7492 on the cross-validation set.
See Model1.py

## Model 2: Logistic Regression Classifier paired with Universal Sentence Encoder embeddings
This system achieved a weighted F1 score of approximately 0.7446 on the cross-validation set.
See Model 2.py

## Usage
```bash
git clone -
python Model1.py

To run model 2, run
```bash
python Model2.py

