# ECE2028 Final Project: Machine Learning Fairness Tech Demo

This repository contains a technical demonstration for evaluating and visualizing fairness disparities in machine learning models. Using the **UCI Adult Income Dataset**, the project builds a baseline predictive model and assesses its performance across different demographic groups (e.g., sex and race) using the Fairlearn library.

## Project Overview

The objective of this project is to demonstrate the inherent biases that can exist in unconstrained machine learning models and provide a framework for measuring those biases. 

### Key Features
* **Automated Preprocessing**: Implements a robust `scikit-learn` pipeline handling missing value imputation, feature scaling for numerical data, and one-hot encoding for categorical data.
* **Baseline Modeling**: Trains an unconstrained Logistic Regression model to serve as a performance and fairness control.
* **Fairness Metrics**: Utilizes the `fairlearn` library to calculate:
    * Demographic Parity Difference
    * Equalized Odds Difference
    * Performance metrics (Accuracy, Precision, Recall, F1) broken down by sensitive groups.
* **Visualizations**: 
    * Disparity bar charts to compare model performance across sex and race groups.
    * Calibration curves to assess the reliability of model probability predictions.

## Getting Started

### Prerequisites
To run this notebook, you will need Python 3.8+ and the libraries listed in the `requirements.txt` file.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zeephus/ECE2028-Final
   ```
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the notebook
   ```bash
   jupyter notebook BiasMitigationComparison.ipynb

## Dataset 
This demo uses the UCI Adult Income Dataset from the UCI Machine Learning Repository. It predicts whether an individual's income is greater than $50k/year based on data. 
