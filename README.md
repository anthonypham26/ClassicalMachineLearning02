# Machine Learning Level 2 🚀

## Index

1. [Recap](#1-recap)
2. [Learning Objectives](#2-learning-objectives)
3. [Classical Machine Learning Overview](#3-classical-machine-learning-overview)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
4. [Training a Simple ML Model](#4-training-a-simple-ml-model)
5. [Performance Metrics](#5-performance-metrics)
6. [Cross-Validation](#6-cross-validation)
7. [Model Evaluation & Hyperparameter Tuning](#7-model-evaluation--hyperparameter-tuning)
    - [Model Evaluation](#model-evaluation)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Conclusion & Next Steps](#8-conclusion--next-steps)

## 1. Recap

In Level 1, we laid the foundation of machine learning by focusing on the early stages of any ML project—understanding what machine learning is and how to prepare data for modeling.

Last week, we explored:

  - ✅ What is Machine Learning  
  - ✅ How to clean and transform raw data into usable form  
    - Working with CSV files, handling missing values, removing outliers, and basic data visualization  
    - Scaling numerical features and encoding categorical variables  
  - ✅ The importance of ethics, fairness, and human values in building responsible AI systems

This week, we’ll move beyond data preparation and talk about the rest of remaining stages of the machine learning pipeline.

- How to choose and train a model  
- How to evaluate model performance using the right metrics  
- How to use cross-validation to ensure reliable results  
- How to fine-tune your model using hyperparameter optimization  
- And finally, how to prepare your model for presentation or deployment

By the end, you’ll understand how to take a project from raw data all the way to a fully trained and tested machine learning model.

## 2. Learning Objectives

- Understand basic ML model categories  
- Distinguish between regression and classification  
- Recognize unsupervised learning problems  
- Train and evaluate a simple model  
- Know what cross-validation is and why it matters  


## 3. Classical Machine Learning Overview

Classical Machine Learning refers to a group of foundational algorithms and techniques that were developed before rapid growth of deep learning and neural networks. These methods are still widely used today because they are fast, interpretable, and effective for solving many real-world problems.

<kbd><img src="images/SupervisedvsUnsupervised.webp" style="border:1px solid grey; border-radius:10px;"></kbd>

**Types of Learning in Classical Machine Learning:**

In Classical Machine Learning, there are a few types of learning methods. The two most common are **Supervised Learning** and **Unsupervised Learning**, but others like **Semi-Supervised** and **Reinforcement Learning** also exist and play their own important roles in specific scenarios.

---

### Supervised Learning 

Supervised learning uses **labeled data**, meaning each training example includes both input features and a correct output (label).

📌 **Goal:** Learn a function that maps inputs to outputs  
📌 **Example:** Predicting house prices based on features like size, location, and number of rooms

Supervised learning algorithms are divided into two categories based on the type of output:

**1) Classification Algorithms (Predict discrete categories)**
- Logistic Regression  
- Support Vector Machines (SVM)  
- k-Nearest Neighbors (k-NN)  
- Decision Trees  
- Random Forest  
- Naive Bayes  
- Gradient Boosting (e.g., XGBoost, AdaBoost)

**2) Regression Algorithms (Predict continuous values)**
- Linear Regression  
- Ridge/Lasso Regression  
- Support Vector Regression (SVR)  
- Decision Tree Regressor  
- Random Forest Regressor  
- k-Nearest Neighbors (k-NN) Regression

### Unsupervised Learning

Unsupervised learning works with **unlabeled data**. The algorithm tries to find patterns, groupings, or structure in the data without knowing the correct output in advance.

<kbd><img src="images/Supervised-and-unsupervised.png" style="border:1px solid grey; border-radius:10px;"></kbd>

📌 **Goal:** Discover hidden structures or relationships within the data  
📌 **Example:** Segmenting customers into different groups based on their purchasing behavior  
📌 **How to Recognize?:** No "target" column, or the goal is to group, compress, or summarize data

Unsupervised learning includes the following categories:

#### 1) Clustering Algorithms 

Clustering algorithms are used to automatically group data points into clusters based on similarity, without needing labeled data.

- K-Means  
- Hierarchical Clustering  
- DBSCAN  
- Mean Shift

#### 2) Dimensionality Reduction Algorithms 

Dimensionality reduction techniques simplify datasets by reducing the number of input features while preserving important information and patterns.

- Principal Component Analysis (PCA)  
- t-SNE  
- Autoencoders *(transitions into deep learning)*  
- Factor Analysis

<kbd><img src="images/dimensionalityReduction.png" style="border:1px solid grey; border-radius:10px;"></kbd>

#### 3) Association Rule Learning 

Association rule learning finds relationships and patterns between variables in large datasets.

- Apriori  
- Eclat

---

All of these algorithms involve a lot of math and reasoning behind each one. This is one of the easiest parts to **implement**, but one of the hardest to **understand deeply**. To learn more, you can explore visual explanations, online tutorials, or dive into the theory to understand how and why they work under the hood.

[Mathematical and Visual explanation of some algorithms](https://mlu-explain.github.io/)


### Other Types of Learning

**Semi-Supervised Learning:** combines a small amount of labeled data with a large amount of unlabeled data. This is useful when labeling is expensive or time-consuming, and we still want to benefit from supervised learning.

**Reinforcement Learning:** involves a model learning by interacting with an environment and receiving rewards or penalties. While it's less common in classical ML, it's widely used in areas like robotics, game-playing agents, and recommendation systems.

## 4. Training a Simple ML Model

Now comes the fun part—actually training a machine learning model!

Once your data is cleaned, transformed, and ready, the coding process is surprisingly simple. In many cases, it only takes a single import and few lines of code to get started.

### Typical Steps in Training a Model:

1. **Split the Data** – Divide your dataset into training and testing sets  
2. **Define the Goal** – Determine whether your problem is *supervised* or *unsupervised*
3. **Choose an Algorithm** – Based on your goal, select a few applicable models to try  
4. **Train & Test** – Train the model on your training data, then test it on your unseen test data  
5. **Evaluate Performance** – Use metrics like **accuracy**, **precision**, or **mean squared error** to decide which model performs best

It’s common to try multiple models and compare their performance before choosing the best one.

---

### Example: K-Nearest Neighbors Classifier

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Create and train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
## 5. Performance Metrics

After training a model, we need a way to measure how well it performs. This is where **performance metrics** come in. The right metric depends on your problem type and what matters most in your specific use case.

Not all “good” results mean the same thing. A model might look great on the surface but fail in areas that matter most for your goal.


**Example 1: Email Spam Filter (Classification):**

  Let’s say you're building a model to detect spam emails.

  - **Accuracy** = How many emails the model got right overall  
    If your model is 95% accurate, that sounds great… right?

  But what if the model misses a lot of spam or, worse, marks real emails as spam?

  That’s where two other metrics come in:

  - **Precision** = Of the emails marked as spam, how many were actually spam?  
    High precision means fewer **false alarms**.

  - **Recall** = Of all the actual spam emails, how many did the model catch?  
    High recall means fewer **missed spam**.

  📌 **If missing spam is okay but marking real emails as spam is bad, go for high precision.**  
  📌 **If catching all spam is critical (even if some real emails are marked incorrectly), go for high recall.**



**Example 2: Predicting House Prices (Regression):**

  Now imagine you’re predicting house prices. The model says a house is worth **$300,000**, but the real value is **$310,000**. That’s a **$10,000 error**.

  We use these metrics to measure how far off the predictions are:

  - **MAE (Mean Absolute Error)** = On average, how many dollars off are we?  
  - **MSE (Mean Squared Error)** = Same idea, but **larger mistakes are punished more** because the errors are squared  
  - **RMSE (Root Mean Squared Error)** = Like MSE, but puts the result back in the original units (like dollars)  
  - **R² Score** = Quantifies how well a regression model's predictions align with the actual data points (closer to 1 is better)

  📌 **If small errors are okay**, use MAE.  
  📌 **If big mistakes are really bad**, use MSE or RMSE to penalize them more.  
  📌 **If you want to know how much variance your model explains**, use R².

The goal is to always pick the metric that matches the **real-world impact** of your predictions. A good model in one case might be a poor fit for another, depending on what errors matter most.

## 6. Cross-Validation

In our earlier example, we used a common approach: splitting the dataset into two parts—one for training and one for testing. While this is simple and widely used, there's a problem. A single train-test split can be **unreliable**, especially with smaller datasets. The model’s performance might vary significantly depending on how the data is divided.

That’s where **Cross-Validation** comes in.

<kbd><img src="images/cross_validation.png" style="border:1px solid grey; border-radius:10px;"></kbd>

Cross-validation is a more **robust and reliable** method for evaluating a machine learning model. Instead of training and testing the model just once, cross-validation splits the data into multiple parts (called **folds**). The model is trained and tested multiple times—each time using a different fold for testing and the remaining folds for training.

In the end, you average the results from each run to get a more stable and accurate estimate of your model’s performance.

- Provides a **better estimate** of model performance  
- Helps **prevent overfitting**  
- **Reduces variance** caused by random train/test splits  

Cross-validation is especially important when you’re tuning hyperparameters or comparing different models. It ensures that the performance you're seeing isn't just a result of a "lucky" data split.

## 7. Model Evaluation & Hyperparameter Tuning

Training a model is just the beginning. After that, the next steps are:

- **Evaluate how well it performs**  
- **Improve it through tuning**

---

### Model Evaluation

Once you've chosen a model and trained it, you need to evaluate how well it's performing. This means looking at the **performance metrics** (like accuracy, precision, or MAE) and checking for issues such as:

- **Overfitting** – The model performs very well on training data but poorly on unseen data  
- **Underfitting** – The model performs poorly on both training and testing data because it hasn’t learned enough  

<kbd><img src="images/OverfitingvsUnderfitting.svg" style="border:1px solid grey; border-radius:10px;"></kbd>

In this figure, the lines represent our model (or function), and the dots are our data points. Each scenario shows how well the model is able to generalize (how well it can make accurate predictions on new, unseen data.)

This means that when we introduce a new data point that the model hasn't seen before, we want the model to make a prediction that is as close as possible to the true value.

- In the overfitting case, the model is too complex and tries to fit every single point in the training data, even the noise. While it may perform very well on the training set, it fails to generalize and performs poorly on new data.

- In the underfitting case, the model is too simple and doesn't capture the underlying pattern in the data. It performs poorly on both the training data and new data because it hasn’t learned enough.

- In the ideal fit, the model captures the true underlying pattern without overcomplicating things. It performs well on both the training data and new data, showing good generalization.

The goal is to find the right balance, where the model is just complex enough to learn the important patterns but not so complex that it memorizes the data.

**How to implement:**
To spot these issues, use graphs like Accuracy or Error curves during training.

<kbd><img src="images/OvsUGraph.jpeg" style="border:1px solid grey; border-radius:10px;"></kbd>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load data
X, y = load_iris(return_X_y=True)

# Choose a simple model
model = LogisticRegression(max_iter=200)

# Get learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy'
)

# Compute average scores across folds
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

# Plot
plt.plot(train_sizes, train_scores_mean, label='Training Score')
plt.plot(train_sizes, val_scores_mean, label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
```
---

### Hyperparameter Tuning

Every model has settings you can adjust, called *hyperparameters*. These are not learned from the data but are defined before training begins.

#### Example:  
For *K-Nearest Neighbors (KNN)*, a key hyperparameter is:

- `n_neighbors`: How many nearby points should the model consider?

You can try different values like 3, 5, or 7 and compare their accuracy.

### Common Hyperparameters by Model

| Model              | Common Hyperparameters                    |
|-------------------|--------------------------------------------|
| KNN                | `n_neighbors`                              |
| Decision Tree      | `max_depth`, `min_samples_split`           |
| Random Forest      | `n_estimators`, `max_depth`                |
| SVM                | `C`, `kernel`, `gamma`                     |
| Gradient Boosting  | `learning_rate`, `n_estimators`, `max_depth` |

The hardest part about hyperparameter tuning is understanding **which hyperparameters to tune and why**. This requires a deeper understanding of how each machine learning algorithm works internally and going into that would take too long for this presentation.

Instead, I recommend starting simple:  
Pick **one algorithm** you're using, and look it up online to see what it is and how it works, then search **Scikit-learn's documentation** to see what hyperparameters are available and what they control.

Focus on understanding a **few key hyperparameters** that commonly affect the model’s behavior, and experiment by changing their values slightly. This helps you see how small adjustments can impact the model’s accuracy, overfitting, or underfitting.

As you grow more comfortable, you'll begin to recognize which hyperparameters matter most for different types of problems.

For now, just remember:
- **Hyperparameters** are settings you define before training your model (like`n_neighbors` in KNN).
- They are different from **parameters**, which are learned by the model during training.

### How to Tune?

We usually combine hyperparameter tuning with **Cross-Validation**, using tools like **Grid Search**.

Grid Search is a method that automatically tests all possible combinations of hyperparameter values to find the best-performing model configuration.

#### Grid Search Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

params = {'n_neighbors': [3, 5, 7, 9]}
grid = GridSearchCV(KNeighborsClassifier(), params, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
```

## 8. Conclusion & Next Steps

At this point, you've gone through the **entire machine learning process** from start to finish:

- You explored and cleaned your data  
- You transformed it to make it model-ready  
- You chose the right algorithms  
- You trained and evaluated multiple models  
- You tuned them to improve performance  

Now all that’s left is to **communicate your results** in a way that’s clear and impactful. Whether it's through a dashboard, a report, or a  slideshow. Presenting your findings effectively is the final step that shows the value of everything you've done.


#### What's Next:

Now that you've completed the full modeling cycle, from preparing data to training, evaluating, and tuning models. You’ve built a solid foundation in classical machine learning.

In the next presentation, we’ll take things further by diving deeper into several key topics we’ve touched on and explore new concepts.

We’ll look at **advanced data cleaning techniques** for handling messy, real-world datasets, including smarter methods for dealing with outliers, missing values, and noisy data.

You’ll also be introduced to **dimensionality reduction techniques**, such as **Principal Component Analysis (PCA)**, which help simplify complex datasets while preserving the most important information.

We’ll also introduce **neural networks**, explaining how they mimic the brain to solve more complex problems and why they’ve become the foundation of modern artificial intelligence.

Finally, we’ll cover **pipelines**, which help automate and organize the entire machine learning workflow.


