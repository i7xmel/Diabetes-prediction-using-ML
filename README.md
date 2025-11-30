# Diabetes Prediction Project

## Introduction

Our project focuses on leveraging advanced predictive modeling and machine learning techniques for early diabetes detection. By integrating diverse health parameters like genetic markers, lifestyle data, medical history, and physiological measurements, we aim to create a sophisticated predictive model. This pioneering approach utilizes cutting-edge machine learning algorithms to uncover subtle patterns in vast datasets, potentially predicting diabetes onset before clinical symptoms appear. Early identification enables timely interventions and personalized healthcare strategies, holding promise for improving patient outcomes and reducing the burden on healthcare systems globally.

## Dataset Information

**Source**: This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

**Objective**: The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.

**Constraints**: Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

**Content**: The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## Descriptive Statistics

- There are 768 rows and 9 columns

**Observations**:
1. There are a total of 768 records and 9 features in the dataset.
2. Each feature can be either of integer or float dataype.
3. Some features like Glucose, Blood pressure, Insulin, BMI have zero values which represent missing data.
4. There are zero NaN values in the dataset.
5. In the outcome column, 1 represents diabetes positive and 0 represents diabetes negative.

## Data Visualization

The data visualization shows the dataset is not balanced since the number of people who have diabetes are less than the number of people who do not have diabetes.

From the correlation heatmap, we can see that there is a high correlation between Outcome and [Glucose, BMI, Age, Insulin]. We can select these features to accept input from the user and predict the outcome.

## Data Preprocessing

1. **Replacing Zero Values with NaN**: This step aims to identify and replace any zero values in certain features (columns) of the dataset with NaN (Not a Number). This is done for the features Glucose, BloodPressure, SkinThickness, Insulin, and BMI. It's a common practice to replace zero values with NaN to better handle missing data during subsequent preprocessing steps.

2. **Replacing NaN with Mean Values**: After identifying the missing values (NaN), the next step is to impute these missing values. In this case, missing values in the features Glucose, BloodPressure, SkinThickness, Insulin, and BMI are replaced with the mean values of their respective columns. Imputing missing values with mean values is a simple and commonly used method to handle missing data.

3. **Feature Scaling using MinMaxScaler**: Feature scaling is a preprocessing technique used to standardize the range of independent variables or features in the dataset. MinMaxScaler scales and translates each feature individually such that it is in the specified range, typically between 0 and 1. This ensures that all features contribute equally to the model training process and prevents features with larger scales from dominating the learning process.

4. **Selecting Features**: This step selects specific features (Glucose, Insulin, BMI, Age) that will be used as input for the machine learning model.

5. **Splitting Data**: The dataset is split into training and testing sets. This is important for evaluating the performance of the machine learning model. The training set is used to train the model, while the testing set is used to evaluate its performance.

## Data Modelling

The following machine learning algorithms were implemented:
- Logistic Regression
- K Nearest Neighbors
- Support Vector Classifier
- Naive Bayes
- Decision Tree
- Random Forest

## Model Evaluation

**Initial Model Performance**:
- Logistic Regression: 72.08%
- K Nearest neighbors: 78.57%
- Support Vector Classifier: 73.38%
- Naive Bayes: 71.43%
- Decision tree: 68.18%
- Random Forest: 75.97%

From the above comparison, we can observe that K Nearest neighbors gets the highest accuracy of 78.57%

**Classification Report Analysis**:
- Class 0 has a precision of 0.81, recall of 0.87, and F1-score of 0.84. This means that the model did a good job of classifying class 0 instances.
- Class 1 has a lower precision of 0.72, recall of 0.63, and F1-score of 0.67. This means that the model did not perform as well on class 1 instances.
- The accuracy is 0.77, which means that the model was correct 77% of the time.

## Alternate Preprocessing

To address class imbalance, we balanced the dataset by oversampling the dataset.

**Model Evaluation with Balanced Data**:
- Logistic Regression: 74.0%
- K Nearest neighbors: 76.5%
- Support Vector Classifier: 73.0%
- Naive Bayes: 70.0%
- Decision tree: 74.0%
- Random Forest: 78.5%

From the above comparison, we can observe that Random Forest gets the highest accuracy of 78.5%

**Classification Report for Balanced Model**:
- Class 0 has a precision of 0.82, recall of 0.68, and F1-score of 0.74. This means that the model did a good job of identifying a high proportion of class 0 instances that it predicted, but it missed a significant number of actual class 0 cases.
- Class 1 has a lower precision of 0.73, recall of 0.85, and F1-score of 0.78. This means that the model did a good job of recalling most of the class 1 cases but it also predicted a high number of false positives.
- The accuracy is 0.77, which means that the model was correct 77% of the time.

## Conclusion

While both models have the same accuracy, the First model performs better in terms of F1-scores for both classes. The first model has a better balance between precision and recall for Class 0 and a higher precision for Class 1. Therefore, the First confusion matrix suggests better overall performance.

## Technologies Used

- Python
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Matplotlib - Data visualization
- Seaborn - Statistical data visualization
- Scikit-learn - Machine learning algorithms and evaluation metrics
- 
## Screenshots
<img width="803" height="590" alt="image" src="https://github.com/user-attachments/assets/973b185c-7161-4bb5-8864-ba4939b13747" />
<img width="896" height="649" alt="image" src="https://github.com/user-attachments/assets/518f949e-f248-4108-a5b9-88b54f1dd177" />
<img width="585" height="314" alt="image" src="https://github.com/user-attachments/assets/106a0389-df99-490a-8bfa-f1a50e1e1ffd" />
<img width="710" height="796" alt="image" src="https://github.com/user-attachments/assets/79e76976-31a3-41d2-85fe-d19c758548c7" />


## Potential Applications

- Early diabetes detection
- Healthcare decision support systems
- Risk assessment tools for medical professionals
- Patient screening and monitoring
- Personalized healthcare strategies


