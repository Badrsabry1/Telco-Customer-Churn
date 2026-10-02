\# Telco Customer Churn Prediction



\## Project Description



This project predicts whether a telecom customer is likely to churn.



The project uses the Telco Customer Churn dataset and Logistic Regression as the machine learning model.



\## Dataset



The dataset contains customer information such as:



\- Gender

\- Senior Citizen

\- Partner

\- Dependents

\- Tenure

\- Phone Service

\- Internet Service

\- Contract

\- Payment Method

\- Monthly Charges

\- Total Charges



The target variable is:



\- Churn



\## Data Preprocessing



The project includes:



\- Converting TotalCharges to numeric values

\- Handling missing TotalCharges values

\- Converting Churn from Yes/No to 1/0

\- Removing customerID

\- Encoding categorical features

\- Scaling numerical features

\- Splitting the data into training and testing sets



\## Machine Learning Model



The model used in this project is:



Logistic Regression



The dataset is divided into:



\- 80% Training Data

\- 20% Testing Data



\## Results



The model achieved the following results:



| Metric | Score |

|---|---:|

| Accuracy | 0.8055 |

| Precision | 0.6572 |

| Recall | 0.5588 |

| F1 Score | 0.6040 |

| ROC-AUC | 0.8421 |



\## Project Files



\- `Teleco.py` - Main Python code

\- `churn\_predictions.csv` - Model predictions

\- `churn\_model.pkl` - Saved trained model

\- `confusion\_matrix.png` - Confusion matrix

\- `roc\_curve.png` - ROC curve

\- `probability.png` - Churn probability distribution

\- `feature\_importance.png` - Important model features



\## Technologies



\- Python

\- Pandas

\- Scikit-learn

\- Matplotlib

\- Joblib

