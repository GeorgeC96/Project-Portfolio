import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# File Path
file_path = 'IBM Employee Attrition.csv'

# Load and preprocess data
df = pd.read_csv(file_path)
df = df.drop_duplicates()
df_encoded = pd.get_dummies(df, columns=[
    'Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 
    'MaritalStatus', 'OverTime'
])
df_standardized = df_encoded.copy()
numerical_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]
scaler = StandardScaler()
df_standardized[numerical_columns] = scaler.fit_transform(df_standardized[numerical_columns])
for col in df_encoded.select_dtypes(include='bool').columns:
    df_encoded[col] = df_encoded[col].astype(int)
for col in df_standardized.select_dtypes(include='bool').columns:
    df_standardized[col] = df_standardized[col].astype(int)

# Define relevant variables
relevant_vars = [
    'Attrition_Yes', 'JobSatisfaction', 'OverTime_Yes', 'DistanceFromHome', 'MonthlyIncome', 
    'Age', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'YearsAtCompany'
]
df_encoded_relevant = df_encoded[relevant_vars]
df_standardized_relevant = df_standardized[relevant_vars]

# Streamlit App
st.title('IBM Employee Attrition Analysis')

# Create tabs for different sections
tabs = st.tabs(["Overview", "Correlation Analysis", "Distributions", "Countplots", "Boxplots", "Model Performance"])

with tabs[0]:
    st.header('Overview')
    st.write("This dashboard provides an analysis of employee attrition data. It includes correlation analysis, distribution plots, countplots, and boxplots to visualize the data and model performance metrics.")

with tabs[1]:
    st.header('Correlation Matrix (Encoded Data)')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_encoded_relevant.corr().round(2), annot=True, cmap='coolwarm', annot_kws={"size": 8}, ax=ax)
    st.pyplot(fig)

    st.header('Correlation Matrix (Standardized Data)')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_standardized_relevant.corr().round(2), annot=True, cmap='coolwarm', annot_kws={"size": 8}, ax=ax)
    st.pyplot(fig)

with tabs[2]:
    st.header('Distributions by Attrition')
    key_numerical_columns = [
        'DistanceFromHome', 'JobSatisfaction', 'MonthlyIncome', 'YearsAtCompany'
    ]
    num_columns = len(key_numerical_columns)
    num_rows = (num_columns + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, num_rows * 5))
    axes = axes.flatten()
    for i, column in enumerate(key_numerical_columns):
        sns.histplot(df_encoded, x=column, hue='Attrition_Yes', multiple='stack', bins=30, ax=axes[i])
        axes[i].set_title(f'Distribution of {column} by Attrition')
    for j in range(num_columns, len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

with tabs[3]:
    st.header('Countplots')
    countplot_columns = ['BusinessTravel_Travel_Rarely', 'BusinessTravel_Travel_Frequently', 'OverTime_Yes']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, column in enumerate(countplot_columns):
        sns.countplot(x=column, data=df_encoded, ax=axes[i])
        axes[i].set_title(f'Countplot of {column}')
    fig.delaxes(axes[-1])  # Remove the extra subplot
    st.pyplot(fig)

with tabs[4]:
    st.header('Boxplots')
    boxplot_columns = ['DistanceFromHome', 'MonthlyIncome', 'Age', 'YearsAtCompany']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, column in enumerate(boxplot_columns):
        sns.boxplot(x='Attrition_Yes', y=column, data=df_encoded, ax=axes[i])
        axes[i].set_title(f'Boxplot of {column} by Attrition')
    st.pyplot(fig)

with tabs[5]:
    st.header('Model Performance')
    features = [
        'Age', 'DistanceFromHome', 'JobSatisfaction', 'MonthlyIncome', 'YearsAtCompany',
        'OverTime_Yes', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely'
    ]
    target = 'Attrition_Yes'
    features = [col for col in features if col in df_encoded.columns]
    X = df_encoded[features]
    y = df_encoded[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.subheader(f"{name} (Without Class Weights) Results:")
        st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        st.text(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Models with class weights
    models_weighted = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'Support Vector Machine': SVC(class_weight='balanced', random_state=42)
    }

    for name, model in models_weighted.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.subheader(f"{name} (With Class Weights) Results:")
        st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        st.text(f"Classification Report:\n{classification_report(y_test, y_pred)}")
