import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('training_data_vt2025.csv', dtype={'ID': str}).dropna().reset_index(drop=True)

# Feature Engineering
data['is_winter'] = data['month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
data['is_spring'] = data['month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
data['is_summer'] = data['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
data['is_fall'] = data['month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
data['rush_hour'] = data['hour_of_day'].apply(lambda x: 1 if (15 <= x <= 19) else 0)
data['night_time'] = data['hour_of_day'].apply(lambda x: 1 if (x >= 20) or (x <= 7) else 0)
data['is_there_snow'] = data['snowdepth'].apply(lambda x: 1 if x > 0 else 0)
data['increase_stock'] = data['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})

# Drop unnecessary columns
columns_to_drop = ['holiday', 'hour_of_day', 'summertime', 'snowdepth', 'month', 'day_of_week', 'snow']
data.drop(columns=columns_to_drop, inplace=True)

# Normalize numerical features
normalized_cols = ['temp', 'humidity']
scaler = StandardScaler()
data[normalized_cols] = scaler.fit_transform(data[normalized_cols])

# Split dataset into train and test
seed_value = 42
data, test_data = train_test_split(data, test_size=0.2, random_state=seed_value, shuffle=False)
x_train = data.drop(columns=['increase_stock'])
y_train = data['increase_stock']
x_testing = test_data.drop(columns=['increase_stock'])
y_testing = test_data['increase_stock']

# Feature selection using RFE with Logistic Regression
logreg_selector = LogisticRegression(random_state=1, max_iter=5000)
logreg_rfe = RFE(estimator=logreg_selector, n_features_to_select=14)
x_train_logreg_selected = logreg_rfe.fit_transform(x_train, y_train)
x_testing_logreg_selected = logreg_rfe.transform(x_testing)
selected_features_logreg = x_train.columns[logreg_rfe.support_]
print(f'Selected Features for Logistic Regression: {list(selected_features_logreg)}')

# Feature selection using RFE with Random Forest
rf_selector = RandomForestClassifier(random_state=1)
rf_rfe = RFE(estimator=rf_selector, n_features_to_select=14)
x_train_rf_selected = rf_rfe.fit_transform(x_train, y_train)
x_testing_rf_selected = rf_rfe.transform(x_testing)
selected_features_rf = x_train.columns[rf_rfe.support_]
print(f'Selected Features for Random Forest: {list(selected_features_rf)}')

LDA_selector = LinearDiscriminantAnalysis()
LDA_rfe = RFE(estimator=LDA_selector, n_features_to_select=14)
x_train_LDA_selected = LDA_rfe.fit_transform(x_train, y_train)
x_testing_LDA_selected = LDA_rfe.transform(x_testing)
selected_features_LDA = x_train.columns[LDA_rfe.support_]
print(f'Selected Features for LDA: {list(selected_features_LDA)}')

# Feature selection using SelectKBest
skb = SelectKBest(score_func=f_classif, k=14)
x_train_skb_selected = skb.fit_transform(x_train, y_train)
x_testing_skb_selected = skb.transform(x_testing)
selected_features_skb = x_train.columns[skb.get_support()]
print(f'Selected Features for SelectKBest: {list(selected_features_skb)}')

# Model definitions

n_features = x_train_skb_selected.shape[1]
n_classes = len(y_train.unique())

# Calculate the maximum number of components
max_components = min(n_features, n_classes - 1)
models = {
    'Logistic Regression': (LogisticRegression(random_state=1, max_iter=5000), {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'saga']
    }),
    'LDA': (LinearDiscriminantAnalysis(), [
        {'solver': ['svd']},  # SVD does not use shrinkage or n_components
        {'solver': ['lsqr'], 'shrinkage': ['auto', 0.1, 0.5, 0.9], 'n_components': list(range(1, max_components + 1))},
        {'solver': ['eigen'], 'shrinkage': ['auto', 0.1, 0.5, 0.9], 'n_components': list(range(1, max_components + 1))}
    ]),
    'QDA': (QuadraticDiscriminantAnalysis(), {'reg_param': [0.0, 0.01, 0.1, 0.5, 1.0]}),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }),
    'Random Forest': (RandomForestClassifier(random_state=1), {
        'n_estimators': [60, 75, 100],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [5, 6, 7],
    }),
    'Dummy': (DummyClassifier(strategy='most_frequent'), {})
}

# Train and evaluate models
results = []
for name, (model, param_grid) in models.items():
    if name == 'Logistic Regression':
        x_train_selected, x_testing_selected = x_train_logreg_selected, x_testing_logreg_selected
    elif name == 'Random Forest':
        x_train_selected, x_testing_selected = x_train_rf_selected, x_testing_rf_selected
    elif name == 'LDA':
        x_train_selected, x_testing_selected = x_train_LDA_selected, x_testing_LDA_selected
    else:
        x_train_selected, x_testing_selected = x_train_skb_selected, x_testing_skb_selected
    
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
        grid_search.fit(x_train_selected, y_train)
        best_model = grid_search.best_estimator_
        print(f'{name} - Best Hyperparameters: {grid_search.best_params_}')
    else:
        best_model = model.fit(x_train_selected, y_train)
    
    y_pred = best_model.predict(x_testing_selected)
    acc = accuracy_score(y_testing, y_pred)
    precision = precision_score(y_testing, y_pred)
    recall = recall_score(y_testing, y_pred)
    f1 = f1_score(y_testing, y_pred)
    
    results.append([name, acc, precision, recall, f1])
    print(f'{name} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

# Create results dataframe
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(results_df)

# Plot accuracy and F1-score for all models
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(results_df['Model'], results_df['Accuracy'], marker='o', linestyle='-', color='#F6C1D3')
plt.xlabel('Models')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Different Models')
plt.xticks(rotation=45)

# F1-score plot
plt.subplot(1, 2, 2)
plt.plot(results_df['Model'], results_df['F1-score'], marker='o', linestyle='-', color='#D9A9D4')
plt.xlabel('Models')
plt.ylabel('Test F1 Score')
plt.title('Test F1 Score of Different Models')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
