from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Handle missing data
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Create pipeline for numerical features
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Replace missing values with median
    ('scaler', StandardScaler())  # Standardize numerical features
])

# Create pipeline for categorical features
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with most frequent value
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine numerical and categorical pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# Fit and transform data using the preprocessor pipeline
transformed_data = preprocessor.fit_transform(data)


from sklearn.model_selection import train_test_split

# Assuming your dataset has a target variable named 'FATAL_NO'
target_column_name = 'FATAL_NO'

# Extract the target variable from the original dataset
target_variable = data[target_column_name]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(transformed_data, target_variable, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Neural Network': MLPClassifier()
}


# Check for NaN values in the target variable y_train
missing_values = y_train.isnull().sum()

if missing_values > 0:
    # If there are NaN values, handle them by imputing or removing them
    # For example, you can use SimpleImputer to replace NaN values with the median or most frequent value
    # Or you can remove rows with NaN values using dropna()
    
    # Example of handling missing values by imputing with the most frequent value
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(strategy='most_frequent')
    y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1))
    
    # Convert back to pandas Series
    y_train = pd.Series(y_train_imputed.flatten(), index=y_train.index)




# Check for NaN values in the test set target variable y_test
missing_values_test = y_test.isnull().sum()

if missing_values_test > 0:
    # If there are NaN values, handle them by imputing or removing them
    # For example, you can use SimpleImputer to replace NaN values with the median or most frequent value
    # Or you can remove rows with NaN values using dropna()
    
    # Example of handling missing values by imputing with the most frequent value
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(strategy='most_frequent')
    y_test_imputed = imputer.fit_transform(y_test.values.reshape(-1, 1))
    
    # Convert back to pandas Series
    y_test = pd.Series(y_test_imputed.flatten(), index=y_test.index)


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Initialize dictionaries to store evaluation metrics
evaluation_metrics = {}

# Loop through each model and calculate evaluation metrics
for name, model in best_models.items():
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    
    # Store evaluation metrics in the dictionary
    evaluation_metrics[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Confusion Matrix': confusion}


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)  # Increase max_iter to 1000
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

