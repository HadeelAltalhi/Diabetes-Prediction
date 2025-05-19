import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             precision_recall_curve, 
                            precision_score, recall_score, f1_score)

# Load the dataset
data = pd.read_csv('Original data/Dataset of Diabetes.csv')

# Print data properties
print("\nDataset Shape:", data.shape)
print("\nNumber of Samples:", len(data))
print("\nFeature Names:", data.columns.tolist())
print("\nData Description:\n", data.describe())

# Encode the target variable
le = LabelEncoder()
data['CLASS'] = le.fit_transform(data['CLASS'])

# Define features & target
X = data.drop(['ID', 'No_Pation', 'CLASS'], axis=1)
y = data['CLASS']

# Convert gender to numerical
X['Gender'] = X['Gender'].map({'M': 1, 'F': 0})

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('Preprocessed data/X.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('Preprocessed data/X_test.csv', index=False)
y_train.to_csv('Preprocessed data/Y.csv', index=False)
y_test.to_csv('Preprocessed data/Y_test.csv', index=False)

# Feature Visualization
X.hist(figsize=(12, 8))
plt.title("Featueres Distributions")
plt.tight_layout()
plt.savefig('Featueres Distributions.png')
plt.show()

sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('Correlation Matrix.png')
plt.show()

sns.pairplot(data, hue='CLASS')
plt.tight_layout()
plt.savefig('Feature Pairplot.png')
plt.show()

# Initialize Models
models = {
    'LogisticRegression': LogisticRegression(),
    'SVM': SVC(probability=True, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'GaussianNB': GaussianNB(),
    'ANN': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

results = {
    'Model': [],
    'Accuracy': [],
    'Precision (Macro)': [],
    'Recall (Macro)': [],
    'F1 (Macro)': []
}

# Create the main figure for subplots
plt.figure(figsize=(20, 20))
n_row = 3
n_col = 3

# Train and evaluate models
for idx, (name, model) in enumerate(models.items(), 1):
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics (using macro-average for multiclass)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Precision (Macro)'].append(precision)
    results['Recall (Macro)'].append(recall)
    results['F1 (Macro)'].append(f1)
    
    # Save predictions
    pd.DataFrame(y_pred, columns=['Prediction']).to_csv(f'Results/prediction_{name}.csv', index=False)
    
    # Print classification report
    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1 (Macro): {f1:.4f}")
   

    # Get probabilities for all classes
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)
    else:
        # For models without predict_proba (like linear SVM without probability=True)
        y_proba = np.zeros((len(y_test), len(np.unique(y_test))))
        for class_idx in range(len(np.unique(y_test))):
            y_proba[:, class_idx] = (y_pred == class_idx).astype(int)

    # Create subplot for this model
    plt.subplot(n_row, n_col, idx)
    colors = ['blue', 'green', 'red']  # Adjust colors based on number of classes
    
    for class_idx, color in zip(range(len(np.unique(y_test))), colors):
        # Check if there are any positive samples for this class
        if np.sum(y_test == class_idx) > 0:
            precision_curve, recall_curve, _ = precision_recall_curve(
                (y_test == class_idx).astype(int), 
                y_proba[:, class_idx]
            )
            plt.plot(
                recall_curve, 
                precision_curve, 
                color=color,
                lw=2,
                label=f'Class {class_idx}'
            )
        else:
            # Skip plotting for classes with no positive samples
            print(f"Warning: No positive samples found for class {class_idx} in {name}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(name)
    plt.legend(loc='best')
    plt.grid(True)

# Adjust layout and save/show the figure
plt.tight_layout()
plt.savefig('All_Models_Precision_Recall.png')
plt.show()

# Create a summary plot of macro-averaged metrics
results_df = pd.DataFrame(results)
plt.figure(figsize=(14, 6))
sns.barplot(data=results_df.melt(id_vars='Model'), x='Model', y='value', hue='variable')
plt.title('Model Performance Comparison (Macro-Averaged Metrics)')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Model Performance Comparison.png')
plt.show()


# --- Model Comparison (Bar Plot) ---
results_df = pd.DataFrame(results)
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Model', palette='viridis', legend=False)
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.savefig('model_accuracy_barplot.png')
plt.show()

# Create a figure for confusion matrices
plt.figure(figsize=(20, 15))
n_row = 3
n_col = 3

for idx, (name, model) in enumerate(models.items(), 1):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create confusion matrix subplot
    plt.subplot(n_row, n_col, idx)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        cbar=False
    )
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('Confusion Matrices.png')
plt.show()

# --- Performance Metrics Heatmap ---
plt.figure(figsize=(12, 8))

# Select the metrics we want to visualize
metrics_df = results_df.set_index('Model')[['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']]

# Create the heatmap
sns.heatmap(
    metrics_df,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    linewidths=.5,
    cbar_kws={'label': 'Score'}
)

plt.title('Model Performance Metrics Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('Model_Performance_Heatmap.png')
plt.show()
