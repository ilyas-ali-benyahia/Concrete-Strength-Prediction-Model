import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load and preprocess data
df = pd.read_excel('data.xlsx')
df.columns = df.columns.str.strip()
df['Age'] = df['Age'].astype(str).str.extract('(\d+)').astype(float)
df['Cure'] = df['Cure'].replace('Air', 'Site')

input_cols = ['Element type', 'Cure', 'Age', 'Rebound', 'UPV (km/s)']
output_col = 'Strength (MPa)'
categorical_cols = ['Element type', 'Cure']

X = df[input_cols]
y = df[output_col]

model_random_state = 42
iterations = 200

all_combined = []
metrics = {
    'iteration': [],
    'r2_rebound': [], 'rmse_rebound': [], 'sd_rebound': [],
    'r2_upv': [], 'rmse_upv': [], 'sd_upv': [],
    'r2_both': [], 'rmse_both': [], 'sd_both': [],
    'r2_rf': [], 'rmse_rf': [], 'sd_rf': [],
    'points_outside_10pct': []  # Track points outside 10% error
}

# List to store all test data for final analysis
all_test_data = []

for i in range(iterations):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i)

    # Random Forest model
    model_rf = Pipeline([
        ('preprocessor', ColumnTransformer(
            [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
            remainder='passthrough')),
        ('regressor', RandomForestRegressor(random_state=model_random_state))
    ])
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    # Calculate percentage of points outside 10% error
    percent_error = np.abs((y_test - y_pred_rf) / y_test) * 100
    points_outside = sum(percent_error > 10)
    metrics['points_outside_10pct'].append(points_outside / len(y_test) * 100)

    # Extract numeric features for linear models
    X_train_numeric = X_train[['Rebound', 'UPV (km/s)']]
    X_test_numeric = X_test[['Rebound', 'UPV (km/s)']]

    # Linear Regression: Rebound only
    model_rebound = LinearRegression().fit(X_train_numeric[['Rebound']], y_train)
    pred_rebound = model_rebound.predict(X_test_numeric[['Rebound']])

    # Linear Regression: UPV only
    model_upv = LinearRegression().fit(X_train_numeric[['UPV (km/s)']], y_train)
    pred_upv = model_upv.predict(X_test_numeric[['UPV (km/s)']])

    # Linear Regression: Both
    model_both = LinearRegression().fit(X_train_numeric, y_train)
    pred_both = model_both.predict(X_test_numeric)

    # Calculate metrics
    metrics['iteration'].append(i+1)

    for model_name, preds in zip(
        ['rebound', 'upv', 'both', 'rf'],
        [pred_rebound, pred_upv, pred_both, y_pred_rf]):
        metrics[f'r2_{model_name}'].append(r2_score(y_test, preds))
        metrics[f'rmse_{model_name}'].append(np.sqrt(mean_squared_error(y_test, preds)))
        metrics[f'sd_{model_name}'].append(np.std(y_test - preds))

    # Save training/testing subsets
    train_info = X_train_numeric.copy()
    train_info['Strength'] = y_train
    train_info.columns = [f'Rebound_train_{i+1}', f'UPV_train_{i+1}', f'Strength_train_{i+1}']

    test_info = X_test_numeric.copy()
    test_info['Strength'] = y_test
    test_info['Pred'] = y_pred_rf
    test_info['Error_Pct'] = percent_error
    test_info['Outside_10pct'] = percent_error > 10
    test_info.columns = [f'Rebound_test_{i+1}', f'UPV_test_{i+1}', f'Strength_test_{i+1}',
                         f'Pred_{i+1}', f'Error_Pct_{i+1}', f'Outside_10pct_{i+1}']

    # Store for plot later
    iteration_test_data = pd.DataFrame({
        'Iteration': i+1,
        'Actual': y_test.values,
        'Predicted': y_pred_rf,
        'Outside_10pct': percent_error > 10
    })
    all_test_data.append(iteration_test_data)

    combined = pd.concat([train_info.reset_index(drop=True), test_info.reset_index(drop=True)], axis=1)
    all_combined.append(combined)

# Save combined prediction data
final_df = pd.concat(all_combined, axis=1)
final_df.to_excel("train_test_predictions_combined.xlsx", index=False)

# Save metrics
results_df = pd.DataFrame(metrics)
results_df.to_excel("linear_rf_model_metrics_200.xlsx", index=False)

# Combine all test data for plotting
all_test_df = pd.concat(all_test_data, ignore_index=True)

# Create plot for one specific iteration (iteration 1)
def plot_iteration(iteration_num=1):
    iteration_data = all_test_df[all_test_df['Iteration'] == iteration_num]

    plt.figure(figsize=(10, 8))

    # Plot points
    plt.scatter(iteration_data['Actual'], iteration_data['Predicted'],
                c=iteration_data['Outside_10pct'].map({True: 'red', False: 'blue'}),
                alpha=0.7)

    # Perfect prediction line
    max_val = max(iteration_data['Actual'].max(), iteration_data['Predicted'].max())
    min_val = min(iteration_data['Actual'].min(), iteration_data['Predicted'].min())
    padding = (max_val - min_val) * 0.1
    line_range = np.linspace(min_val - padding, max_val + padding, 100)
    plt.plot(line_range, line_range, 'k--', label='Perfect prediction')

    # Add 10% error lines
    plt.plot(line_range, line_range * 1.1, 'r--', alpha=0.5, label='+10% error')
    plt.plot(line_range, line_range * 0.9, 'r--', alpha=0.5, label='-10% error')

    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.title(f'Iteration {iteration_num}: Actual vs Predicted Strength with 10% Error Bands')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add count of points outside 10% error
    outside_count = iteration_data['Outside_10pct'].sum()
    total_count = len(iteration_data)
    outside_percent = outside_count / total_count * 100

    plt.annotate(f'Points outside 10% error: {outside_count}/{total_count} ({outside_percent:.1f}%)',
                xy=(0.05, 0.05), xycoords='axes fraction', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'iteration_{iteration_num}_strength_prediction.png', dpi=300)
    plt.close()

    return outside_count, total_count, outside_percent

# Function to plot all iterations in one figure
def plot_all_iterations_summary():
    plt.figure(figsize=(12, 10))

    # Group by iteration and plot average results
    iteration_summary = all_test_df.groupby('Iteration').agg({
        'Actual': 'mean',
        'Predicted': 'mean',
        'Outside_10pct': 'mean'
    }).reset_index()

    # Create a plot showing percentage of points outside 10% for each iteration
    plt.bar(iteration_summary['Iteration'], iteration_summary['Outside_10pct'] * 100)
    plt.xlabel('Iteration')
    plt.ylabel('Percentage of Points Outside 10% Error (%)')
    plt.title('Percentage of Points Outside 10% Error Across All Iterations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('points_outside_10pct_summary.png', dpi=300)
    plt.close()

    # Create an overall scatter plot combining data from all iterations
    plt.figure(figsize=(12, 10))
    plt.scatter(all_test_df['Actual'], all_test_df['Predicted'],
                c=all_test_df['Outside_10pct'].map({True: 'red', False: 'blue'}),
                alpha=0.3)

    # Perfect prediction line
    max_val = max(all_test_df['Actual'].max(), all_test_df['Predicted'].max())
    min_val = min(all_test_df['Actual'].min(), all_test_df['Predicted'].min())
    padding = (max_val - min_val) * 0.1
    line_range = np.linspace(min_val - padding, max_val + padding, 100)
    plt.plot(line_range, line_range, 'k--', label='Perfect prediction')

    # Add 10% error lines
    plt.plot(line_range, line_range * 1.1, 'r--', alpha=0.5, label='+10% error')
    plt.plot(line_range, line_range * 0.9, 'r--', alpha=0.5, label='-10% error')

    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.title('All Iterations: Actual vs Predicted Strength with 10% Error Bands')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add count of points outside 10% error
    outside_count = all_test_df['Outside_10pct'].sum()
    total_count = len(all_test_df)
    outside_percent = outside_count / total_count * 100

    plt.annotate(f'Points outside 10% error: {outside_count}/{total_count} ({outside_percent:.1f}%)',
                xy=(0.05, 0.05), xycoords='axes fraction', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('all_iterations_strength_prediction.png', dpi=300)
    plt.close()

    return outside_count, total_count, outside_percent

# Plot first iteration as an example
outside_count, total_count, outside_percent = plot_iteration(iteration_num=1)
print(f"Iteration 1: {outside_count}/{total_count} points ({outside_percent:.1f}%) are outside the 10% error bands")

# Plot results for all iterations
all_outside_count, all_total_count, all_outside_percent = plot_all_iterations_summary()
print(f"All iterations: {all_outside_count}/{all_total_count} points ({all_outside_percent:.1f}%) are outside the 10% error bands")

# Generate Excel summary of points outside 10% for each iteration
iteration_summary = all_test_df.groupby('Iteration').agg({
    'Outside_10pct': ['sum', 'count', lambda x: sum(x)/len(x)*100]
}).reset_index()

iteration_summary.columns = ['Iteration', 'Points_Outside_10pct', 'Total_Points', 'Percent_Outside']
iteration_summary.to_excel('iteration_10pct_error_summary.xlsx', index=False)

print("Analysis completed. Files saved:")
print("- train_test_predictions_combined.xlsx")
print("- linear_rf_model_metrics_200.xlsx")
print("- iteration_1_strength_prediction.png")
print("- all_iterations_strength_prediction.png")
print("- points_outside_10pct_summary.png")
print("- iteration_10pct_error_summary.xlsx")