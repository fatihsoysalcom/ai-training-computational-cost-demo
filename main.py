import time
import random

def generate_data(num_samples, true_m, true_b, noise_level=0.5):
    """Generates synthetic data for linear regression."""
    X = [i for i in range(num_samples)]
    y = [true_m * x + true_b + random.uniform(-noise_level, noise_level) for x in X]
    return X, y

def train_linear_model(X, y, learning_rate=0.0001, epochs=5000):
    """
    Simulates training a simple linear regression model using gradient descent.
    This iterative process represents the computational demand on AI infrastructure.
    """
    m = 0.0  # Initial slope
    b = 0.0  # Initial intercept
    n = len(X)

    print(f"Starting model training for {epochs} epochs...")
    start_time = time.time() # Start measuring computational time

    for epoch in range(epochs):
        # These calculations simulate the 'work' done by AI infrastructure
        # to learn from data. More complex models and larger datasets
        # require significantly more processing power (GPUs, specialized chips).
        dm_sum = 0.0
        db_sum = 0.0
        for i in range(n):
            y_pred = m * X[i] + b
            error = y_pred - y[i]
            dm_sum += error * X[i]
            db_sum += error

        # Update parameters using gradient descent
        m -= learning_rate * (dm_sum / n)
        b -= learning_rate * (db_sum / n)

        if (epoch + 1) % (epochs // 10) == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Current m: {m:.4f}, b: {b:.4f}")

    end_time = time.time() # End measuring computational time
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration:.2f} seconds.")
    # This duration highlights the need for efficient AI infrastructure
    # to reduce training time and cost, thereby accelerating ROI.

    return m, b, training_duration

def predict(X_new, m, b):
    """Makes predictions using the trained model."""
    return [m * x + b for x in X_new]

# --- Main execution ---
if __name__ == "__main__":
    # 1. Simulate data generation (representing real-world data collection)
    true_slope = 2.5
    true_intercept = 5.0
    num_data_points = 500 # More data points => more computational need
    print(f"Generating {num_data_points} synthetic data points...")
    X_data, y_data = generate_data(num_data_points, true_slope, true_intercept, noise_level=10)
    print("Data generation complete.\n")

    # 2. Train the AI model (the core computational task requiring infrastructure)
    # The 'epochs' and 'num_data_points' directly impact the computational load.
    # High-performance AI infrastructure (GPUs, specialized chips) accelerates this.
    learned_m, learned_b, duration = train_linear_model(X_data, y_data)

    print(f"\nLearned Model Parameters:")
    print(f"  Slope (m): {learned_m:.4f} (True: {true_slope})")
    print(f"  Intercept (b): {learned_b:.4f} (True: {true_intercept})")

    # 3. Make predictions (representing the 'return' or 'value' from AI investment)
    new_X_values = [50, 51, 52]
    predictions = predict(new_X_values, learned_m, learned_b)

    print(f"\nPredictions for new data points (demonstrating AI's output/value):")
    for i, x_val in enumerate(new_X_values):
        print(f"  X = {x_val}, Predicted Y = {predictions[i]:.4f}")

    # The ability to make accurate predictions is the 'return' on the investment
    # in infrastructure that enabled this training.
    print("\nThis example demonstrates that training AI models (even simple ones) requires significant computation over time.")
    print("Investing in robust AI infrastructure (like GPUs and cloud services) reduces this time and enables faster, more complex model development and deployment, leading to quicker returns.")
