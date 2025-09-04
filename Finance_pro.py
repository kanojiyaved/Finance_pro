import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Generate synthetic dataset
np.random.seed(42)
num_samples = 500
income = np.random.randint(30000, 150000, num_samples)
expenses = np.random.randint(10000, 80000, num_samples)
savings = income - expenses
market_trend_score = np.random.uniform(-1, 1, num_samples)

# Define risk tolerance (randomly for training)
risk_tolerance = np.random.randint(1, 6, num_samples)

# Investment Allocation Formulas
stocks = (risk_tolerance * 0.2 + market_trend_score * 0.5) * savings * 0.3
bonds = ((5 - risk_tolerance) * 0.2 + (1 - market_trend_score) * 0.3) * savings * 0.4
crypto = (risk_tolerance * 0.4 + market_trend_score * 0.7) * savings * 0.2
real_estate = (savings * 0.1) + (risk_tolerance * 5000)

# Expected Annual Returns
expected_returns = {
    "Stocks": 0.12,   # 12% annual return
    "Bonds": 0.06,    # 6% annual return
    "Crypto": 0.08,   # 8% annual return
    "Real_Estate": 0.07  # 7% annual return
}

# Store data in a DataFrame
df = pd.DataFrame({
    'Income': income,
    'Expenses': expenses,
    'Savings': savings,
    'Risk_Tolerance': risk_tolerance,
    'Market_Trend_Score': market_trend_score,
    'Stocks': stocks,
    'Bonds': bonds,
    'Crypto': crypto,
    'Real_Estate': real_estate
})

# Define features and target variables
X = df[['Income', 'Expenses', 'Savings', 'Risk_Tolerance', 'Market_Trend_Score']]
Y = df[['Stocks', 'Bonds', 'Crypto', 'Real_Estate']]

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Function to Get Best Investment Strategy
def get_best_investment(investment_amount, investment_duration, risk_tolerance):
    market_trend = float(input("Enter market trend score (-1.0 to +1.0, where +1.0 = Strong Growth, -1.0 = Recession): "))

    # Create a DataFrame for the new user
    new_user_df = pd.DataFrame([[investment_amount, investment_amount * 0.5, investment_amount * 0.5, risk_tolerance, market_trend]],
                               columns=['Income', 'Expenses', 'Savings', 'Risk_Tolerance', 'Market_Trend_Score'])

    # Predict investment allocation
    investment_plan = rf_model.predict(new_user_df)[0]

    # Map predictions to asset names
    investment_options = ['Stocks', 'Bonds', 'Crypto', 'Real_Estate']
    investment_results = {option: investment_plan[i] for i, option in enumerate(investment_options)}

    # Get the best investment option
    best_investment = max(investment_results, key=investment_results.get)
    best_amount = investment_results[best_investment]

    # Calculate expected return after the given time period
    annual_return = expected_returns[best_investment]
    total_return = best_amount * ((1 + annual_return) ** investment_duration)  # Compound interest formula

    # Print investment recommendation
    print("\n🔹 AI Investment Recommendation 🔹")
    print(f"✅ Best Investment Option: {best_investment}")
    print(f"💰 Recommended Investment Amount: ${round(best_amount, 2)}")
    print(f"⏳ Expected Return after {investment_duration} years: ${round(total_return, 2)}")
    print(f"📈 Total Growth: {round(((total_return / best_amount) - 1) * 100, 2)}%")

# Take user inputs
investment_amount = float(input("Enter the total amount you want to invest: $"))
investment_duration = int(input("Enter your investment duration (in years): "))
risk_tolerance = int(input("Enter your risk tolerance (1-5, where 1 = Low Risk, 5 = High Risk): "))

# Run the AI model
get_best_investment(investment_amount, investment_duration, risk_tolerance)
