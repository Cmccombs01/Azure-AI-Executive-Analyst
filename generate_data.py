import pandas as pd
import numpy as np
import random

# Setting a seed for reproducibility
np.random.seed(42)

# Generate 2000 realistic SaaS customers
num_customers = 2000
data = {
    "CustomerID": [f"CUST-{1000 + i}" for i in range(num_customers)],
    "Company_Size": np.random.choice(["Startup", "Mid-Market", "Enterprise"], num_customers, p=[0.5, 0.3, 0.2]),
    "Monthly_Spend": np.random.normal(500, 150, num_customers).astype(int),
    "Support_Tickets_Last_Month": np.random.poisson(2, num_customers),
    "Days_Since_Last_Login": np.random.exponential(15, num_customers).astype(int),
    "Feature_Usage_Score": np.random.normal(60, 20, num_customers).astype(int),
}

df = pd.DataFrame(data)

# Logic for Churn (Cancellation): 
# High tickets, low usage, or no logins = High risk of leaving
df["Churn_Risk"] = 0
df.loc[(df["Support_Tickets_Last_Month"] > 4) & (df["Feature_Usage_Score"] < 50), "Churn_Risk"] = 1
df.loc[df["Days_Since_Last_Login"] > 30, "Churn_Risk"] = 1

# Add some randomness to make the Machine Learning work for it
flip_indices = np.random.choice(df.index, size=int(num_customers * 0.1), replace=False)
df.loc[flip_indices, "Churn_Risk"] = 1 - df.loc[flip_indices, "Churn_Risk"]

df.to_csv("customer_data.csv", index=False)
print("✅ Enterprise Dataset 'customer_data.csv' generated successfully!")