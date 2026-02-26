import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🧠 Booting up the Machine Learning Engine...")

# 1. Load the Enterprise Data
df = pd.read_csv("customer_data.csv")

# 2. Data Preprocessing (Translating words to math for the AI)
# The AI doesn't understand "Startup" or "Enterprise", so we convert them to numbers
df = pd.get_dummies(df, columns=["Company_Size"], drop_first=True)

# Define our Features (X) and our Target (y)
# We drop CustomerID because a name doesn't cause churn, and we drop Churn_Risk because it's the answer!
X = df.drop(columns=["CustomerID", "Churn_Risk"])
y = df["Churn_Risk"]

# Save the exact column names so our Web App knows what to ask the user for later
feature_columns = X.columns.tolist()
with open("model_features.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

# 3. Split the Data (80% for studying, 20% for the final exam)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Model
print("📊 Training the Random Forest Algorithm...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Take the Final Exam and check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"✅ Model Training Complete!")
print(f"🎯 AI Prediction Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed AI Performance Report:")
print(classification_report(y_test, predictions))

# 6. Save the AI's "Brain" to a file
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved successfully as 'churn_model.pkl'. Ready for deployment!")