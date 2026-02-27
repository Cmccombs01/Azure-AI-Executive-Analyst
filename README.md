# 📊 Azure AI Executive Analyst

A dual-layered AI platform designed to bridge the gap between predictive data science and actionable business strategy. This tool predicts customer churn using Machine Learning and instantly generates automated, high-level retention strategies using **Azure OpenAI (GPT-4o-mini)**.

## 🚀 The Elevator Pitch
Most business intelligence tools stop at predicting *what* will happen. The Azure AI Executive Analyst takes it a step further by autonomously recommending *what to do about it*. By combining a trained Machine Learning classification model with the reasoning capabilities of Azure OpenAI, this application acts as a virtual data scientist and strategic advisor for executive teams.

---

## 🛠️ Core Capabilities

* **🔮 Predictive Churn Modeling:** Utilizes a trained Machine Learning model (`churn_model.pkl`) to analyze customer data and accurately predict which users are at the highest risk of canceling their service.
* **🧠 AI Strategic Advisor:** Feeds high-risk customer profiles directly into Azure OpenAI (GPT-4o-mini) to automatically generate personalized, high-level retention strategies and marketing interventions.
* **📊 End-to-End Pipeline:** Includes scripts for generating synthetic customer data (`generate_data.py`), training and exporting the ML model (`train_model.py`), and running the final interactive application (`app.py`).

---

## 💻 Tech Stack
* **LLM Engine:** Azure OpenAI (GPT-4o-mini)
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Application Framework:** Python (app.py)

---

## 📂 Project Structure

* `app.py`: The main application file that integrates the ML model and Azure OpenAI for real-time analysis.
* `train_model.py`: The data science script used to clean the data, engineer features, and train the predictive churn model.
* `generate_data.py`: A utility script used to create the synthetic `customer_data.csv` dataset for training and testing.
* `churn_model.pkl` & `model_features.pkl`: The serialized, pre-trained Machine Learning model and its required feature mappings.
* `customer_data.csv`: The dataset utilized for model training and active predictions.

---

## ⚙️ How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/Cmccombs01/Azure-AI-Executive-Analyst.git](https://github.com/Cmccombs01/Azure-AI-Executive-Analyst.git)
cd Azure-AI-Executive-Analyst
