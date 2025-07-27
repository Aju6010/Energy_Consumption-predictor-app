# ⚡ Energy Consumption Predictor

This is a machine learning-powered web application that predicts the **energy consumption** of a building based on environmental and operational factors.
Built using **Linear Regression**, the app takes user input for features like temperature, humidity, occupancy, and usage settings, and predicts the energy usage.

🚀 **Live App:**  
👉 [Try it here](https://energyconsumption-predictor-app-kuesamjlafjegmlltujzxx.streamlit.app/)

---

## 🔍 Features

- Predicts energy consumption based on:
  - Temperature
  - Humidity
  - Square footage
  - Occupancy
  - Renewable energy usage
  - HVAC and lighting usage
  - Day of the week
- Real-time input through an interactive Streamlit UI
- Auto-handles derived features like:
  - HVAC–Temperature interaction
  - People per square foot
  - Holiday detection based on weekends

---

## 🧠 Model

- **Algorithm**: Linear Regression (from scikit-learn)
- **Preprocessing**: StandardScaler used to scale inputs
- **Model Artifacts**:
  - `linear_regression_model.pkl` – Trained model
  - `scaler.pkl` – Feature scaler used during training

---

## 📁 Project Structure

```
energy_consumption-predictor-app/
│
├── app.py                      # Streamlit web app
├── train_model.py              # Model training script
├── linear_regression_model.pkl # Saved ML model
├── scaler.pkl                  # Saved scaler
├── dataset.csv                 # Energy consumption dataset 
└── README.md                   # This file
```

---

## 🛠️ How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/energy_consumption-predictor-app.git
   cd energy_consumption-predictor-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📦 Requirements

```
pandas
scikit-learn
streamlit
joblib
```

You can save these to a `requirements.txt` file with:

```bash
pip freeze > requirements.txt
```

---

## 💡 Example Use Cases

- Smart building management systems
- Estimating daily energy bills
- Simulating energy savings based on user behavior

---

## 📣 Acknowledgments

Developed as part of an AI/ML learning project.  
Feel free to contribute or suggest improvements!

---

## 📜 License

MIT License – free to use and modify.
