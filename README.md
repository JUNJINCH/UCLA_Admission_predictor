# UCLA Admission Predictor

This project is a Streamlit-based web application that predicts a student's chance of being admitted to UCLA's graduate program using a trained **MLP (Multi-Layer Perceptron)** neural network model.

It takes a student's academic profile as input (e.g., GRE, TOEFL, CGPA) and outputs a binary admission result: **Admitted** or **Not Admitted**, with performance metrics and a confusion matrix visualization.


## Features
- Predicts **admission chances** using historical student data and a trained neural network.
- Inputs:
  - GRE Score
  - TOEFL Score
  - University Rating
  - SOP & LOR Strength
  - Undergraduate GPA
  - Research Experience
- Output: Admission Result + Recommendation
- Performance Metrics: Accuracy, Precision, Recall
- Confusion Matrix Visualization
- Modular code structure for maintainability
- Integrated logging and error handling

## How to Run

1. Install dependencies: pip install -r requirements.txt
2. Train the model: python train_model_script.py
3. Launch the app: streamlit run app.py