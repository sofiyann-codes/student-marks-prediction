# Student Marks Prediction Using Machine Learning

A beginner-friendly AI/ML mini project that predicts a student's marks based on number of study hours, using the **Linear Regression** algorithm.

Built during my AI/ML Internship at **Xtragrad Pvt Ltd**.

---

## What This Project Does

You enter how many hours a student studied → the model predicts their expected marks.

Example:
```
Enter number of study hours: 6
→ If you study for 6.0 hours, you are likely to score: 64.7 / 100
```

---

## How It Works

1. **Dataset** — 20 sample students with study hours and marks
2. **Split** — 80% used to train the model, 20% used to test it
3. **Train** — Linear Regression finds the best straight line through the data
4. **Evaluate** — R² Score and Mean Absolute Error check the accuracy
5. **Predict** — User inputs hours, model outputs predicted marks
6. **Graph** — Scatter plot with regression line saved as PNG

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Load and manage the dataset |
| Scikit-learn | Linear Regression algorithm + metrics |
| Matplotlib | Draw the graph |

---

## Project Structure

```
XtraGrad-miniproject/
│
├── student_marks_prediction.py   # Main project file
├── marks_prediction_plot.png     # Graph generated after running
└── README.md                     # This file
```

---

## How to Run

**Step 1 — Install the libraries**
```bash
python -m pip install pandas scikit-learn matplotlib
```

**Step 2 — Run the project**
```bash
python student_marks_prediction.py
```

**Step 3 — Enter study hours when asked**
```
Enter number of study hours: 7
→ If you study for 7.0 hours, you are likely to score: 73.0 / 100
```

---

## Model Accuracy

| Metric | Value | What it means |
|---|---|---|
| R² Score | ~0.97 | Model is 97% accurate |
| Mean Absolute Error | ~3.2 | Prediction is off by ~3 marks on average |

---

## Key Concepts Learned

- What supervised machine learning is
- How to prepare and split a dataset
- How Linear Regression works
- How to evaluate a model (R² Score, MAE)
- How to make predictions using a trained model
- How to visualize data with Matplotlib

---

## Author

**Mohammed Sofiyan Pasha**  
B.Tech Information Technology — KITSW, Warangal  
AI/ML Intern @ Xtragrad Pvt Ltd
