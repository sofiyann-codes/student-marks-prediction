# I'm importing the tools I need
# pandas helps me work with data like a spreadsheet
# LinearRegression is the actual ML algorithm
# train_test_split splits my data into training and testing parts
# mean_absolute_error and r2_score help me check how good my model is
# matplotlib is for drawing the graph

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# This is my dataset
# I have 20 students - how many hours they studied and what marks they got
# In real projects this would come from a CSV file, but here I'm creating it manually

data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 0.5],
    'marks':       [20, 30, 40, 50, 60, 65, 75, 80, 88, 95,
                    25, 35, 45, 55, 62, 70, 78, 85, 92, 15]
}

# I'm converting that data into a DataFrame
# DataFrame is basically a table with rows and columns, like Excel

df = pd.DataFrame(data)

print("Here is my dataset:")
print(df.to_string(index=False))
print(f"Total students in dataset: {len(df)}")


# Now I'm separating my data into input and output
# X is the input  - study hours (what I give the model)
# y is the output - marks       (what I want the model to predict)

X = df[['study_hours']]
y = df['marks']


# I'm splitting the data - 80% for training, 20% for testing
# I train the model on 80% of data
# Then I test it on the remaining 20% it has never seen before
# This tells me if the model actually learned or just memorized

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nI used {len(X_train)} students to train the model")
print(f"I used {len(X_test)} students to test the model")


# Here I'm creating the Linear Regression model and training it
# fit() is where the actual learning happens
# The model finds the best straight line through all the data points

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel is trained!")
print(f"The slope is: {model.coef_[0]:.2f}")
print(f"The intercept is: {model.intercept_:.2f}")
print(f"So the formula is: Marks = {model.coef_[0]:.2f} x Hours + {model.intercept_:.2f}")


# Now I'm testing the model on the 20% data it never saw during training
# I compare what it predicted vs what the actual marks were

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\nHow accurate is my model?")
print(f"Mean Absolute Error: {mae:.2f}  (on average my prediction is off by this many marks)")
print(f"R2 Score: {r2:.2f}  (1.0 means perfect, anything above 0.9 is great)")


# These are some example predictions just to show the model working

print("\nSample predictions:")
for hours in [3, 5, 7, 9]:
    predicted_marks = model.predict([[hours]])[0]
    print(f"  If a student studies {hours} hours → predicted marks: {predicted_marks:.1f}")


# Now I'm asking the user to enter their own study hours
# The model will predict marks for whatever number they type

print("\nNow you try it!")
try:
    user_hours = float(input("Enter number of study hours: "))
    predicted = model.predict([[user_hours]])[0]
    predicted = max(0, min(100, predicted))  # keeping marks between 0 and 100
    print(f"If you study for {user_hours} hours, you are likely to score: {predicted:.1f} / 100")
except ValueError:
    print("Please enter a valid number")


# Finally I'm drawing a graph
# Blue dots = actual data points from my dataset
# Red line  = the regression line my model learned

plt.figure(figsize=(8, 5))
plt.scatter(df['study_hours'], df['marks'], color='royalblue', label='Actual student data', zorder=5)
plt.plot(df['study_hours'], model.predict(df[['study_hours']]),
         color='crimson', linewidth=2, label='Predicted line')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Student Marks Prediction using Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('marks_prediction_plot.png', dpi=150)
plt.show()
print("Graph saved as marks_prediction_plot.png")