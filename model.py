from sklearn.linear_model import LogisticRegression
import pandas as pd

data = pd.read_csv("kindey stone urine analysis.csv")
del data["cond"]
del data["ph"]

training_X = data.sample(frac=0.7)
validation_X = data.drop(training_X.index)
training_Y = training_X["target"]
validation_Y = validation_X["target"]
del training_X["target"]
del validation_X["target"]

model = LogisticRegression().fit(training_X, training_Y)
accuracy = model.score(validation_X, validation_Y)
print("The model accuracy is " + str(accuracy))

