from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import datasets
import pickle

iris = datasets.load_iris()
x = iris.data
y = iris.target

indexs = np.arange(len(x))
np.random.shuffle(indexs)
x = x[indexs]
y = y[indexs]

split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

pred = model.predict(x_test)
accuracy = np.mean(pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

with open("model/model.pkl", 'wb') as f:
    pickle.dump(model, f)

print("The model was successfully trained and saved to 'model/model.pkl'.")
