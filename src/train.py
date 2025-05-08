from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
iris=load_iris()
x_train,y_train,x_test,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=42)


mod=LogisticRegression()
mod.fit(x_train,y_train)
y_pred=mod.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {acc:.2f}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(mod, f)
