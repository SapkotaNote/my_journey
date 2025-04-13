from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
clz = DecisionTreeClassifier()
x = iris.data
y = iris.target
print(iris.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

clz.fit(x_train, y_train)
print(".....")

y_pred = clz.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print(f".....:{accuracy}")

speal_length = float(input("Enter Speal Lenght:"))
speal_wedth = float(input("Enter Speal Lenght:"))
petal_length = float(input("Enter Speal Lenght:"))
petal_wedth = float(input("Enter Speal Lenght:"))

inputs = [[speal_length, speal_wedth, petal_length, petal_wedth ]]

resu = clz.predict(inputs)
if resu[0] == 0:
    print("The flower is Setosa")
elif resu[0] == 1:
    print("The flower is Versicolor")
else: 
    print("The flower is Virginica")


