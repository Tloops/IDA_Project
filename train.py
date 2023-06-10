import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score

# load node2vec embeddings
emb = np.loadtxt('data/node2vec_128.emb', skiprows=1)
emb = emb[np.argsort(emb[:, 0])][:, 1:]

# load attributes of nodes
attr = np.loadtxt('data/attr.csv', delimiter=',', dtype=np.float32)
attr = attr[:, 1:]
# node_id, degree, gender, major, second_major, dormitory, high_school
print((attr==0).astype(np.int32).sum(axis=0))

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
attr_mm = min_max_scaler.fit_transform(attr)

X_mm = np.concatenate((emb, attr_mm), axis=1)
X = np.concatenate((emb, attr), axis=1)
X_mm_train, X_mm_test = X_mm[:4000], X_mm[4000:]
X_train, X_test = X[:4000], X[4000:]

# load labels of nodes
y_train = np.loadtxt('data/label_train.csv', delimiter=',', dtype=np.float32)[:, 1]
y_test = np.loadtxt('data/label_test.csv', delimiter=',', dtype=np.float32)[:, 1]
print(np.max(y_train), np.min(y_train), np.max(y_test), np.min(y_test))

# train a SVM classifier
print("=" * 50)
print("train a SVM classifier")
print("With min-max normalization")
svc = SVC(random_state=0)
svc.fit(X_mm_train, y_train)
y_predict = svc.predict(X_mm_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

print("Without min-max normalization")
svc = SVC(random_state=0)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

# train a decision tree classifier
print("=" * 50)
print("train a decision tree classifier")
print("With min-max normalization")
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_mm_train, y_train)
y_predict = dtc.predict(X_mm_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

print("Without min-max normalization")
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

# train a random forest classifier
print("=" * 50)
print("train a random forest classifier")
print("With min-max normalization")
rfc = RandomForestClassifier(n_estimators=200, max_features=X_train.shape[1]//2, max_depth=10, random_state=0)
rfc.fit(X_mm_train, y_train)
y_predict = rfc.predict(X_mm_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

print("Without min-max normalization")
rfc = RandomForestClassifier(n_estimators=200, max_features=X_train.shape[1]//2, max_depth=10, random_state=0)
rfc.fit(X_train, y_train)
y_predict = rfc.predict(X_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

# train a bagging classifier
print("=" * 50)
print("train a bagging classifier")
print("With min-max normalization")
bc = BaggingClassifier(random_state=0)
bc.fit(X_mm_train, y_train)
y_predict = bc.predict(X_mm_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

print("Without min-max normalization")
bc = BaggingClassifier(random_state=0)
bc.fit(X_train, y_train)
y_predict = bc.predict(X_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

# train a knn classifier
print("=" * 50)
print("train a knn classifier")
print("With min-max normalization")
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_mm_train, y_train)
y_predict = knn.predict(X_mm_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

print("Without min-max normalization")
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

# train a naive bayes classifier
print("=" * 50)
print("train a naive bayes classifier")
print("With min-max normalization")
gnb = GaussianNB()
gnb.fit(X_mm_train, y_train)
y_predict = gnb.predict(X_mm_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

print("Without min-max normalization")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_predict = gnb.predict(X_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))
