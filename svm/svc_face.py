import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import time
import seaborn as sbn
from sklearn.metrics import confusion_matrix

faces_data = fetch_lfw_people(min_faces_per_person=60)
# print(faces_data.target_names, '\n', faces_data.images.shape)
# fig, ax = plt.subplots(3, 5)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces_data.images[i], cmap='bone')
#     axi.set(xticks=[], yticks=[], xlabel=faces_data.target_names[faces_data.target[i]])

pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
Xtrain, Xtest, ytrain, ytest = train_test_split(faces_data.data, faces_data.target, random_state=42)
param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
start_time = time.time()
grid.fit(Xtrain, ytrain)
print('耗时:{}'.format(time.time() - start_time))
print(grid.best_params_)
best_model = grid.best_estimator_
yfit = best_model.predict(Xtest)
fig, ax = plt.subplots(4, 6)

for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces_data.target_names[yfit[i]].split()[-1], color='black' if yfit[i] == ytest[i] else 'red')

# plt.show()
plt.savefig('./out_data/image/svm_svc_face.png')
plt.pause(10)
# plt.close()
print(accuracy_score(ytest, yfit))
print(classification_report(ytest, yfit, target_names=faces_data.target_names))
mat = confusion_matrix(ytest, yfit)
sbn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces_data.target_names,
            yticklabels=faces_data.target_names)
plt.xlabel('True label')
plt.ylabel('predict label')
plt.savefig('./out_data/image/svm_svc_label.png')
plt.pause(10)
