import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
# load the training dataset
wine = pd.read_csv('wine.data')

print(wine.head(10))

print(wine.columns)
print()
#check Null Values:
print(wine.isnull().sum())
l=wine['wine_type']
wine=wine.drop(columns='wine_type')
wine=wine.assign(wine_type=l)
print(wine.head(10))

print(wine.columns)
# Separate features and labels
features = ['Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh',
       'Magnesium', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols',
       'Proanthocyanins', 'ColorIntensity', 'Hue', 'DilutedWines', 'Proline']

label = 'wine_type'
X, y = wine[features].values, wine[label].values
encoder=OrdinalEncoder()
y=encoder.fit_transform(y.reshape(-1, 1))
print(y)
#Extract information from Dataset
from matplotlib import pyplot as plt
for col in features:
    wine.boxplot(column=col, by='wine_type', figsize=(6,6))
    plt.title(col)
plt.show()



from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0,stratify=y)

print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))
print ('Training label cases: %d\nTest label cases: %d' % (y_train.shape[0], y_test.shape[0]))


# Train the model
from sklearn.linear_model import LogisticRegression

# Set regularization rate:Then take the inverse of regularization strength; must be a positive float. 
# Like in support vector machines, smaller values specify stronger regularization.
reg = 0.01



from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,confusion_matrix,precision_score,recall_score

numeric_feature=[0,1,2,3,4,5,6,7,8,9,10,11,12]
numeric_transformer=Pipeline(steps=[('scaler',StandardScaler())])


wine_classes= ['grape1','grape2','grape3']
preprocessor=ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numeric_feature)
        
    ]
)

pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('logregressor',RandomForestClassifier(n_estimators=100))

])

model=pipeline.fit(X_train,(y_train))

print(model)


#Evaluate the Model
# Get predictions from test data
predictions = model.predict(X_test)
y_scores = model.predict_proba(X_test)

# Get evaluation metrics
cm = confusion_matrix(y_test, predictions)
print ('Confusion Matrix:\n',cm, '\n')
print('Accuracy:', accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions,average='macro'))
print("Overall Recall:",recall_score(y_test, predictions,average='macro'))

print(y_test)
# Get ROC metrics for each class
fpr = {}
tpr = {}
thresh ={}
for i in range(len(wine_classes)):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_scores[:,i], pos_label=i)
    
# Plot the ROC chart
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=wine_classes[0] + ' vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label=wine_classes[1] + ' vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label=wine_classes[2] + ' vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()


auc = roc_auc_score(y_test,y_scores, multi_class='ovr')
print('Average AUC:', auc)


# Confusion matrix
from sklearn.metrics import confusion_matrix

# Print the confusion matrix
mcm = confusion_matrix(y_test, predictions)
print(mcm)

plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(wine_classes))
plt.xticks(tick_marks, wine_classes, rotation=45)
plt.yticks(tick_marks, wine_classes)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()


import joblib

# Save the model as a pickle file
filename = './wine_model.pkl'
joblib.dump(model, filename)


# Load the model from the file
model = joblib.load(filename)

# predict on a new sample
# The model accepts an array of feature arrays (so you can predict the classes of multiple wine_features in a single call)
# We'll create an array with a single array of features, representing one wine_feature
X_new = np.array([[13.34,.94,2.36,17,110,2.53,1.3,.55,.42,3.17,1.02,1.93,750
]])
print ('New sample: {}'.format(list(X_new[0])))

# Get a prediction
pred = model.predict(X_new)

# The model returns an array of predictions - one for each set of features submitted
# In our case, we only submitted one wine_feature, so our prediction is the first one in the resulting array.
print('Predicted class is {}'.format(pred[0]))
