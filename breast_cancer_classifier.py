# Alyssa Klein
# Breast Cancer Classifier
# Codeacademy Machine Learning Coach project

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load breast cancer data into variable and print
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])

# print feature names to understand data
print(breast_cancer_data.feature_names)

# Store data and labels in variables and print length for verification
training_data, validation_data, training_labels, validation_labels = \
    train_test_split(breast_cancer_data.data, breast_cancer_data.target, train_size = 0.8, random_state = 100)
print("Length of training data: " + str(len(training_data)))
print("Length of training labels: " + str(len(training_labels)) + "\n")

# Create and train classifier with n = 3 k-neighbors
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(training_data, training_labels)

# Print the score (accuracy) of the classifier
print("Accuracy: " + str(classifier.score(validation_data, validation_labels)) + "\n")

# Store scores with index number in array called 'scores'
# Store accuracy scores alone in array called 'accuracies'
scores = []
accuracies = []
for k in range(1,101):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data, training_labels)
    this_score = classifier.score(validation_data, validation_labels)
    print("k = " + str(k) + ": " + str(this_score))
    scores.append([this_score, k])
    accuracies.append(this_score)

# Get maximum accuracy k and print
print("\nMaximum accuracy: " + str(max(scores)) + "\n")

# Store values 1-100 k in a list
k_list = []
for x in range(1,101):
    k_list.append(x)
print(k_list)

# Set x and y axis of plot for visualization and show plot
x = k_list
y = accuracies
plt.plot(x, y)
plt.show()

# Set labels for graph and show graph
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.plot(x,y)
plt.show()