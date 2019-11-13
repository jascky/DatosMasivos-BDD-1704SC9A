# Support Vector Machine 

SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.

Spark ML supports binary classification with linear SVM. 

To start using SVM in spark we import the next dependencies to our scala file
```scala
import org.apache.spark.ml.classification.LinearSVC
```
We load the trainig data
```scala
val training = spark.read.format("libsvm").load("../sample_libsvm_data.txt")
```
To initialize a new svm instance:
```scala
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
```

To train the model:
```scala
val lsvcModel = lsvc.fit(training)
```