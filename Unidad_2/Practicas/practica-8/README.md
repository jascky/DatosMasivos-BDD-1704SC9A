# Naive Bayes classifier

Naive Bayes classifiers are a family of simple probabilistic, multiclass classifiers based on applying Bayes’ theorem with strong naive assumptions between every pair of features.

Naive Bayes can be trained very efficiently. With a single pass over the training data, it computes the conditional probability distribution of each feature given each label. For prediction, it applies Bayes’ theorem to compute the conditional probability distribution of each label given an observation.

To start using Naive Bayes we need to import this to our scala file:
```scala
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
```

Load and split the data set into a training set and a test set
```scala
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
```

To train the model:
```scala
val model = new NaiveBayes().fit(trainingData)
```

To show the predictions of the model:
```scala
val predictions = model.transform(testData)
predictions.show()
```


