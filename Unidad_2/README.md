# Unit II - Data Classification

### Practices
1. [Basic Stadistics](#basic-stadistics)
2. [Decision Tree](#decision-tree)
3. [Random Forest](#random-forest)
4. [Gradient boosted tree](#gradient-boosted-tree)
5. [Multilayer Perceptron](#multilayer-perceptron)
6. [Support Vector Machine](#support-vector-machine)
7. [Naive-Bayes](#naive-bayes)

### Test
- [Multilayer Perceptron with Iris dataset](#multilayer-perceptron-with-iris-dataset)


## Basic stadistics
### Correllation
Correlation refers to a mutual relationship or association between quantities. In almost any business, it is useful to express one quantity in terms of its relationship with others. For example, sales might increase when the marketing department spends more on TV ads. Spark MLLib provides a `correlation` function that easily helps calculate the correlation between to quantities.

### Hypothesis testing
Hypothesis testing is a powerful tool in statistics to determine whether a result is statistically significant, whether this result occurred by chance or not. Currently Spark MLlib provides hypothesis testing with `chi square test`.

A chi-square test is used in statistics to test the independence of two events. Given the data of two variables, we can get observed count O and expected count E. Chi-Square measures how expected count E and observed count O deviates each other.

### Summsrizer

Spark MLlib offers a easy to use set of functions that can handle operations on dataframes columns, one of them is the set of summarizer functions that let you sum diferent columns and set of columns wich is pretty useful when working with basics stadisctics.
   

## Decision Tree
- Decision tree algorithm falls under the category of supervised learning. They can be used to solve both regression and classification problems.
- Decision tree uses the tree representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree.
- We can represent any boolean function on discrete attributes using the decision tree.

## Random Forest
Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.


To use radom forest algorith we'll need to import the next dependencies to out scala file
```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
```

Split the data into training and test sets (70% and 30%).
```scala
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
```

To train a RFC model we pass the label column and the features columns, lastly passing the number of binary trees randomforent will use
```scala
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
```
## Gradient boosted tree

Gradient-Boosted Trees (GBTs) are ensembles of decision trees. GBTs iteratively train decision trees in order to minimize a loss function. Spark MLlib implementation supports GBTs for binary classification and for regression, using both continuous and categorical features.

To start using gbt we need to import the next dependencies in our scala file
```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
```

Fit on whole dataset to include all labels in index and index labels, adding metadata to the label column.
```scala
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
```

Split the data into training and test sets (70% and 30% ).
```scala
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
```

To train model:
```scala
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
```
## Multilayer Perceptron

A multilayer perceptron classifier is a deep, artificial neural network. It is composed of more than one perceptron. They are composed of an input layer to receive the signal, an output layer that makes a decision or prediction about the input, and in between those two, an arbitrary number of hidden layers that are the true computational engine of the MLP. MLPs with one hidden layer are capable of approximating any continuous function.


To start using MLPC spark implementatio we import the next dependencies to our scala file
```scala
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
```

Index labels, adding metadata to the label column and fit on whole dataset to include all labels in index.
```scala
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
```
Split the data into training and test sets (70% training and 30% testing).
```scala
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
```

To train the model:
```scala
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
```
## Support Vector Machine

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
## Naive-Bayes

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


## Multilayer Perceptron with Iris dataset

To start using MLCP we'll need to import the lines below and adding some dependencies we'll need to clean our dataset.
```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
```

Loads dataset from csv file and additionally it infers the schema of the dataset 
```scala
val data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("iris.csv")
```
Prints the schema
```scala
data.printSchema()
```
Generates a new column called "features" that it's popullated with the other columns in a single vector per row
```scala
val assembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")
val output = assembler.transform(data)
```

Generates a new column with indexed values taken form the string column "species",
this in order to let the algorithm process correctly the dataset 
```scala
val indexer = new StringIndexer().setInputCol("species").setOutputCol("label")
val indexed = indexer.fit(output).transform(output)
```
Splits the dataset into two subsets one for training and one for testing
```scala
val splits = indexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
```

Array in wich we declare our layers
```scala
val layers = Array[Int](4, 5, 4, 3)
```

We initialize our trainer instanciating MLCP and passing as parameters the layers array
```scala
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
```

We train the model passing the `train` subset 
```scala
val model = trainer.fit(train)
```

Prints the result of the run with the test subset, the predictions it made and 
the accuracy it had with those predictions
```scala
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```

