# Random Forest
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