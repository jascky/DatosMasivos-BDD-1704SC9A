# Gradient-Boosted Trees 

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