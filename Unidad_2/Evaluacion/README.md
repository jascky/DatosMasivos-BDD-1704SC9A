# Multi layer percepton

To start using MLCP we'll need to import the lines below and adding some dependencies we''l need to clean our dataset.
```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
```

Loads dataset from the csv file and additionally it infers the schema of the dataset 
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