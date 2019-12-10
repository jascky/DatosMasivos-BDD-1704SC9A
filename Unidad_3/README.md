# Unit III - Clustering

[![](https://img.shields.io/badge/scala-2.11.12-red)]()
[![](https://img.shields.io/badge/java-1.8.0-orange)]()
[![](https://img.shields.io/badge/Spark-2.4.3-yellow)]()

### Practices
1. [K-means](#kmeans)
2. [Logistic Regresion](#logistic-regresion)

## Kmeans

We start by importing  `SparkSession` we'll need this in order to use all dataframes object functions, also importing `VectorAssembler` from `ml` in order to process the dataset correctly by transforming it with our vector assembler this way the kmeans model can threat correctly our dataset.
```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```

Use the following code below to set the **Error** reporting
```scala
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```

Initialize a `SparkSession`
```scala
val spark = SparkSession.builder().getOrCreate()
```

Import `Kmeans` model from `clustering`
```scala
import org.apache.spark.ml.clustering.KMeans
```

Loads dataset from `.csv` file, and infering the dataset schema
```scala
val data = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale_data.csv")
```

Creates a new dataframe `feature_data` only selecting integer columns and removing the labels
```scala
val feature_data = data.select($"Fresh",$"Milk",$"Grocery",$"Frozen",$"Detergents_Paper",$"Delicassen")
```

Initialize a VectorAssembler object passing as InputCols an Array of strings cotaining the columns name
```scala
val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")
val dataset = assembler.transform(feature_data)
```

Trains the Kmeans model with `k=3` and passing the dataset transform by the VectorAssembler `dataset`
```scala
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(dataset)
```

Evaluate clustering by computing Within Set Sum of Squared Errors
```
val WSSSE = model.computeCost(dataset)
println(s"Within Set Sum of Squared Errors = $WSSSE")
```


Shows the result
```
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
```

## Logistic regresion

To start using spark's ml logistic regresion, we need to use the lines below, this will import the `logisticRegresion` model
```scala
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
```

We instanciate LogisticRegresion model 
```scala
val lr = new LogisticRegression()
```
Logistic regrsion model expects a feature vector, containing categorical integers as it's features, to do that we can use a pipeline, to set a path to tranform the columns we need of our dataset and at the end of the pipeline we'll have a clean dataset, after that we can now process our data in the model so thats why the logisticregresion model it's at the of the pipeline
```scala
import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(genderIndexer,embarkIndexer,genderEncoder,embarkEncoder,assembler,lr))
```
We pass a subset of the dataset to the pipeline, generating a trained model
```scala
val model = pipeline.fit(training)
```

With our trained model we pass a test subset to try our model
```scala
val results = model.transform(test)
```