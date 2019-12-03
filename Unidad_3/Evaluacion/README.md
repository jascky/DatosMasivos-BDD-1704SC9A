# Kmeans

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