// Start a Spark Session
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Spark Session
val spark = SparkSession.builder().getOrCreate()

// Import clustering Algorithm
import org.apache.spark.ml.clustering.KMeans

// Loads data.
// val dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")
val data = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale_data.csv")

val feature_data = data.select($"Fresh",$"Milk",$"Grocery",$"Frozen",$"Detergents_Paper",$"Delicassen")

val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")
val dataset = assembler.transform(feature_data)

// Trains a k-means model.
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(dataset)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(dataset)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)