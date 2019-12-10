val t1 = System.nanoTime

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{SparkSession}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank/bank-full.csv")

df.printSchema()
df.show(1)

val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val featCol = change2.withColumn("y",'y.cast("Int"))

featCol.show(1)

val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val featSet = assembler.transform(featCol)

featSet.show(1)

val cambio = featSet.withColumnRenamed("y", "label")
val features = cambio.select("label","features")
features.show(1)

//Multilayer perceptron
val split = features.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = split(0)
val test = split(1)


val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

val model = trainer.fit(train)

val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

//metricas
val duration = (System.nanoTime - t1) / 1e9d
println(s"Duracion: ${duration}")
val mb = 1024*1024
val runtime = Runtime.getRuntime
println("** Used Memory:  " + (runtime.totalMemory - runtime.freeMemory) / mb + "mb")
