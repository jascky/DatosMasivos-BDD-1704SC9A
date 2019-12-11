
val t1 = System.nanoTime

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank/bank-full.csv")

df.printSchema()
df.show(1)

val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))

newcolumn.show(1)

val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val features = assembler.transform(newcolumn)

features.show(1)

val cambio = features.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)

//DecisionTree
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)

val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4) 
val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))

val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)

predictions.select("predictedLabel", "label", "features").show(5)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

//metricas
val duration = (System.nanoTime - t1) / 1e9d
println(s"Duracion: ${duration}")
val mb = 1024*1024
val runtime = Runtime.getRuntime
println("** Used Memory:  " + (runtime.totalMemory - runtime.freeMemory) / mb + "mb")
