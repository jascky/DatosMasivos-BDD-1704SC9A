import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer

// carga csv e infiere el schema del dataset
val data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("iris.csv")

// se inprime el schema 
data.printSchema()

// se genera un vector con las columnas individuales y se crea una nueva columna
// "features"" con los vectores generados
val assembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")
val output = assembler.transform(data)

// se genera una nueva clomna con datos categoricos a indexados en enteros
// de la columna "species"
val indexer = new StringIndexer().setInputCol("species").setOutputCol("label")
val indexed = indexer.fit(output).transform(output)

// se divide el data set en dos subsets uno para entrenar y otro para probar el modelo
val splits = indexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// array para especificar las layers
val layers = Array[Int](4, 5, 4, 3)

// se crea un nuevo trianer instancioando MLPC y indicando los layers que ocupa
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// se entrena el modelo
val model = trainer.fit(train)

// se imprime el resultado y la prediccon del algoritmo
// junto con la precision que tuvo la prediccion
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")