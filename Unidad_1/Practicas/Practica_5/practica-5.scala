import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Sales.csv")
df.printSchema()
df.show()

// 1
df.select(max("Sales")).show()

//2
df.select(sum("Sales")).show()

//3
df.select(last("Sales")).show()

//4
df.select(kurtosis("Sales")).show()

//5
df.groupBy("Company").count().show()

//6
df.select(first("Company")).show()

//7
df.select(last("Company")).show()

//8
df.select(skewness("Sales")).show()

//9
df.select(approx_count_distinct("Sales")).show()

//10
df.select(avg("Sales")).show()

//11
df.select(stddev_pop("Sales")).show()

//12
df.select(stddev_samp("Sales")).show()

//13
df.select(variance("Sales")).show()

//14
df.select(var_samp("Sales")).show()

//15
df.select(collect_list("Sales")).show()
