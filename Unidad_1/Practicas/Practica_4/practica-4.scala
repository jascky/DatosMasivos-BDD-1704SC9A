import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")

df.printSchema()

// 1
df.select(min("Low")).show()
// 2
df.select(min("High")).show()
// 3
df.select(mean("High")).show()
// 4
df.filter( $"High" < 300 && $"Low" > 100).show()
// 5
df.select(dayofweek(df("Date"))).show()
// 6
df.select(dayofmonth(df("Date"))).show()
// 7
df.select(dayofyear(df("Date"))).show()
// 8
df.select(weekofyear(df("Date"))).show()
// 9
df.select(last_day(df("Date"))).show()
// 10
df.sort($"High".desc).show()
// 11
df.sort($"High".asc).show()
// 12
df.select($"High").take(2)
// 13
df.head(4)
// 14
df.select("Date").first()
// 15
df.select($"Date",$"High").collect()
// 16
df.select($"Date",$"High").count()
// 17
df.filter($"High" > 300).count()
// 18
df.filter($"Low" < 300).collect()
// 19
df.select("Low").repartition().show()
// 20
df.select("Date").distinct().show()
