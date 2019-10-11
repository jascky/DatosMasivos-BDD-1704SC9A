import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix.csv")

df.columns

df.printSchema()

df.show()

df.head(5)

df.describe().show()


val df2 = df.withColumn("HV Ratio", df("High") + df("Volume"))

df.select(min("Volume")).show()

df.select(max("Volume")).show()

df.filter($"Close" < 600).count()

val time = (df.filter($"High" > 500).count() * 100) / df.count()

df.select(corr($"High",$"Volume")).show()

 

val highestYears = df.withColumn("Year", year($"Date"))
highestYears.select($"Year",$"High").groupBy("Year").max().show()


val mothAvg = df.withColumn("Month", month($"Date"))
mothAvg.select($"Month",$"Close").groupBy("Month").avg().show()
