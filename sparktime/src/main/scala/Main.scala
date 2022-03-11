import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.regression.RandomForestRegressor

object Main extends App {
  val spark = SparkSession
    .builder()
    .appName("YahooStocks")
    .master("local")
    .getOrCreate()
  import spark.implicits._
  val dataset = spark.read.option("header", "true").csv("yahoo_stock.csv")
  val closingColumn = dataset.select("Close")
  var dayClosing = closingColumn
    .withColumn("day", monotonicallyIncreasingId)
    .withColumn("day", col("day").cast("int"))
    .withColumn("Close", col("Close").cast("float"))
  val w = Window
    .orderBy("day")
    .rowsBetween(
      -1,
      -1
    ) // Create window to have lagged value (yesterday closing)
  dayClosing = dayClosing.withColumn("yesterday", sum($"Close") over w)
  val regressionTable = new VectorAssembler()
    .setInputCols(Array("day", "yesterday"))
    .setOutputCol("features")
    .setHandleInvalid("skip")
    .transform(dayClosing)
  val model =
    new RandomForestRegressor().setLabelCol("Close").setFeaturesCol("features")
  val linMo = model.fit(regressionTable)
}
