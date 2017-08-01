
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.feature.StandardScaler

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.sql.functions._ 


val forestCoverTrainDF = sqlContext.read.format("csv").option("header","true").option("inferSchema","true")load("/FileStore/tables/xm53w63w1499922730350/train.csv")
val testDF = sqlContext.read.format("csv").option("header","true").option("inferSchema","true").load("/FileStore/tables/1ggn6cko1499922910223/test.csv")

forestCoverTrainDF.take(5)

//Verify the schema to be of type Integer/Double to simplify computation
forestCoverTrainDF.printSchema()
testDF.printSchema()

//Verify if any missing values in data. If so we need we either need to fill them or ignore them
forestCoverTrainDF.describe().show()
//Since all columns have count equal to forestCoverTrainDF.count() so no missing value

// Learning :
// No attribute is missing as count is 15120 for all attributes. Hence, all rows can be used
// Negative value(s) present in Vertical_Distance_To_Hydrology. Hence, some tests such as chi-sq cant be used.
// Wilderness_Area and Soil_Type are one hot encoded. Hence, they could be converted back for some analysis
// Attributes Soil_Type7 and Soil_Type15 can be removed as they are constant
// Scales are not the same for all. Hence, rescaling and standardization may be necessary for some algos

//Verify distribution of different Cover_Type across training data. Claa distribution 
forestCoverTrainDF.groupBy("Cover_Type").count().show()

//Verify correlation in between different columns. Consider features with abs (correlation) > 0.5 (keeping 0.5 as threshold value)
//Correlation more applicable to continous functions then binary. hence ignoring Soil_Type and Wilderness area.

val corrdDF= forestCoverTrainDF.select("Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Cover_Type")

val rows = new VectorAssembler().setInputCols(corrdDF.columns).setOutputCol("vs")
  .transform(corrdDF).select("vs").rdd

val matrixRow = rows
  .map(_.getAs[org.apache.spark.ml.linalg.Vector](0))
  .map(org.apache.spark.mllib.linalg.Vectors.fromML)


val correlMatrix: Matrix = Statistics.corr(matrixRow, "pearson")
//highly correlated feature may be reduces=d using PCA
//Hillshade_9am and Hillshade_3pm = -0.78
//Horizontal_Distance_To_Hydrology and Vertical_Distance_To_Hydrology = 0.65
//Aspect and Hillshade_3pm = 0.64
//Hillshade_Noon and Hillshade_3pm = 0.61
//Slope and Hillshade_Noon = -0.61
//Aspect and Hillshade_9am = -0.59
//Elevation and Horizontal_Distance_To_Roadways = 0.58

//Data Preperations
//Drop Soil_Type 7 and Soil_type 15 and its std dev is 0 and Id as it will not be used for predictions

val cleanedDF = corrdDF.drop("Cover_Type")

val featureUsed= Array("Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points")

val assembler = new VectorAssembler()
  .setInputCols(featureUsed)
  .setOutputCol("featuresUnscaled")

val assembled = assembler.transform(cleanedDF)
//assembled.show()

//Apply Standard Scaling only in non-categorical data
val scaler = new StandardScaler()
  .setInputCol("featuresUnscaled")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false)

// Compute summary statistics by fitting the StandardScaler.
val scalerModel = scaler.fit(assembled)

// Normalize each feature to have unit standard deviation.
val scaledData = scalerModel.transform(assembled)
scaledData.take(5)

//Feature Selection
//Chi-Squared

val selectedFeaturesDF = forestCoverTrainDF.drop("Soil_Type7").drop("Soil_Type15") //Cover_type is target column and Soil_Type7/15 has std dev 0
/*val modifiedDF = selectedFeaturesDF.drop("Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",   "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points") //Dropping them since the value will be captured as scaledFeatures


val scaledFeaturesIndexedDF = scaledData.select("scaledFeatures").withColumn("Id", monotonically_increasing_id())


val newDF= modifiedDF.join(scaledFeaturesIndexedDF, modifiedDF.col("Id") === scaledFeaturesIndexedDF.col("Id")) */
val df= selectedFeaturesDF.drop("Id") 

val vAssembler = new VectorAssembler()
  .setInputCols(df.columns)
  .setOutputCol("features") 

val vAssembled = vAssembler.transform(df)

val selector = new ChiSqSelector()
  .setNumTopFeatures(30)
  .setFeaturesCol("features")
  .setLabelCol("Cover_Type")
  .setOutputCol("selectedFeatures")
vAssembled.columns
val result = selector.fit(vAssembled).selectedFeatures

//println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
//val importantFeatures = result.selectedFeatures

//selectedFeaturesDF: org.apache.spark.sql.DataFrame = [Id: int, Elevation: int ... 52 more fields]
//df: org.apache.spark.sql.DataFrame = [Elevation: int, Aspect: int ... 51 more fields]
//vAssembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_8ababc60ae13
//vAssembled: org.apache.spark.sql.DataFrame = [Elevation: int, Aspect: int ... 52 more fields]
//selector: org.apache.spark.ml.feature.ChiSqSelector = chiSqSelector_beec904c71e0
//result: Array[Int] = Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33)
