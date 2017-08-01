import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.Bucketizer

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.clustering.KMeans

val neighbourHoodDF = sqlContext.read.format("csv").option("header","true").option("inferSchema","true")load("/FileStore/tables/test/neighbourDataSet.csv")
//neighbourHoodDF.take(5)
//neighbourHoodDF.printSchema()

//Cast Age from Integer to Double type
val dfNew = neighbourHoodDF.withColumn("nage",  neighbourHoodDF("age").cast(DoubleType))
    .drop("age").withColumnRenamed("nage", "age")

////Cast LivingDuration from Integer to Double type
val modifiedNeighBourDF = dfNew.withColumn("nLivingDuration",  dfNew("LivingDuration").cast(DoubleType))
    .drop("LivingDuration").withColumnRenamed("nLivingDuration", "LivingDuration")
    
 val categoricalColumns = Array("FoodHabits", "LifeStyle", "education", "Gender", "MainLocation", "Interest", "Community")

//tranformor to convert string to category values
val foodHabitsIndexer = new StringIndexer().setInputCol("FoodHabits").setOutputCol("foodHabitsIndex")
val lifeStyleIndexer = new StringIndexer().setInputCol("LifeStyle").setOutputCol("lifeStyleIndex")
val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
val genderIndexer = new StringIndexer().setInputCol("Gender").setOutputCol("genderIndex")
val mainLocationIndexer = new StringIndexer().setInputCol("MainLocation").setOutputCol("mainLocationIndex")
val interestIndexer = new StringIndexer().setInputCol("Interest").setOutputCol("interestIndex")
val communityIndexer = new StringIndexer().setInputCol("Community").setOutputCol("communityIndex")

//Encode the value using One-Hot Encoding
val foodHabitsEncoder = new OneHotEncoder().setInputCol("foodHabitsIndex").setOutputCol("foodHabitsEncode")
val lifeStyleEncoder = new OneHotEncoder().setInputCol("lifeStyleIndex").setOutputCol("lifeStyleEncode")
val educationEncoder = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("educationEncode")
val genderEncoder = new OneHotEncoder().setInputCol("genderIndex").setOutputCol("genderEncode")
val mainLocationEncoder = new OneHotEncoder().setInputCol("mainLocationIndex").setOutputCol("mainLocationEncode")
val interestEncoder = new OneHotEncoder().setInputCol("interestIndex").setOutputCol("interestEncode")
val communityEncoder = new OneHotEncoder().setInputCol("communityIndex").setOutputCol("communityEncode")

//Bucketize the age and livingDuration
val ageSplits = Array(0.0,10.0,20.0,30.0,40.0,50.0,Double.PositiveInfinity)
val ageBucketize = new Bucketizer().setInputCol("age").setOutputCol("ageBucketed").setSplits(ageSplits)
//val ageBucketedData = ageBucketize.transform(modifiedNeighBourDF).select("age","ageBucketed").show(10)


val livingDurationSplits = Array(0.0,5.0,10.0,Double.PositiveInfinity)
val livingDurationBucketize = new Bucketizer().setInputCol("LivingDuration").setOutputCol("livingDurationBucketed").setSplits(livingDurationSplits)
//val livingDurationBucketedData = livingDurationBucketize.transform(modifiedNeighBourDF).select("LivingDuration","livingDurationBucketed").show(10)

val assembler = new VectorAssembler().setInputCols(Array("foodHabitsEncode","lifeStyleEncode","educationEncode","genderEncode",                                                          "mainLocationEncode","interestEncode","communityEncode","ageBucketed","livingDurationBucketed")).setOutputCol("features")
val kmeans = new KMeans().setK(12).setFeaturesCol("features").setPredictionCol("prediction")

//Execute the pipeline
val clusterPipeline = new Pipeline().setStages(Array(foodHabitsIndexer,foodHabitsEncoder,lifeStyleIndexer,lifeStyleEncoder,educationIndexer,educationEncoder,genderIndexer,genderEncoder,
                                                  mainLocationIndexer,mainLocationEncoder,interestIndexer,interestEncoder,
                                                 communityIndexer,communityEncoder,ageBucketize,livingDurationBucketize,assembler,
                                                  kmeans))

val kMeansPredictionModel = clusterPipeline.fit(modifiedNeighBourDF)
 
val predictionResult = kMeansPredictionModel.transform(modifiedNeighBourDF)
predictionResult.persist()
predictionResult.select("features","prediction").show()
