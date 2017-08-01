//Following example have been executed on community edition of Databricks
// and therefore no need to initialize spark Context.
//It predicts the ad-click fraud probability based on LogisticRegression.
// Simple example to demonstrate usage of built ML algos.

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}

import java.math.BigInteger
import java.net.{UnknownHostException, InetAddress}
import org.apache.commons.lang.StringUtils
import java.text.SimpleDateFormat

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

//load training data

val df = sqlContext.read.format("csv").option("header", "true").load("/FileStore/tables/plstrjcz1493888515817/train_data.csv")
 
df.show()
df.printSchema()

//load test data

val testdf = sqlContext.read.format("csv").option("header", "true").load("/FileStore/tables/ruzrdpya1495188002657/test_data.csv")
 testdf.show()
 testdf.printSchema()

//replace null with 0 for training data
val df1 = df.na.replace("*", Map("null" -> "0"))

//replace null with 0 for test data
val testdf1 = testdf.na.replace("*", Map("null" -> "0"))
testdf1.show()



//Data parsing function

def ipToBigInteger(s: String): Double = {
  try {
    val i = InetAddress.getByName(s)
    val a: Array[Byte] = i.getAddress() 
    new BigInteger(1, a).longValue()
  } catch {
    case e: UnknownHostException => -1
  }
}


def convertTimetoEpochTime(str:String):Double ={
  try {
    val sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS+05:30")  
    val date = sdf.parse(str)
    val epoch = date.getTime()
    epoch 
  } catch {
    case e: Exception => 0d
  }
}


def getDoubleValue(input: String): Double = input match {
 case "0" => 0.0
 case "1" => 1.0
 case _ => 0.0
 
}


//Get parsed training data

val parsedData = df1.map{row => {
  
  LabeledPoint( 
   getDoubleValue(row.getString(14)),   
   Vectors.dense(convertTimetoEpochTime(row.getString(0)),row.getString(1).toDouble,row.getString(2).toDouble,
                                                        ipToBigInteger(row.getString(5)),row.getString(6).toDouble,row.getString(7).toDouble,row.getString(8).toDouble,ipToBigInteger(row.getString(9)),
                                                       row.getString(10).toDouble,row.getString(11).toDouble,row.getString(12).toDouble)
)
}
}

//Get parsed test data

val testParsedData = testdf1.map{row => {
  
  LabeledPoint( 
   0.0,   
   Vectors.dense(convertTimetoEpochTime(row.getString(0)),row.getString(1).toDouble,row.getString(2).toDouble,
                                                        ipToBigInteger(row.getString(5)),row.getString(6).toDouble,row.getString(7).toDouble,row.getString(8).toDouble,ipToBigInteger(row.getString(9)),
                                                       row.getString(10).toDouble,row.getString(11).toDouble,row.getString(12).toDouble)
)
}
}



parsedData.count() //1048088

testParsedData.count() //1039294

//Only for initial testing purpose training data was split
//val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
//val trainingData = splits(0)
//val testData = splits(1)


//Train the model on parsed training data
val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(parsedData.rdd)

//val trainingData = splits(0)
//val testData = splits(1)


//val labelAndPreds = testParsedData.map { point =>
  //val prediction = model.predict(point.features)
 // (point.label, prediction,point.features)
//}
//val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testParsedData.count
//val fraudClick = labelAndPreds.filter (r=> r._2>0).count.toDouble


model.clearThreshold()
 
val probableLabelAndPreds = testParsedData.map { point =>
  val predictionProbability = model.predict(point.features)
  (point.features, predictionProbability,if (predictionProbability > 0.5) 1 else 0)
}
//1 indicates fraud and 0 indicates no fraud
val fraudResult = ProbablelabelAndPreds.filter(r=> r._3 >0)
display(fraudResult)
