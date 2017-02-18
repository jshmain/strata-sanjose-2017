package com.cloudera.workshop

import org.apache.log4j._
import org.apache.spark.ml.clustering.KMeans

import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.functions._

object kmeans{
    Logger.getRootLogger.setLevel(Level.OFF)
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {
    // System.setProperty("hadoop.home.dir", "C:\\Hadoop");

    val session = org.apache.spark.sql.SparkSession.builder().
      master("local[4]")
      .appName("FlightClustering")
      .getOrCreate()

    // Load the data
    val inputData = session.read
        .option("header","true")
        .option("inferSchema","true").csv(args(0))

    inputData.printSchema()
    inputData.show(50)

    // Transform Day into Something Usable
    val isSat = udf {(x:String) => if (x.toLowerCase.equals("saturday")) 1 else 0}
    val isSun = udf {(x: String) => if (x.toLowerCase.equals("sunday")) 1 else 0}
    val isMon = udf {(x: String) => if (x.toLowerCase.equals("monday")) 1 else 0}

    val transformedDay = inputData.withColumn("Saturday", isSat(inputData("Day")))
                               .withColumn("Sunday", isSun(inputData("Day")))
                               .withColumn("Monday", isMon(inputData("Day")))
    transformedDay.printSchema()
    transformedDay.show()

    // Transform Time into something usable
    val dayFract = udf {(x:String) =>
                          if (x == null)
                            0
                          else
                          {
                            val formatter = new java.text.SimpleDateFormat("h:m a")
                            val curr = formatter.parse(x).getTime.toDouble
                            val full = formatter.parse("11:59 PM").getTime.toDouble
                            curr/full
                          }
                        }

    val transformedTime = transformedDay.withColumn ("dateFract",dayFract(transformedDay("Arrival Time")))
    transformedTime.printSchema()
    transformedTime.show()

    // Maybe use flight for something
    val flightInfo = udf {(x:String) => if (x == null)
                                      0
                                    else
                                    {
                                      //val code = x.charAt(0).toDouble + x.charAt(1).toDouble
                                      //code / ('Z'.toDouble * 10)
                                      0
                                    }
    }

    val transformedFlight = transformedTime.withColumn("Flight",flightInfo(transformedTime("Flight Number")))

    transformedFlight.printSchema()
    transformedFlight.show()

    // Assemble My Features
    val assembler = new VectorAssembler()
                          .setInputCols(Array("Saturday","Sunday","Monday","dateFract","Flight"))
                             .setOutputCol("features")

    val featurizedData = assembler.transform(transformedFlight)
    featurizedData.printSchema()
    featurizedData.show()


    // Scale my features
    //val scaler = new StandardScaler()
    //                     .setInputCol("features")
    //                       .setOutputCol("scaled_features")
    //val scalerModel = scaler.fit(featurizedData)
    //val scaledData = scalerModel.transform(featurizedData)
    // Trains a k-means model

    val kmeans = new KMeans()
      .setK(2)
      .setFeaturesCol("features")
      .setPredictionCol("clusterId")
    val model = kmeans.fit(featurizedData)


    val predictedCluster = model.transform(featurizedData)
    predictedCluster.printSchema()
    predictedCluster.show(100)


    session.stop()
  }
}
