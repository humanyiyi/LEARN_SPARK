package com.udbac.spark.example.mllib

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/7.
  */
object KmeansTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("KmeanTest")
    val sc =new SparkContext(conf)

    val rawTrainingData = sc.textFile("D:\\test.csv")
    val parsedTrainingData =
      rawTrainingData.filter(!isColumnNameLine(_)).map(line => {
        Vectors.dense(line.split(",").map(_.trim).filter(!"".equals(_)).map(_.toDouble))
      }).cache()

    val numClusters = 8  //预计分8个列簇
    val numIterations = 30 //迭代次数
    val runTimes = 3  //并行度
    var clusterIndex: Int = 0
    val clusters: KMeansModel =
      KMeans.train(parsedTrainingData,numClusters,numIterations, runTimes)

    println("Cluter Number:" + clusters.clusterCenters.length)

    println("Cluter Centers Information OverView:")
    clusters.clusterCenters.foreach(
      x => {
        println("Center Point of Cluster " + clusterIndex + ":")
        println(x)
        clusterIndex += 1
      })
    val rawTestData = sc.textFile("D:\\test.csv")
    val parsedTestData = rawTestData.map(line => {
      Vectors.dense(line.split(",").map(_.trim).filter(!"".equals(_)).map(_.toDouble))
    })
    parsedTestData.collect().foreach(testDataLine =>{
      val predictedClusterIndex:
        Int = clusters.predict(testDataLine)
      println("The data " + testDataLine.toString + "belongs to cluster " + predictedClusterIndex)
    })  //数组属于哪个列簇
    println("Spark MLlib K-means clustering test finished.")
  }
  private def isColumnNameLine(line: String): Boolean = {
    if (line != null && line.contains("Channel")) true
    else false
  }

}
