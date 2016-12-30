package com.apache.spark.mllib.regression

import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.Map

/**
  * Created by root on 2016/12/29.
  */
object dataTransformTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Test")
    val sc = new SparkContext(conf)
    val rowData = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\groceries.csv")
    val results = rowData.map(line => line.split(","))

    val dt = results.map { result =>

      var dt = Map[String,Int]()
      for (i <- 0 to result.size - 1) {
      val data = (result(i) -> 0)
        dt += data
    }
//      dt.toVector
      dt
    }
//    dt.foreach(println)
    dt.saveAsTextFile("D:\\UDBAC\\LEARN_SPARK\\data\\output")

//    results.first().foreach(println)
//    println(results.first().size)

//    val dataKey = rowData.map{ line =>
//      val parts = line.split(",")
//      for(i <- 0 to parts.size - 1){
//        val data = Map(parts(i) -> 0)
//        data.foreach(println)
//      }
//      null
//    }
//   dataKey.foreach(println)

  }
}
