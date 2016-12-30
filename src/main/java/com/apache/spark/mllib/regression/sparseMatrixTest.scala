package com.apache.spark.mllib.regression

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/30.
  */
object sparseMatrixTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Sparse Matrix Test")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\groceries1.csv")
    val records = rowData.map(line => line.split(","))

   records.collect().foreach(println)
  }
}
