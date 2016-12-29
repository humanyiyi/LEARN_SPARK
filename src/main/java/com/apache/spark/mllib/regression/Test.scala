package com.apache.spark.mllib.regression

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/29.
  */
object Test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Test")
    val sc = new SparkContext(conf)
    val rowData = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\groceries.csv")
    val result = rowData.map(line => line.split(","))
    println(result.first())
  }
}
