package com.apache.spark.ecamples

import org.apache.spark.sql.SparkSession

/**
  * Created by root on 2016/12/14.
  */
object ExceptionHandlingTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("ExceptionHandingTest")
      .master("local")
      .getOrCreate()

    spark.sparkContext.parallelize(0 until spark.sparkContext.defaultParallelism).foreach ( i =>
    if (math.random > 0.75) {
      throw new Exception("Testing exception handling")
    }
    )
  }

}
