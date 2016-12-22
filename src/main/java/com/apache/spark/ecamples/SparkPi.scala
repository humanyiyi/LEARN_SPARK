package com.apache.spark.ecamples

import org.apache.spark.sql.SparkSession
import org.apache.spark.util.random

/**
  * Created by root on 2016/12/1.
  */
object SparkPi {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
        .master("local")
      .appName("Spark Pi")
      .getOrCreate()
    val slices = if (args.length > 0) args(0).toInt else 2
    val n = math.min(100000L * slices, Int.MaxValue).toInt // avoid overflow
    val count = spark.sparkContext.parallelize(1 until n, slices).map { i =>
      val x = Math.random * 2 - 1
      val y = Math.random * 2 - 1
      if (x*x + y*y < 1) 1 else 0
    }.reduce(_ + _)
    println(n)
    println(count)
    println("Pi is roughly " + 4.0 * count / (n - 1))
    spark.stop()
  }
}
