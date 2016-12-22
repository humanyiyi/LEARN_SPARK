package com.apache.spark.ecamples

import org.apache.spark.sql.SparkSession

/**
  * Created by root on 2016/12/5.
  */
object HdfsTest {

  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      System.out.println("Usage: HDFSTest <file>")
      System.exit(1)
    }
    val spark = SparkSession
      .builder
      .master("local")
      .appName("HdfsTest")
      .getOrCreate()
    val file = spark.read.text(args(0)).rdd
    val mapped = file.map(s => s.length).cache()
    for (iter <- 1 to 10) {
      val start = System.currentTimeMillis()
      for (x <- mapped) {
        x + 2
      }
      val end = System.currentTimeMillis()
      println("Iteration " + iter + " took " + (end-start) + " ms")
    }
    spark.stop()
  }

}
