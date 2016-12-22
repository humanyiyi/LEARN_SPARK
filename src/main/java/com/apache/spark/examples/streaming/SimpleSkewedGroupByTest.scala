package com.apache.spark.examples.streaming

import java.util.Random

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession


/**
  * Created by root on 2016/12/1.
  */
object SimpleSkewedGroupByTest {
  def main(args: Array[String]) {
    val spark = SparkSession   //spark2.0之后新的功能，代替SQLContest和HiveContest
      .builder
      .master("local")
      .appName("SimpleSkewedGroupByTest")
      .getOrCreate()
    //    val sparkConf = new SparkConf().setMaster("local")


    val numMappers = if (args.length > 0) args(0).toInt else 2
    val numKVPairs = if (args.length > 1) args(1).toInt else 1000
    val valSize = if (args.length > 2) args(2).toInt else 1000
    val numReducers = if (args.length > 3) args(3).toInt else numMappers
    val ratio = if (args.length > 4) args(4).toInt else 5.0


    val pairs1 = spark.sparkContext.parallelize(0 until numMappers, numMappers).flatMap { p =>
      val ranGen = new Random
      val result = new Array[(Int, Array[Byte])](numKVPairs)
      for (i <- 0 until numKVPairs) {
        val byteArr = new Array[Byte](valSize)
        ranGen.nextBytes(byteArr)
        val offset = ranGen.nextInt(1000) * numReducers
        if (ranGen.nextDouble < ratio / (numReducers + ratio - 1)) {
          result(i) = (offset, byteArr)
        } else {
          val key = 1 + ranGen.nextInt(numReducers - 1) + offset
          result(i) = (key, byteArr)

        }
      }
      result
    }.cache
    pairs1.count

    println("RESULT : " + pairs1.groupByKey(numReducers).count)
    spark.stop()
  }
}
