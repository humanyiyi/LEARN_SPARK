package com.apache.spark.ecamples

import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

/**
  * Created by root on 2016/12/12.
  */
object SkewedGroupByTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("skewedGroupByTest")
    val sc = new SparkContext(conf)

    val numMappers = if (args.length > 0) args(0).toInt else 2
    var numKVPairs = if (args.length > 1) args(1).toInt else 1000
    val valSize = if (args.length > 2) args(2).toInt else 1000
    val numReducers = if (args.length > 3) args(3).toInt else numMappers

    val pairs1 = sc.parallelize(0 until numMappers, numMappers).flatMap{ p =>
      val ranGen = new Random

      numKVPairs = (1.0 * (p + 1) / numMappers * numKVPairs).toInt
      val arr1 = new Array[(Int,Array[Byte])](numKVPairs)
      for (i <- 0 until numKVPairs) {
        val byteArr = new Array[Byte](valSize)
        ranGen.nextBytes(byteArr)
        arr1(i) = (ranGen.nextInt(Int.MaxValue), byteArr)
      }
      arr1
    }.cache()

    pairs1.count()

    println(pairs1.groupByKey(numReducers).count())
    sc.stop()
  }

}
