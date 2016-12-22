//测试多个Broadcast共享变量。

package com.apache.spark.ecamples

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/14.
  */
object MultiBroadcastTest {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local").setAppName("MultiBroadCastTest")
    val sc = new SparkContext(conf)
    val slices = if (args.length > 0) args(0).toInt else 2
    val num = if (args.length > 1) args(1).toInt else 1000000

    val arr1 = new Array[Int](num)
    for (i <- 0 until arr1.length) {
      arr1(i) = i
    }

    val arr2 = new Array[Int](num)
    for (i <- 0 until arr2.length) {
      arr2(i) = i
    }

    val barr1 = sc.broadcast(arr1)
    val barr2 = sc.broadcast(arr2)
    val observedSizes: RDD[(Int, Int)] = sc.parallelize(1 to 10, slices).map {
      _ => (barr1.value.length,barr2.value.length)
    }

    observedSizes.collect().foreach(i => println(i))

    sc.stop()
  }
}
