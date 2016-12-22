package com.apache.spark.ecamples

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/2.
  */
object Test {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local").setAppName("Test")
    val sc = new SparkContext(sparkConf)
//    val a = sc.parallelize(List((1,2),(3,4),(3,6),(3,2)))
//    a.reduceByKey((x,y) => x+y).collect().foreach(println)
    val a = sc.parallelize(1 to 4, 2)
    val b = a.flatMap(x => 1 to x)
    b.collect().foreach(println)
  }

}
