package com.apache.spark.ecamples

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/11/30.
  */
object WordCount {
  def main(args: Array[String]): Unit ={
    if(args.length < 1){
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val conf = new SparkConf()
       .setMaster("local").setAppName("aaa")
    val sc = new SparkContext(conf)
    val line = sc.textFile(args(0))
    line.flatMap(_.split(" ")).map((_,1)).reduceByKey(_+_).collect().foreach(println)

    sc.stop()

  }

}
