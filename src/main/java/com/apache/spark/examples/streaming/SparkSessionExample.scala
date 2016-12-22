//读取文件
package com.apache.spark.examples.streaming

import org.apache.spark.sql.SparkSession

/**
  * Created by root on 2016/12/1.
  */
object SparkSessionExample {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession
      .builder
      .master("local")
      .appName("SparkSession")
      .getOrCreate()
    val df = sparkSession.read.option("header","true").csv("D:\\UDBAC\\LEARN_SPARK\\user.csv")
    df.show()
  }

}
