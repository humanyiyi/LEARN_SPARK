package com.apache.spark.examples.streaming

import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
  * Created by root on 2016/12/1.
  */
object NetwordCount {
  def main(args: Array[String]) {
    if(args.length < 2){
      System.err.println("Usage: NetworkCount <hostname> <port>")
      System.exit(1)
    }

    val sparkConf = new SparkConf().setMaster("local").setAppName("NetwordCount")
    val ssc = new StreamingContext(sparkConf,Seconds(2))
    val lines = ssc.socketTextStream(args(0),args(1).toInt,StorageLevel.MEMORY_AND_DISK)
    val words = lines.flatMap(_.split(" "))
    val wordCounts = words.map(x => (x,1)).reduceByKey(_+_)
    wordCounts.print()
    ssc.start()
    ssc.awaitTermination()
  }

}
