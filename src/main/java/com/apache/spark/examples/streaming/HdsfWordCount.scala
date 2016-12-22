package com.apache.spark.examples.streaming

import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
/**
  * Created by root on 2016/12/1.
  */
object HdsfWordCount {
  def main(args: Array[String]) {
    if(args.length < 1){
      System.err.println("Usage: HdfsWordCount <directory>")
      System.exit(1)
    }

//    StreamingExamples.setStreamingLogLevels()
    val sparkConf = new SparkConf().setMaster("hadoop-04").setAppName("HdfsWordCount")
    // Create the context
    //创建StreamingContext对象，与集群进行交互
    val ssc = new StreamingContext(sparkConf, Seconds(20))
    // Create the FileInputDStream on the directory and use the
    // stream to count words in new files created
    //如果目录中有新创建的文件，则读取
    val lines = ssc.textFileStream(args(0))
    //分割为单词
    val words = lines.flatMap(_.split(" "))
    //统计单词出现次数
    val wordCounts = words.map(x => (x,1)).reduceByKey(_+_)
    //打印结果
    wordCounts.print()
    ssc.start()
//    ssc.stop()
    //一直运行，除非人为干预再停止
    ssc.awaitTermination()
  }

}
