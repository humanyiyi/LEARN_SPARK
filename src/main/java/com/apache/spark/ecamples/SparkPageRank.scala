package com.apache.spark.ecamples

import org.apache.spark.sql.SparkSession

/**
  * Created by root on 2016/12/8.
  */
object SparkPageRank {

    def showWarning(): Unit ={
      System.err.println(
        """WARN: This is a naive implementation of PageRank and is given as example!
          |Please use the PageRank implementation found in org.apache.spark.graphx.lib.PageRank
          |for more conventional use.
        """.stripMargin)
    }

  def main(args: Array[String]): Unit = {
//    if(args.length < 1) {
//      System.out.println("Usage: SparkPageRank <file> <iter>")
//      System.exit(1)
//    }
    showWarning()

    val spark = SparkSession
      .builder
      .master("local")
      .appName("PageRank Test")
      .getOrCreate()
    val sc = spark.sparkContext
    val iters = 10

//    val iters = if(args.length > 1) args(1).toInt else 10
//    val lines = spark.read.textFile("D:\\pagerank.txt").rdd
//    val links = lines.map{ s =>
//    val parts = s.split("\\s+")
//      (parts(0),parts(1))
//    }.distinct().groupByKey().cache()
val links = sc.parallelize(Array(('A',Array('D')),('B',Array('A')),
  ('C',Array('A','B')),('D',Array('A','C'))),2).map(x => (x._1, x._2)).cache()

    // 初始化rank值，2表示分两个partition
    var ranks = sc.parallelize(Array(('A',1.0),('B',1.0),('C',1.0),('D',1.0)), 2)
//    var ranks = links.mapValues(v =>1.0)

    for (i <- 1 to iters){
      val contribs = links.join(ranks).values.flatMap{ case (urls, rank) =>
      val size = urls.size
          urls.map(url => (url, rank / size))
      }
      ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
    }
    val output = ranks.collect()
    output.foreach(tup => println(tup._1 + " has rank: " + tup._2 + "."))
    spark.stop()
  }


}
