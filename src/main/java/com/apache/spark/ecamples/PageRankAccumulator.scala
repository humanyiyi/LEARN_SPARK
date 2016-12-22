package com.apache.spark.ecamples

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/8.
  */
object PageRankAccumulator {
  def main(args: Array[String]): Unit = {
    val iters = 20
    val conf = new SparkConf().setMaster("local").setAppName("PageRank")
    val sc = new SparkContext(conf)

    val lines = sc.textFile("D:\\pagerank.txt",1)
    val links = lines.map(line => {
      val parts = line.split("\\s+")
      (parts(0),parts(1))
    }).distinct().groupByKey().cache()

    val nodes = scala.collection.mutable.ArrayBuffer.empty ++ links.keys.collect()
    val newNodes = scala.collection.mutable.ArrayBuffer[String]()
    for {s <- links.values.collect()
      k <- s if (!nodes.contains(k))
    }{
      nodes += k
      newNodes += k
    }
    val linkList = links ++ sc.parallelize(for (i <- newNodes) yield (i,List.empty))
    val nodeSize = linkList.count()
    var ranks = linkList.mapValues(v => 1.0 / nodeSize)

    for (i <- 1 to iters){
      val dangling = sc.accumulator(0.0)
      val contribs = linkList.join(ranks).values.flatMap{
        case (urls, rank) => {
          val size = urls.size
          if(size == 0){
            dangling += rank
            List()
          }else{
            urls.map(url => (url, rank / size))
          }
        }
      }
      contribs.count()
      val dangingValue = dangling.value
      ranks = contribs.reduceByKey(_ + _).mapValues[Double]( p =>
      0.1*(1.0 /nodeSize) + 0.9 * (dangingValue / nodeSize + p))
      println("----------------" + i + "------------------------")
      ranks.foreach(s => println(s._1 + " - " + s._2))
    }
  }

}
