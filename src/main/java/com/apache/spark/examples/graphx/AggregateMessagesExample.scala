package com.apache.spark.examples.graphx

import org.apache.spark.graphx.{Graph, VertexRDD}
import org.apache.spark.graphx.util.GraphGenerators
import org.apache.spark.sql.SparkSession


/**
  * Created by root on 2016/12/2.
  */
object AggregateMessagesExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Agg")
      .getOrCreate()

    val sc =spark.sparkContext

    val graph: Graph[Double, Int] =
      GraphGenerators.logNormalGraph(sc, numVertices = 100).mapVertices((id,_) => id.toDouble)

    val olderFollowers: VertexRDD[(Int, Double)] = graph.aggregateMessages[(Int,Double)](
      triplet => {
        if(triplet.srcAttr > triplet.dstAttr){
          triplet.sendToDst(1,triplet.srcAttr)
        }
      },
      (a,b) => (a._1 + b._1, a._2 + b._2)
    )

    val avgAgeOfOlderFollowers: VertexRDD[Double] =
      olderFollowers.mapValues((id, value) =>
      value match {case (count, totalAge) => totalAge / count})

    avgAgeOfOlderFollowers.collect().foreach(println(_))

    spark.stop()
  }

}
