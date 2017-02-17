package com.udbac.spark.example.mllib

import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/16.
  */
object BisectingKMeansExample {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("BisectingKMeansExample").setMaster("local")
    val sc = new SparkContext(conf)

    def parse(line: String): Vector = Vectors.dense(line.split(" ").map(_.toDouble))
    val data = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\kmeans_data.txt").map(parse).cache()

    val bkm = new BisectingKMeans().setK(6)
    val model = bkm.run(data)

    println(s"Compute Cost: ${model.computeCost(data)}")
    model.clusterCenters.zipWithIndex.foreach{ case (center, idx) =>
      println(s"Cluster Center ${idx} : ${center}")
    }

    sc.stop()
  }
}
