//物品推荐

package com.apache.spark.mllib.recomm

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

/**
  * Created by root on 2016/12/16.
  */
object IterTest {
//  val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))

  def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double ={
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("iterTest").setMaster("local")
    val sc = new SparkContext(conf)
    val rowData = sc.textFile("D:\\资料\\spark\\ml-100k\\u.data")
    val rowRating = rowData.map(_.split("\t").take(3))
    val ratings = rowRating.map{case Array(user, movie, rating) =>
    Rating(user.toInt, movie.toInt,rating.toDouble)}    //org.apache.spark.mllib.recommendation.Rating类对用户ID，影片ID，评分的封装
    val model = ALS.train(ratings, 50, 10, 0.01)  //train(ratings: RDD[Rating], rank: Int, iterations: Int, lambda: Double),其中rank对应ALS模型中的因子个数
                                                  //即“两个小矩阵U（m*k）和V（n*k）”中的K，iteration运行迭代次数，lambda 控制模型的正则化过程，从而控制模型的过程拟合情况
    val itemId = 567
    val k = 10
    val itemFactor = model.productFeatures.lookup(itemId).head
    val itemVector = new DoubleMatrix(itemFactor)
//    println(cosineSimilarity(itemVector,itemVector))

    val sims = model.productFeatures.map{ case (id, factor) =>
      val factorVector = new DoubleMatrix(factor)
      val sim = cosineSimilarity(factorVector, itemVector)
      (id, sim)
    }
    val sortedSims = sims.top(k)(Ordering.by[(Int, Double), Double]{
      case (id, similarity) => similarity
    })
    val sortedSims2 = sims.top(k+1)(Ordering.by[(Int, Double), Double]{
      case (id, similarity) => similarity
    })
//    println(sortedSims.take(10).mkString("\n"))

    val movies = sc.textFile("D:\\资料\\spark\\ml-100k\\u.item")

    val titles = movies.map(line =>line.split("\\|").take(2)).map(array =>
      (array(0).toInt,
        array(1))).collectAsMap()
    println(sortedSims2.slice(1, 11).map{ case (id, sim) => (titles(id), sim)}.mkString("\n"))
  }
}
