//K值平均准确率
package com.apache.spark.mllib.recomm

import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

/**
  * Created by root on 2016/12/19.
  */
object MAPTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("MAP Test")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile(args(0))
    val rowRatings = rowData.map(_.split("\t").take(3))
    val ratings = rowRatings.map{case Array(user, movie, rating) =>
    Rating(user.toInt, movie.toInt, rating.toDouble)}
    val model = ALS.train(ratings, 50, 10, 0.01)
    val itemFactors = model.productFeatures.map{case (id,factor) => factor }.collect()

    val itemMatrix = new DoubleMatrix(itemFactors)
   // println(itemMatrix.rows,itemMatrix.columns)  //(1682,50)
    val imBroadcast = sc.broadcast(itemMatrix)
    val allRecs = model.userFeatures.map{ case (userId, array) =>
    val userVector = new DoubleMatrix(array)
    val scores = imBroadcast.value.mmul(userVector)
    val scoredWithId = scores.data.zipWithIndex.sortBy(-_._1)
    val recommendedIds = scoredWithId.map(_._2 + 1).toSeq
      (userId, recommendedIds)
    }
    val userMovies = ratings.map{ case Rating(user, product, rating) => (user, product)}.groupBy(_._1)
    val predictedAndTrueForRating = allRecs.join(userMovies).map{ case (userID, (predicted,actualWithIds)) =>
      val actual = actualWithIds.map(_._2)
      (predicted.toArray, actual.toArray)
    }
    val rankingMetrics = new RankingMetrics(predictedAndTrueForRating)
    println("Mean Average Precision = " + rankingMetrics.meanAveragePrecision)   //输出结果Mean Average Precision = 0.19193049145023097
  }
}
