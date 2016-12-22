//推荐模型评估 均方差
package com.apache.spark.mllib.recomm

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/19.
  */
object MSETest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("MSETest")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile(args(0))
    val rowRatings = rowData.map(_.split("\t").take(3))
    val ratings = rowRatings.map{case Array(user, movie, rating) =>
    Rating(user.toInt, movie.toInt, rating.toDouble)}
    val model = ALS.train(ratings, 50, 10, 0.01)
    val userProducts = ratings.map{ case Rating(user, product, rating ) => (user, product)}
    val predictions = model.predict(userProducts).map{
      case Rating(user, product,rating) => ((user,product), rating)
    } //评估值
    val ratingsAndPredictions = ratings.map{
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)  //(user, product) 为主键关联
//    println(ratingsAndPredictions.first())  //输出结果((533,919),(2.0,2.273981792913454))
    val predictedAndTrue = ratingsAndPredictions.map{ case ((user, product), (actual, predicted)) => (actual, predicted)}
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)

    println("Mean Square Error = " + regressionMetrics.meanSquaredError)   //均方差
    println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)  //均方根差
  }
}
