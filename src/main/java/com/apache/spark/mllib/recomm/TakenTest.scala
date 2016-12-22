//用户推荐
package com.apache.spark.mllib.recomm

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by root on 2016/12/16.
  */
object TakenTest {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("takenTest")
    val sc = new SparkContext(conf)
    val rowData = sc.textFile("D:\\资料\\spark\\ml-100k\\u.data")
    val rowRatings = rowData.map(_.split("\t").take(3))
    val ratings = rowRatings.map{case Array(user,movie,rating) =>
      Rating(user.toInt,movie.toInt,rating.toDouble)}
    val model = ALS.train(ratings,50, 10 , 0.01)
//    val userCounts = model.userFeatures.count()
//    println("userCounts: " + userCounts)
//    val productCounts = model.productFeatures.count()
//    println("productCounts : " + productCounts)

//    val predictedRating = model.predict(789,123)
//    println(predictedRating)   //预测用户789对电影123的平分

    val userID = 789
    val k = 10
    val topKRacs = model.recommendProducts(userID,k)  //为用户789所能推荐的物品及对应的预计得分
//    println(topKRacs.mkString("\n"))
    val movies = sc.textFile("D:\\资料\\spark\\ml-100k\\u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(array =>
      (array(0).toInt,array(1))).collectAsMap()    //titles输出为（电影名称，具体评分）

//    val moviesForUser = ratings.keyBy(_.user).lookup(789)  //用户评价过的电影
//    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product),rating.rating)).foreach(println)  //789用户评分排名前十的电影

    topKRacs.map(rating => (titles(rating.product),rating.rating)).foreach(println)  //给该用户推荐的10部电影名称和评分



  }

}
