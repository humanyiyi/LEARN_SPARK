package com.apache.spark.mllib.recomm

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/15.
  */
object RecommendationExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("CollaborativeFilteringExample").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\als\\test.data")
    val ratings = data.map(_.split(",") match {case Array(user, item, rate) => Rating(user.toInt, item.toInt, rate.toDouble)})

    //创建ALS推荐模型
    val rank = 10
    val numIterations = 10
    val model = ALS.train(ratings, rank, numIterations, 0.01)

    val usersProducts = ratings.map{ case Rating(user, product, rate) => (user, product)}
    val predictions = model.predict(usersProducts).map{ case Rating(user, product, rate) => ((user, product), rate)}
    val ratesAndPreds = ratings.map{ case Rating(user, product, rate) => ((user, product), rate)}.join(predictions)
    val MSE = ratesAndPreds.map{ case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()
    println("Mean Squared Error = " + MSE)
    model.save(sc, "D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\als\\output")
    val sameModel = MatrixFactorizationModel.load(sc, "D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\als\\output")
  }
}
