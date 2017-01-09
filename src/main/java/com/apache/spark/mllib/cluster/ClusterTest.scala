package com.apache.spark.mllib.cluster

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/1/6.
  */
object ClusterTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("ClusterTest")
    val sc = new SparkContext(conf)

    val movies = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\ml-100k\\u.item")
    println(movies.first)

    val genres = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\ml-100k\\u.genre")
    genres.take(5).foreach(println)

    val genreMap = genres.filter(!_.isEmpty).map(line => line.split("\\|")).map(array => (array(1),array(0))).collectAsMap
    println(genreMap)

    val titlesAndGenres = movies.map(_.split("\\|")).map{array =>
      val genres = array.toSeq.slice(5, array.size)
      val genresAsSigned = genres.zipWithIndex.filter{case (g, idx) =>
          g == "1"
      }.map{ case (g, idx) =>
          genreMap(idx.toString)
      }
      (array(0).toInt, (array(1), genresAsSigned))
    }
    println(titlesAndGenres.first)

    //训练推荐模型
    val rawData = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\ml-100k\\u.data")
    val rawRatings = rawData.map(_.split("\t").take(3))
    val ratings = rawRatings.map{ case Array(user, movie, rating) =>
        Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    ratings.cache
    val alsModel = ALS.train(ratings, 50, 10, 0.1)

    val movieFactors = alsModel.productFeatures.map{ case (id, factor) =>
      (id, Vectors.dense(factor))}
    val movieVectors = movieFactors.map(_._2)
    val userFactory = alsModel.userFeatures.map{ case (id, factor) =>
      (id, Vectors.dense(factor))}
    val userVectors = userFactory.map(_._2)

    //归一化
    val movieMatrix = new RowMatrix(movieVectors)
    val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
    val userMatrix = new RowMatrix(userVectors)
    val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
    println("Movie factors mean: " + movieMatrixSummary.mean)
    println("Movie factors variance: " + movieMatrixSummary.variance)
    println("User factors mean: " + userMatrixSummary.mean)
    println("User factors variance: " + userMatrixSummary.variance)

    val numClusters = 5
    val numIteration = 10
    val numRuns = 3

    val movieClusterModel = KMeans.train(movieVectors, numClusters, numIteration, numRuns)

    val userClusterModel = KMeans.train(userVectors, numClusters, numIteration, numRuns)

    val movie1 = movieVectors.first
    val movieCluster = movieClusterModel.predict(movie1)
//    println(movieCluster)
    val predictions = movieClusterModel.predict(movieVectors)
    println(predictions.take(10).mkString(","))
  }
}
