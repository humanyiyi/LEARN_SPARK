package com.apache.spark.mllib.regression

import org.apache.spark.mllib.regression.IsotonicRegressionModel
import org.apache.spark.mllib.regression.IsotonicRegression
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/28.
  */
object IsotonicRegressionExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("IsotonicRegression")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\sample_isotonic_regression_libsvm_data.txt")
    val parsedData = data.map{ line =>
      val parts = line.split("\t").map(_.toDouble)
      (parts(0), parts(1), 1.0)
    }

    val splits = parsedData.randomSplit(Array(0.6, 0.4),seed = 11L)
    val training =splits(0)
    val test = splits(1)

    val model = new IsotonicRegression().setIsotonic(true).run(training)

    val predictionAndLabel = test.map{ point =>
      val predictedLabel = model.predict(point._2)
      (predictedLabel,point._1)
    }
//    predictionAndLabel.foreach(println)
    val meanSquaredError = predictionAndLabel.map{ case (p, 1) => math.pow( (p - 1), 2)}.mean()
    println("Mean Squared Error = " + meanSquaredError)

    model.save(sc, "myIsotonicRegressionModel")
    val sameModel = IsotonicRegressionModel.load(sc,"myIsotonicRegressionModel")
  }
}
