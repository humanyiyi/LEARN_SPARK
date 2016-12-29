package com.apache.spark.mllib.regression

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/29.
  */
object LogisticRegressionLearning {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("LogisticRegressionLearning")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\logRegression.data")
    val parseData = data.map{ line =>
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()
    parseData.foreach(println)
    val model = LogisticRegressionWithSGD.train(parseData,50)
    val target = Vectors.dense(-1)
    val result = model.predict(target)
    println("model.weights:")
    println(model.weights)
    println(result)
    println(model.predict(Vectors.dense(10)))
    sc.stop()
  }
}
