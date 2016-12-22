package com.apache.spark.mllib.classifier

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/22.
  */
object LRClassifier {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("LogisticRegressionWithSGD Classifier")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile(args(0))
    val records = rowData.map(line => line.split("\t"))

    val data = records.map{ r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size -1).map( d => if (d =="?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    val numIteration = 10
    val lrModel = LogisticRegressionWithSGD.train(data, numIteration)
    println(lrModel.predict(data.first.features))
  }
}
