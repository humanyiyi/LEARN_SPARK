package com.apache.spark.mllib.classifier

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/23.
  */
object SVMClassifier {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("SVM Classifier")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile(args(0))
    val records = rowData.map(line => line.split("\t"))
    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map( d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    val numData = data.count
    val numIteration = 10
    val svmModel = SVMWithSGD.train(data, numIteration)
    val svmTotalCorrect = data.map{ point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val svmAccuracy = svmTotalCorrect /numData
    println(svmAccuracy)//0.5146720757268425
    val metrics = Seq(svmModel).map { model =>
      val scoreAndLabels = data.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR,metrics.areaUnderROC)
    }
    println(metrics)//List((SVMModel,0.7567586293858841,0.5014181143280931))

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(data.map(lp => lp.features))
    val scaledData = data.map(lp => LabeledPoint(lp.label,scaler.transform(lp.features)))

    val svmModelScaled = SVMWithSGD.train(scaledData, numIteration)
    val svmTotalCorrectScaled = scaledData.map { point =>
      if (svmModelScaled.predict(point.features) == point.label) 1 else 0
    }sum
    val svmAccuracyScaled = svmTotalCorrectScaled / numData
    val svmPredictionVsScaled = scaledData.map { point =>
      (svmModelScaled.predict(point.features), point.label)
    }
    val svmMetricsScaled = new BinaryClassificationMetrics(svmPredictionVsScaled)
    val svmPr = svmMetricsScaled.areaUnderPR
    val svmRoc = svmMetricsScaled.areaUnderROC
    println(f"${svmModelScaled.getClass.getSimpleName}\nAccuracy: ${svmAccuracyScaled * 100}%2.4f%%\nArea under PR: ${svmPr * 100.0}%2.4f%%\nArea under ROC: ${svmRoc * 100.0}%2.4f%%")
    //SVMModel
    //Accuracy: 62.0284%
    //Area under PR: 72.7693%
    //Area under ROC: 61.9409%
  }
}
