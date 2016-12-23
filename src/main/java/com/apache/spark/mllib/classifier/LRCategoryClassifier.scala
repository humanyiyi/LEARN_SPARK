package com.apache.spark.mllib.classifier

import breeze.optimize.proximal.LogisticGenerator
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/22.
  */
//标准化数据加上类别区分的逻辑回归模型
object LRCategoryClassifier {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("LRCategory Classifier")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile(args(0))
    val records = rowData.map(line => line.split("\t"))

    //1-of-k 编码
    val categories = records.map( r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size

    val dataCategories = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }

    val numData = dataCategories.count
    val scalerCats = new StandardScaler(withMean = true, withStd = true)
      .fit(dataCategories.map(lp => lp.features))
    val scaledDataCats = dataCategories.map(lp =>
    LabeledPoint(lp.label,scalerCats.transform(lp.features)))

    val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats,numCategories)
    val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
      if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracyScaledCats = lrTotalCorrectScaledCats / numData
    val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
      (lrModelScaledCats.predict(point.features), point.label)
    }
    val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
    val lrPrCats = lrMetricsScaledCats.areaUnderPR
    val lrRocCats = lrMetricsScaledCats.areaUnderROC
    println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: ${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%")
    //LogisticRegressionModel
    //Accuracy: 66.5720%
    //Area under PR: 75.8129%
    //Area under ROC: 66.5418%
  }

}
