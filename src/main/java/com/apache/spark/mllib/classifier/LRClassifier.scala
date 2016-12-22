package com.apache.spark.mllib.classifier

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
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
//    println(lrModel.predict(data.first.features))
    val numData = data.count
    val lrTotalCorrect = data.map {point =>
      if( lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracy = lrTotalCorrect / numData
//    println(lrAccuracy)  //逻辑回归模型准确率 0.5146720757268425
    val metrics = Seq(lrModel).map { model =>
      val scoreAndLabels = data.map { point =>
        (model.predict(point.features),point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }
//    println(metrics) //List((LogisticRegressionModel,0.7567586293858841,0.5014181143280931))


    /**特征化向量*/
    val vectors = data.map(lp => lp.features)
    val matrix = new RowMatrix(vectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    /**标准化的逻辑回归模型的性能评估*/
    val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIteration)
    val lrTotalCorrectScaled = scaledData.map { point =>
      if (lrModelScaled.predict(point.features) == point.label) 1 else 0
    }sum
    val lrAccuracyScaled = lrTotalCorrectScaled / numData

    val lrPredictionsVsTrue = scaledData.map { point =>
      (lrModelScaled.predict(point.features), point.label)
    }
    val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
    val lrPr = lrMetricsScaled.areaUnderPR
    val lrRoc = lrMetricsScaled.areaUnderROC
    println(f"${lrModelScaled.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaled * 100}%2.4f%%\nArea under PR: ${lrPr * 100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%")
    //LogisticRegressionModel
    //Accuracy: 62.0419%
    //Area under PR: 72.7254%
    //Area under ROC: 61.9663%
  }
}
