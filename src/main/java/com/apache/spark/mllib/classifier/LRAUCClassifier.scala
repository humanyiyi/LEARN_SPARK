package com.apache.spark.mllib.classifier

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by root on 2016/12/22.
  */
object LRAUCClassifier {
  def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = {
    val lr = new LogisticRegressionWithSGD
    lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    lr.run(input)
  }
  def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
    val scoreAndLabels = data.map { point =>
      (model.predict(point.features),point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label, metrics.areaUnderROC)
  }
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("LR AUC Classifier")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile(args(0))
    val records = rowData.map(line => line.split("\t"))
    val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size

    val dataCategories = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories )
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }

    val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(lp => lp.features))
    val scaledDataCats = dataCategories.map(lp =>
    LabeledPoint(lp.label,scalerCats.transform(lp.features)))
    scaledDataCats.cache
    val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats,numCategories)
    val iterResults = Seq(1, 5, 10, 50).map { param =>  //不同迭代次数
      val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
      createMetrics(s"$param iterations", scaledDataCats, model)
    }
//    iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.4f%%")}
    //1 iterations, AUC = 64.9520%
    //5 iterations, AUC = 66.6161%
    //10 iterations, AUC = 66.5483%
    //50 iterations, AUC = 66.8143%

//    val stepResult = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param => //不同步长
//      val model = trainWithParams(scaledDataCats, 0.0, 10,new SimpleUpdater, param )
//      createMetrics(s"$param step size", scaledDataCats, model)
//    }
//    stepResult.foreach{ case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.4f%%")}
    //0.001 step size, AUC = 64.9659%
    //0.01 step size, AUC = 64.9644%
    //0.1 step size, AUC = 65.6264%
    //1.0 step size, AUC = 66.5418%
    //10.0 step size, AUC = 64.8967%

    val regResult = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map{ param =>
      val model = trainWithParams(scaledDataCats, param, 10 , new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter", scaledDataCats, model)
    }
    regResult.foreach{ case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.4f%%")}
    //0.001 L2 regularization parameter, AUC = 66.5483%
    //0.01 L2 regularization parameter, AUC = 66.5475%
    //0.1 L2 regularization parameter, AUC = 66.6338%
    //1.0 L2 regularization parameter, AUC = 66.0375%
     //10.0 L2 regularization parameter, AUC = 35.3253%

    val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4),123)
    val train = trainTestSplit(0)
    val test = trainTestSplit(1)
    val regResultsTest = Seq(0.0, 0.001,0.0025, 0.005, 0.01).map{ param =>
      val model = trainWithParams(train,param, 10, new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter", train, model)
    }
    regResultsTest.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.4f%%")}
    //0.0 L2 regularization parameter, AUC = 66.9638%
    //0.001 L2 regularization parameter, AUC = 66.9638%
    //0.0025 L2 regularization parameter, AUC = 66.9856%
    //0.005 L2 regularization parameter, AUC = 66.9856%
    //0.01 L2 regularization parameter, AUC = 66.9856%
  }
}
