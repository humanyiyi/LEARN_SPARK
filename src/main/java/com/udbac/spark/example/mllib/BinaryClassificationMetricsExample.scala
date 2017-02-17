package com.udbac.spark.example.mllib

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/15.
  */
object BinaryClassificationMetricsExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("BinaryClassificationMetricsExample").setMaster("local")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, "D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\sample_binary_classification_data.txt")

    val Array(training, test) = data.randomSplit(Array(0.6,0.4))
    training.cache()

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    //清除预测阈值，使模型返回概率
    model.clearThreshold()

    val predictionAndLabels = test.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    //实例化度量对象
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    //Precision by threshold  通过阈值精确
    val precision = metrics.precisionByThreshold()
    precision.foreach{case (t, p) =>
    println(s"Threshold: $t , Precision: $p")
    }

    //Recall by threshold   通过阈值召回
    val recall = metrics.recallByThreshold()
    recall.foreach{ case (t, r) =>
    println(s"Threshold: $t, Recall: $r")
    }

    //Precision_Recall Curve
    val PRC = metrics.pr()

    //F-measure
    val flScore = metrics.fMeasureByThreshold()
    flScore.foreach{ case (t, f) =>
    println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    fScore.foreach{ case (t, f) =>
    println(s"Threshold：$t, F-score: $f, Beta = 0.5")
    }

    //AUPRC
    val auPRC = metrics.areaUnderPR()
    println("Area under precision-recall curve = " + auPRC)

    //compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    //ROC Curve
    val roc = metrics.roc()
    println("roc: " + roc)

    //AUROC  召回率
    val auROC = metrics.areaUnderROC()
    println("Area under ROC = " + auROC)

  }
}
