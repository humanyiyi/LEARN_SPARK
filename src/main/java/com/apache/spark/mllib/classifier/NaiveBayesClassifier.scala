package com.apache.spark.mllib.classifier

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/22.
  */
object NaiveBayesClassifier {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Naive Bayes Classifier")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile(args(0))
    val records = rowData.map(line => line.split("\t"))


    val nbData = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", "")) //去掉数据中多余的引号
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map( d => if( d == "?") 0.0 else d.toDouble)
        .map(d => if (d < 0) 0.0 else d)   //朴素贝叶斯模型要求特征值非负。将负特征值设为0
      LabeledPoint(label, Vectors.dense(features))
    }
    val nbModel = NaiveBayes.train(nbData)
    val numData = nbData.count
//    println(nbModel)   //org.apache.spark.mllib.classification.NaiveBayesModel@5853495b

//    val dataPoint = nbData.first
//    val prediction = nbModel.predict(dataPoint.features)
//    println(prediction)  //对训练数据中的第一个样本模型预测值为1
//    println(dataPoint.label)  //实际标签为0.0

    val nbTotalCorrect = nbData.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracy = nbTotalCorrect / numData
    println(nbAccuracy)  //正确率0.5803921568627451

    val nbMetrics = Seq(nbModel).map { model =>
      val scoreAndLabels = nbData.map{ point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }
//    println(nbMetrics) //准确率List((NaiveBayesModel,0.6808510815151734,0.5835585110136261))

    //1-of-k编码的类型特征
    val categories = records.map( r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size
//    println(numCategories)  //类别数14
    val dataNB = records.map{ r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      LabeledPoint(label, Vectors.dense(categoryFeatures))
    }
    val nbModelCats = NaiveBayes.train(dataNB)
//    println(nbModelCats.predict(dataNB.first.features))  //第一个样本的预测结果为1.0

    /**1-of-k 编码类型特征的朴素贝叶斯模型性能评估*/
    val nbTotalCorrectCats = dataNB.map{ point =>
      if (nbModelCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracyCats = nbTotalCorrectCats / numData
    val nbPredictionsVsTrueCats = dataNB.map { point =>
      (nbModelCats.predict(point.features), point.label)
    }
    val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
    val nbPrCats = nbMetricsCats.areaUnderPR
    val nbRocCats = nbMetricsCats.areaUnderROC
    println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy: ${nbAccuracyCats * 100}%2.4f%%\nArea under PR: ${nbPrCats * 100.0}%2.4f%%\nArea under ROC: ${nbRocCats * 100.0}%2.4f%%")
    //NaiveBayesModel
    //Accuracy: 60.9601%
    //Area under PR: 74.0522%
    //Area under ROC: 60.5138%
  }
}
