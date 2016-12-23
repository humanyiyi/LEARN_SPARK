package com.apache.spark.mllib.classifier


import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/23.
  */
object DecisionTreeClassifier {
  def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
    DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
  }
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Decision Tree Classifier")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile(args(0))
    val records = rowData.map(line => line.split("\t"))
    val data = records.map{ r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map( d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    val numData = data.count
    val maxTreeDepth = 5
    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)
    val dtTotalCorrect = data.map { point =>
      if (dtModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val dtAccuracy = dtTotalCorrect / numData
    println(dtAccuracy) //0.6482758620689655

    val dtMetrics = Seq(dtModel).map{ model =>
      val scoreAndLabels = data.map{ point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0,point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName,metrics.areaUnderPR,metrics.areaUnderROC)

    }
    println(dtMetrics) //List((DecisionTreeModel,0.7430805993331199,0.6488371887050935))

    val dtResultsEntropy = Seq(1, 2,3, 4, 5, 10, 20).map{ param =>
      val model = trainDTWithParams(data, param, Gini)
      val scoreAndLabels = data.map{ point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param tree depth" , metrics.areaUnderROC)
    }
    dtResultsEntropy.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.4f%%")}
  }
}
