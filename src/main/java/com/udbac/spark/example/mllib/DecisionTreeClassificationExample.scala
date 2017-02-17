package com.udbac.spark.example.mllib

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/15.
  */
object DecisionTreeClassificationExample {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DecisionTreeClassificationExample").setMaster("local")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,"D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\sample_libsvm_data.txt")

    val splits = data.randomSplit(Array(0.7,0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData,numClasses,categoricalFeaturesInfo,impurity,maxDepth,maxBins)   // maxDepth 树的最大深度   categoricalFeaturesInfo分类功能信息  numClasses分类的类数

    val labelAndPreds = testData.map{ point =>
    val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model: \n" + model.toDebugString)

    model.save(sc,"D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\output")
    val sameModel = DecisionTreeModel.load(sc,"D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\output")
  }
}
