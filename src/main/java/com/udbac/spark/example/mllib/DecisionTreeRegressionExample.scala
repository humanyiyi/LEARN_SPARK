package com.udbac.spark.example.mllib

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/15.
  */
object DecisionTreeRegressionExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DecisionTreeRegressionExample").setMaster("local")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, "D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\sample_libsvm_data.txt")
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val categoricalFeaturesInfo = Map[Int, Int]()
    val immutable = "variance"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, immutable, maxDepth,maxBins)

    val labelsAndPredictions = testData.map{
     point => val prediction = model.predict(point.features)
       (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{case (v, p) =>
    math.pow(v -p ,2)}.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression tree model: \n" + model.toDebugString)

    model.save(sc, "D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\output")
    val sameModel = DecisionTreeModel.load(sc,"D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\output")
  }
}
