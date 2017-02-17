package com.apache.spark.mllib.regression

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/15.
  */
object RegressionMetricsExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RegressionMetricsExample").setMaster("local")
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)
    //加载数据
    val data = MLUtils.loadLibSVMFile(sc,"D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\sample_linear_regression_data.txt").cache()

    //Build the model
    val numIterations = 100
    val model = LinearRegressionWithSGD.train(data,numIterations)  //SGD: stochastic gradient descent  线性回归


    //Get predictions
    val valuesAndPreds = data.map{ point =>
      val prediction = model.predict(point.features)
      (prediction, point.label) //(预测值，实际值)
    } //返回的是一个RDD数据类型

    println(valuesAndPreds.getClass)   //    class org.apache.spark.rdd.MapPartitionsRDD   RDD数据类型


    println("value and predict")

//    valuesAndPreds.foreach(println)

    //Instantiate metrics object
    val metrics = new RegressionMetrics(valuesAndPreds)

    println("Model Parameter")
    var i = 1
    model.weights.toArray.foreach(
      a =>{
        println("parameter" + i + ":" + a)
        i+=1
      }
    )
    println("model intercept: " + model.intercept)

    println(s"MSE = ${metrics.meanSquaredError}")  //平均平方误差
    println(s"RMSE = ${metrics.rootMeanSquaredError}")  //标准平均平方误差

    //R-squared
    println(s"R-squared = ${metrics.r2}")

    //Mean absolute error
    println(s"MAE = ${metrics.meanAbsoluteError}")

    //Explained variance
    println(s"Exaplained variance = ${metrics.explainedVariance}")

    sc.stop()
  }

}
