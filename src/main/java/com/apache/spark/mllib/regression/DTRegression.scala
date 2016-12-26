package com.apache.spark.mllib.regression

import breeze.linalg.sum
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/26.
  */
object DTRegression {
  def get_mapping(rdd: RDD[Array[String]], idx: Int)={
    rdd.map(filed => filed(idx)).distinct().zipWithIndex().collectAsMap()
  }
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("DTRegression")
    val sc = new SparkContext(conf)
    val rowData = sc.textFile(args(0))
    val records = rowData.map(line => line.split(","))
    records.cache()
    val mappings = for (i <- Range(2, 10)) yield get_mapping(records, i)

    val cat_len = sum(mappings.map(_.size))
    val num_len = records.first().slice(10, 14).size
    val total_len = cat_len + num_len

    val data = records.map{ record =>
      val features = record.slice(2, 14).map(_.toDouble)
      val label = record(record.size - 1).toDouble
      LabeledPoint(label, Vectors.dense(features))
    }
    val categoricalFeaturesInfo = Map[Int, Int]()
    val tree_model = DecisionTree.trainRegressor(data,categoricalFeaturesInfo,"variance",5,23)
    val true_vs_predicted = data.map(p => (p.label, tree_model.predict(p.features)))
    println( true_vs_predicted.take(5).toVector.toString())

    val MSE = true_vs_predicted.map(value =>
    {
      (value._1-value._2)*(value._1-value._2)
    }).mean()

    val MAE = true_vs_predicted.map(value =>
    {
      math.abs(value._1-value._2)
    }).mean()
    val RMSLE = true_vs_predicted.map(value =>
    {
      math.pow(math.log(value._1+1)-math.log(value._2+1),2)
    }).mean()
    println("MSE: " + MSE + "\nMAE: " + MAE + "\nRMSLE: " + RMSLE)
  }
}
