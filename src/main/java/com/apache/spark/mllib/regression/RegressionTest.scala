package com.apache.spark.mllib.regression

import breeze.linalg.sum
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/26.
  */
object RegressionTest {
  def get_mapping(rdd: RDD[Array[String]], idx: Int) = {
    rdd.map(filed => filed(idx)).distinct().zipWithIndex().collectAsMap()
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Regression Test")
    val sc = new SparkContext(conf)
    val rowData = sc.textFile(args(0))
    val records = rowData.map(line => line.split(","))
    val numData = rowData.count()
    records.cache()
    val mappings = for (i <- Range(2, 10)) yield get_mapping(records, i)

    val cat_len = sum(mappings.map(_.size))
    val num_len = records.first().slice(10, 14).size
    val total_len = cat_len + num_len
//   println(cat_len, num_len, total_len)
    val data = records.map { record =>
      val cat_vec = Array.ofDim[Double](cat_len)
      var i = 0
      var step = 0
      for (filed <- record.slice(2, 10)) {
        val m = mappings(i)
        val idx = m(filed)
        cat_vec(idx.toInt + step) = 1.0
        i = i + 1
        step = step + m.size
      }
      val num_vec = record.slice(10, 14).map(x => x.toDouble)

      val features = cat_vec ++ num_vec
      val label = record(record.size - 1).toInt

      LabeledPoint(label, Vectors.dense(features))
    }

    //    val linear_model = LinearRegressionWithSGD.train(data, 10, 0.1) //Vector((16.0,135.94648455498353), (40.0,134.38058174607252), (32.0,134.18407938613737), (13.0,133.88699144084512), (1.0,133.77899037657545))
    val linear_model = LinearRegressionWithSGD.train(data, 10, 0.5)
    val true_vs_predicted = data.map(p => (p.label, linear_model.predict(p.features)))
    println(true_vs_predicted.take(5).toVector.toString())
  }
}
