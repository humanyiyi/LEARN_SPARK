package com.udbac.spark.example.mllib

import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/21.
  */
object NormalizerExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("NormalizerExample").setMaster("local")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,"D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\sample_libsvm_data.txt")

    val normalizer1 = new Normalizer()
    val normalizer2 = new Normalizer(p = Double.PositiveInfinity)

    val data1 = data.map(x => (x.label, normalizer1.transform(x.features)))
    val data2 = data.map(x => (x.label, normalizer2.transform(x.features)))

    println("data1: ")
    data1.foreach(x => println(x))

    println("data2: ")
    data2.foreach(x => println(x))

    sc.stop()
  }
}
