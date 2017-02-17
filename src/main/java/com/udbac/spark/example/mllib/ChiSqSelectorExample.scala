package com.udbac.spark.example.mllib

import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/16.
  */

/**
  *基于卡方校验的特征选择
  * 卡方校验：
  * 在分类资料统计推断中一般检验一个样本是否符合预期的一个分类
  * 是统计样本的实际值与理论推断值之间的偏离程度
  * 卡方值越小，越趋于符合
  */
object ChiSqSelectorExample {  //卡方选择
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ChiSqSelectorExample").setMaster("local")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, "D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\sample_libsvm_data.txt")
    val discretizedData = data.map{ lp =>   //创建数据处理空间
      LabeledPoint(lp.label,Vectors.dense(lp.features.toArray.map{x => (x / 16).floor}))  //floor 向下取整
    }

    val selector = new ChiSqSelector(50)  //创建选择50个特征的卡方校验
    val transformer = selector.fit(discretizedData)  //创建训练模型
    val filteredData = discretizedData.map{lp =>     //过滤前50个特征
      LabeledPoint(lp.label, transformer.transform(lp.features))
    }
    println(filteredData.count())
    println("filtered data: ")
    filteredData.foreach(x => println(x))

    sc.stop()
  }
}
