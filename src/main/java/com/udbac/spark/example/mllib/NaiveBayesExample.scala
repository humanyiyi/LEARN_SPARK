package com.udbac.spark.example.mllib


import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/8.
  */
object NaiveBayesExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("NaiveBayesExample")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,"D:\\test.txt")

    val Array(training,test) = data.randomSplit(Array(0.6,0.4))

    val model = NaiveBayes.train(training,lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features),p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
println("accuracy: "+ accuracy)
//    model.save(sc,"")
//    val sameModel = NaiveBayesModel.load(sc,"")
  }

}
