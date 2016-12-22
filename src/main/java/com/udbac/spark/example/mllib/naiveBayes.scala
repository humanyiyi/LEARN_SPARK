
package com.udbac.spark.example.mllib



import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/8.
  */
//朴素贝叶斯分类器
object naiveBayes {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("naivebayes")
    val sc = new SparkContext(conf)
    //读入数据
    val data = sc.textFile("D:\\naivebayes.txt")
    val parsedData = data.map{line =>
    val parts = line.split(",")
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }
    //把数据的60%作为训练集，40%作为测试集
    val splits = parsedData.randomSplit(Array(0.6,0.4),seed = 11L)
    val training = splits(0)
    val test = splits(1)
    //获取训练模型，第一个参数为数据，第二个参数为平滑参数，默认为1.0，可改
    val model = NaiveBayes.train(training,lambda = 1.0)
    //对模型进行准确度分析
    val predictionAndLabel = test.map(p => (model.predict(p.features),p.label))
    val accuracy = 1.0*predictionAndLabel.filter(x => x._1 == x._2).count()/test.count()

    println("accuracy-->" + accuracy)
    println("Predictionof (0.0, 2.0, 0.0, 1.0):"+model.predict(Vectors.dense(0.0,2.0,0.0,1.0)))
  }

}
