package com.udbac.spark.example.mllib

import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/17.
  */
object lda {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("lda").setMaster("local")
    val sc = new SparkContext(conf)

    //加载数据，返回的数据格式为： documents：RDD[(Long,Vector)]
    //其中：Long为文章的ID，Vector为文章分词后的词向量
    //可以读取指定目录下的数据，通过分词以及数据格式的转换，转换成RDD[(Long,Vector)]即可
    val data = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\sample_lda_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(" ").map(_.toDouble)))
    val corpus = parsedData.zipWithIndex.map(_.swap).cache()

    //建立模型，设置训练参数，训练模型
    /**
      * k: 主题数，或者聚类中心数
      * DocConcentration： 文章分布的超参数（Dirichlet分布的参数），必须 >1.0
      * TopicConcentration: 主题分布的超参数（Dirichlet分布的参数），必须 > 1.0
      * MaxIteration: 迭代次数
      * setSeed：随机种子
      * CheckpointInterval：迭代计算时检查点的间隔
      * Optimizer：优化计算方法，目前支持“em”，“online”
      */
    val ldaModel = new LDA()
      .setK(3)
      .setDocConcentration(5)
      .setTopicConcentration(5)
      .setMaxIterations(20)
      .setSeed(0L)
      .setCheckpointInterval(10)
      .setOptimizer("em")
      .run(corpus)

    //模型输出，模型参数输出，结果输出
    println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + "  words:")
    val topics =ldaModel.topicsMatrix
    for (topic <- Range(0,3)) {
      print("Topic " + topic + ":")
      for (word <- Range(0, ldaModel.vocabSize)) {print("\t" + topics(word, topic))}
      println()
    }
  }
}
