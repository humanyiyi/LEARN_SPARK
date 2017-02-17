package com.udbac.spark.example.mllib

import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/2/17.
  */
object Word2VecExample {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Word2VecExample").setMaster("local")
    val sc = new SparkContext(conf)

    val input = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\mllib\\sample_lda_data.txt").map(line => line.split(" ").toSeq)

    val word2Vec = new Word2Vec()

    val model = word2Vec.fit(input)

    val synonyms = model.findSynonyms("1", 5)

    for ((synonym, cosineSimilarity) <- synonyms){
      println(s"$synonym      $cosineSimilarity")
    }
//    model.save(sc, "")
//    val sameModel = Word2VecModel.load(sc, "")

    sc.stop()
  }
}
