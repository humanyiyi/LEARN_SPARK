package com.udbac.ua.spark.mr

import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.HashSet

/**
  * Created by root on 2017/2/23.
  */
object AslogUnique {
  def main(args: Array[String]): Unit = {


    val conf = new SparkConf().setAppName("AslogUnique").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\As.log")

    data.map(line => (log(line),"")).groupByKey().keys.saveAsTextFile("D:\\UDBAC\\LEARN_SPARK\\data\\output1")
//      saveAsTextFile("D:\\UDBAC\\LEARN_SPARK\\data\\output1")

  }


  def log(line: String): String ={
    val uaSet = new HashSet[String]()
    var res = new String
    val tokens = line.split("\t")
    if (tokens.length == 12) {
      val uaString = tokens(8)
      val uaHash = UAHashUtils.hashUA(uaString)
      if (uaSet.contains(uaHash)){}
      uaSet.add(uaHash)
      val parsedUA = UAHashUtils.handleUA(uaString)
      res = UAHashUtils.hashUA(parsedUA) + "\t" + parsedUA
    }
    res
  }

}
