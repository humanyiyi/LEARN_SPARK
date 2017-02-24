package com.udbac.ua.spark.mr

import org.apache.commons.lang.StringUtils
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.HashMap

/**
  * Created by root on 2017/2/21.
  */
object SdclogCookie {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("SdclogCookie").setMaster("local")
    val sc = new SparkContext(conf)
    //    val fields:Array[String] = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\1.txt").collect()

    val data = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\test.log")
    data.map { line => (logMap(line, QueryProperties.query()), "") }.groupByKey().keys.saveAsTextFile("")
//      .collect().foreach(println)

  }

  def logMap(line: String, fields: Array[String]): String = {
    //    val fields = Array("WT.mobile","WT.a_cat")
    val uaMap = new HashMap[String, String]()
    val tokens = StringUtils.split(line, " ")
    var res = new String
    if (tokens.length == 15) {
      val queryed = getquery(tokens(7), fields)
      val uaString = tokens(11).replaceAll("[+]", " ")
      val uaHash = UAHashUtils.hashUA(uaString)
      var uaDemension: String = null
      if (uaMap.contains(uaHash)) {
        val ua = uaMap.get(uaHash)
        uaDemension =
          ua match {case Some(u) => u}
      } else {
        uaDemension = UAHashUtils.handleUA(uaString)
        uaMap.put(uaHash, uaDemension)
      }
      res ++= queryed + "\t" + uaDemension
    }
    res
  }

  def getquery(queryStr: String, fields: Array[String]): String = {

    val queryMap = new HashMap[String, String]()
    val querys = StringUtils.split(queryStr, "&")
    for (query <- querys) {
      val kv = StringUtils.split(query, "=")
      kv.length match {
        case 2 => queryMap.put(kv(0), kv(1))
      }
    }
    val sb = new StringBuffer()
    var fie: String = null
    for (field <- fields) {
      if (field.contains("?")) {
        val fiesplits = StringUtils.split(field, "?")
        var flag = true
        for (fiesplit: String <- fiesplits if flag) {
          if (!queryMap.get(fiesplit).isEmpty) {
            fie = fiesplit
            flag = false
          }
        }
      } else {
        fie = field
      }
      sb.append(queryMap.get(fie) match { case Some(q) => q }).append("\t")
    }
    sb.substring(0, sb.length() - 1)
  }
}
