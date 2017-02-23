package com.udbac.ua.spark.mr

import org.apache.commons.lang.StringUtils
import org.apache.spark.SparkConf

import scala.collection.mutable
import scala.util.control.Breaks

/**
  * Created by root on 2017/2/21.
  */
object SdclogCookie {
  val fields = null
  val uaMap = new mutable.HashMap[String, String]()
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("").setMaster("")
  }

  def getquery(queryStr : String, fields:String*): String ={

    val queryMap = new mutable.HashMap[String, String]()
    val querys = StringUtils.split(queryStr, "&")
    for (query <- querys) {
      val kv = StringUtils.split(query, "=")
      kv.length match {case 2 => queryMap.put(kv(0),kv(2))}
    }
    val sb = new StringBuffer()
    val loop = new Breaks
    for (field <- fields) {
      if (field.contains("?")) {
        val fiesplits = StringUtils.split(field, "?")
         val fiel = for (fiesplit <- fiesplits) {
          if (queryMap.get(fiesplit) != null) {
            fiesplit
            loop.break()
          }
        }
      }

      sb.append(queryMap.get(field)).append("\t")
    }
    sb.substring(0,sb.length() - 1)
  }
}
