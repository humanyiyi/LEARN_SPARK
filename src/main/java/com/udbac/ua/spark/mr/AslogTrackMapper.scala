package com.udbac.ua.spark.mr

import org.apache.commons.lang.StringUtils
import org.apache.commons.lang.StringUtils._
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.{HashMap, StringBuilder}
import scala.util.control.Breaks

/**
  * Created by root on 2017/2/23.
  */
object AslogTrackMapper {
  val vec = Array[String]("m2", "m1c", "m1a", "m9b", "m9", "m2a", "uuid",
    "m1", "m3", "m1b", "m9c", "mo")

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("AslogTrackMapper").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\As.log")

    data.map { line => log(line)}.foreach(println)
//      .saveAsTextFile("D:\\UDBAC\\LEARN_SPARK\\data\\output1")
  }

  def log(line: String): String ={
    val ua_hash = new HashMap[String, String]()
    var res = new String
    val token = line.split("\t")
    var uaid: String = null
    var daytime: String = null
    var auid: String = null

    if (token.length == 12) {
      daytime = token(0).substring(0, 19).replaceAll("[\\-:]", "").replace("T", " ")
      val uaStr = token(8)

      if (ua_hash.contains(uaStr)) {
        val uaid = ua_hash.get(uaStr)
      }
      else {
        val parsedUA = UAHashUtils.handleUA(uaStr)
        uaid = UAHashUtils.hashUA(parsedUA)
        ua_hash.put(uaStr, uaid)
      }
      val wxid = getWxid(token(5) + "," + token(6), token(10), token(2), token(8))
      val auid = UAHashUtils.hashUA(wxid)
      res ++= (daytime + "\t" + uaid + "\t" + auid)
    }
    res
  }


  def getWxid(aurl_aarg: String, auid: String, addr: String, uagn: String): String = {

    val loop = new Breaks
    var wxid: String = null
    val querys = aurl_aarg.split("[,&]")
    val queryMap = new HashMap[String, String]()
    for (query <- querys) {
      val kv = split(query, "=")
      if (kv.length == 2) {
        if (startsWith(kv(1), "__") && endsWith(kv(1), "__")) {
          kv(1) = null
        }
        queryMap.put(kv(0), kv(1))
      }
    }
    var flag = true
    for (ve: String <- vec if flag) {
      if (!queryMap.get(ve).isEmpty) {
//        println(queryMap.get(ve))
        val op = queryMap.get(ve)
        wxid = op match {case Some(o) => o}
        flag = false
      }
    }
    if (StringUtils.isBlank(wxid)) {
      if (auid.length > 0) {
        wxid = auid
      } else {
        wxid = addr + "#" + uagn
      }
    }
    wxid
  }

}
