package com.apache.spark.mllib.classifier

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2017/1/5.
  */
object NewsDateTrain {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("NewsDateTrain")
    val sc = new SparkContext(conf)

    val rdd = sc.wholeTextFiles("D:\\UDBAC\\LEARN_SPARK\\data\\20_newsgroups\\*")
    val text = rdd.map{ case (file, text) => text}
//    println(text.count)
    val newsgroups = rdd.map{ case (file,text) =>
        file.split("/").takeRight(2).head
    }
    val countByGroup = newsgroups.map(n => (n, 1)).reduceByKey(_ + _).collect.sortBy(-_._2).mkString("\n")
    println(countByGroup)
    val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_.toLowerCase))
    println(whiteSpaceSplit.distinct.count)
    println(whiteSpaceSplit.sample(true, 0.3, 42).take(100).mkString(","))
    //移除不是单词的字符
    val nonWordSplit = text.flatMap(t =>
      t.split("""\W+""").map(_.toLowerCase))
    println(nonWordSplit.distinct.count)

    println(nonWordSplit.distinct.sample(true, 0.3, 42).take(100).mkString(","))

    //过滤掉数字和包含数字的单词
    val regex = """[^0-9]*""".r
    val filterNumbers = nonWordSplit.filter(token =>
      regex.pattern.matcher(token).matches)
    println(filterNumbers.distinct.count)

    println(filterNumbers.distinct.sample(true, 0.3, 42).take(100).mkString("\n"))

    val tokenCounts = filterNumbers.map(t => (t, 1)).reduceByKey(_ + _)
    val oreringDesc = Ordering.by[(String, Int),Int](_._2)
    println(tokenCounts.top(20)(oreringDesc).mkString("\n"))

    //过滤掉停用词
    val stopwords = Set("the", "a", "an", "of", "or", "in", "for", "by", "on", "but", "is", "not", "with", "as", "was", "if", "they", "are", "this", "and",
    "it", "have", "from", "at", "my", "be", "that", "to")
    val tokenCountsFilteredStopwords = tokenCounts.filter{case(k,v) => !stopwords.contains(k)}
    println(tokenCountsFilteredStopwords.top(20)(oreringDesc).mkString("\n"))

    //过滤掉仅含一个字符的单词
    val tokenCountsFilteredSize = tokenCountsFilteredStopwords.filter{ case (k, v) => k.size >=2}
    println(tokenCountsFilteredSize.top(20)(oreringDesc).mkString("\n"))

    //去掉文本中出现频率很低的单词
    val oreringAsc = Ordering.by[(String, Int), Int](-_._2)
    println(tokenCountsFilteredSize.top(20)(oreringAsc).mkString("\n"))

    //过滤掉出现次数很少的单词
    val rareTokens = tokenCounts.filter{ case(k,v) => v < 2}.map{case (k,v) => k}.collect.toSet
    val tokenCountsFilteredAll = tokenCountsFilteredSize.filter{ case (k, v) => !rareTokens.contains(k)}
    println(tokenCountsFilteredAll.top(20)(oreringAsc).mkString("\n"))

    //过滤逻辑组织到一个函数中
    def tokenize(line: String): Seq[String] = {
      line.split("""\W+""")
        .map(_.toLowerCase)
        .filter(token => regex.pattern.matcher(token).matches)
        .filterNot(token => stopwords.contains(token))
        .filterNot(token => rareTokens.contains(token))
        .filterNot(token => token.size >= 2)
        .toSeq
    }
    println(text.flatMap(doc => tokenize(doc)).distinct.count)

    val tokens = text.map(doc => tokenize(doc))
    println(tokens.first.take(20))
  }
}
