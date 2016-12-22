package com.apache.spark.ecamples

import java.io.File
import scala.io.Source._

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/13.
  */
object DFSReadWriteTest {

  private var localFilePath: File = new File(".")
  private var dfsDirPath: String = ""

  private val NPARAMS = 2

  private def readFile(filename: String): List[String] = {
    val lineIter: Iterator[String] = fromFile(filename).getLines()
    val lineList: List[String] = lineIter.toList
    lineList
  }

  private def printUsage(): Unit = {
    val usage: String = "DFS Read-Write Test\n" +
    "\n" +
    "Usage: localFile dfsDir\n" +
    "\n" +
    "localFile - (string) local file to use in test\n" +
    "dfsDir - (string) DFS directory for read/write tests\n"

    println(usage)
  }

  private def parseArgs(args: Array[String]): Unit ={
    if (args.length != NPARAMS){
      printUsage()

      System.exit(1)
    }
    var i = 0

    localFilePath = new File(args(i))
    if (!localFilePath.isFile){
      System.err.println("Given path (" + args(i) + ") is not a file.\n")
      System.exit(1)
    }
    i += 1
    dfsDirPath = args(i)
  }

  def runLocalWordCount(fileContents: List[String]): Int = {
    fileContents.flatMap(_.split(" "))
      .flatMap(_.split("\t"))
      .filter(_.nonEmpty)
      .groupBy(w => w)
      .mapValues(_.size)
      .values
      .sum
  }

  def main(args: Array[String]): Unit = {
    parseArgs(args)

    println("Performing local word count")
    val fileContents = readFile(localFilePath.toString)
    val localWordCount = runLocalWordCount(fileContents)

    println("Createing SparkContext")

    val conf = new SparkConf().setMaster("local").setAppName("DFSReadWriteTest")
    val sc = new SparkContext(conf)

    println("Writing local file to DFS")
    val dfsFilename = dfsDirPath + "\\wordcount.txt"
    val fileRDD = sc.parallelize(fileContents)
    fileRDD.saveAsTextFile(dfsFilename)

    println("Reading file from DFS and running Word Count")
    val readFileRDD = sc.textFile(dfsFilename)

    val dfsWordCount = readFileRDD
      .flatMap(_.split(" "))
      .flatMap(_.split("\t"))
      .filter(_.nonEmpty)
      .map(w => (w, 1))
      .countByKey()
      .values.sum

    sc.stop()

    if (localWordCount == dfsWordCount) {
      println(s"Succes! Local Word Count ($localWordCount) " + s"and DFS Word Count ($dfsWordCount) argee.")
    }else {
      println(s"Failure! Local Word Count ($localWordCount) " + s"and DFS Word Count ($dfsWordCount) disagee." )
    }
  }

}
