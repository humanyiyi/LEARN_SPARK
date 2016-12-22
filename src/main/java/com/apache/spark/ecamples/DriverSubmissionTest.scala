package com.apache.spark.ecamples

import scala.collection.JavaConverters._
import org.apache.spark.util.Utils


/**
  * Created by root on 2016/12/14.
  */
object DriverSubmissionTest {
  def main(args: Array[String]) {
    if (args.length < 1) {
      println("Usage: DriverSubmissionTest <seconds-to-sleep>")
      System.exit(0)
    }
    val numSecondsToSleep = args(0).toInt

    val env = System.getenv()
    val properties = System.getProperties()

    println("Environment variables containing SPARK_TEST:")
    env.asScala.filter { case (k, _) => k.contains("SPARK_TEST")}.foreach(println)

    println("System properties containing spark.test:")
//    properties.filter { case (k, _) => k.toString.contains("spark.test") }.foreach(println)

    for (i <- 1 until numSecondsToSleep) {
      println(s"Alive for $i out of $numSecondsToSleep seconds")
      Thread.sleep(1000)
    }
  }
}