package com.udbac.spark.example.mllib

import java.nio.ByteBuffer
import java.util.{Random => JavaRandom}

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.util.hashing.MurmurHash3

import scala.collection.mutable

/**
  * Created by root on 2017/2/17.
  */

private case class VocabWord(
                              var word: String,  //分词
                              var cn: Int,//计数
                              var point: Array[Int],//存储路径，即经过得结点
                              var code: Array[Int],//记录Huffman编码
                              var codeLen: Int//存储到达该叶子结点，要经过多少个结点
                            )
class CWord2Vec extends Serializable{

  private val random = new JavaRandom()
  private var seed = new JavaRandom().nextLong()
  private var vectorSize = 100  //向量大小
  private var learningRate = 0.025   //学习率
  private var numPartitions = 1
  private var numIterations = 60 //迭代次数
  private var minCount = 5 //关键词的上下窗口
  private var maxSentenceLength = 1000  //每条语句以长度maxSentenceLength分组

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_CODE_LENGTH = 40
  private var window = 5
  private var trainWordsCount = 0L
  private var vocabSize = 0

  private var vocab: Array[VocabWord] = null
  private var vocabHash = mutable.HashMap.empty[String, Int]

  /*词典构建*/
  private def learnVocab[S <: Iterable[String]](dataset: RDD[S]){
    val words = dataset.flatMap(x => x)

    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)  //分词计数
      .filter(_._2 >= minCount)  //过滤频数少于minCount的分词
      .map(x => VocabWord(
        x._1,
        x._2,
        new Array[Int](MAX_CODE_LENGTH),
        new Array[Int](MAX_CODE_LENGTH),
        0))
      .collect()
      .sortWith((a, b) => a.cn > b.cn) //按频数从大到小排序

    vocabSize = vocab.length //词典的元素个数
    require(vocabSize >0 ,"The vocabulary size should be > 0. You may need to check the setting of minCount, which could be large enough to remove all your words in sentences.")

    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a  // 生成hashMap（K：word， V：a）--> 对词典中所有元素进行映射，方便查找
      trainWordsCount += vocab(a).cn  //计算语料C中分词的数量
      a += 1
    }

  }

  /*Create Huffman Tree*/
  private def createBinaryTree(): Unit ={
    val count = new Array[Long](vocabSize * 2 + 1)  //二叉树中的所有节点
    val binary = new Array[Int](vocabSize * 2 + 1)  //设置每个节点的Huffman编码： 左1， 右0
    val parentNode = new Array[Int](vocabSize *2 + 1) //存储每个节点的父节点
    val code = new Array[Int](MAX_CODE_LENGTH)  //存储每个叶子节点的Huffman编码
    val point = new Array[Int](MAX_CODE_LENGTH)  //存储每个叶子节点的路径（经历过那些节点）
    var a = 0
    while(a < vocabSize) {
      count(a) = vocab(a).cn  //初始化叶子节点，以频数作为权值，叶子： 0~vocabSize-1
      a += 1
    }
    while (a < 2 * vocabSize) {
      count(a) = 10^9   //10的9次方，非叶子节点，初始化为最大值
      a += 1
    }
    var pos1 = vocabSize - 1
    var pos2 = vocabSize

    var min1i = 0
    var min2i = 0

    a = 0
    while (a < vocabSize - 1) {
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        }
        else {
          min1i = pos2
          pos2 += 1
        }
      }else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        }else{
          min2i = pos2
          pos2 += 1
        }
      }else{
        min2i = pos2
        pos2 += 1
      }
      count(vocabSize + a) = count(min1i) + count(min2i)
      parentNode(min1i) = vocabSize + a
      parentNode(min2i) = vocabSize + a
      binary(min2i) = 1
      a += 1
    }
    var i = 0
    a = 0
    while (a < vocabSize) {
      var b = a
      i = 0
      while (b != vocabSize * 2 -2) {  //vocabSize * 2 - 2 表示根节点
        code(i) = binary(b)  //第b个节点的Huffman编码是0 or 1
        point(i) = b   //存储路径，经过b节点
        i += 1
        b = parentNode(b)
      }
      vocab(a).codeLen = i
      vocab(a).point(0) = vocabSize - 2
      b = 0
      while (b < i) {
        vocab(a).code(i - b - 1) = code(b)  //记录Huffman编码
        vocab(a).point(i - b) = point(b) - vocabSize //记录经过得节点
        b += 1
      }
      a += 1
    }
  }


  //创建sigmoid函数查询表
  private def createExpTable(): Array[Float] = { //初始化ExpTable，初始化参数为0-999的e值
  val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }

  def fit[S <: Iterable[String]](dataset: RDD[S]): Word2VecModel = {
    learnVocab(dataset)  //构建词典
    createBinaryTree()   //构建 Huffman 树

    val sc = dataset.context
    val expTable = sc.broadcast(createExpTable())
    val bcVocab = sc.broadcast(vocab)
    val bcVocabHash = sc.broadcast(vocabHash)

    val sentences: RDD[Array[Int]] = dataset.mapPartitions { sentenceIter =>
      // Each sentence will map to 0 or more Array[Int]
      sentenceIter.flatMap { sentence =>
        val wordIndexes = sentence.flatMap(bcVocabHash.value.get)// 将分词转化为对应的目录值（index）
        wordIndexes.grouped(maxSentenceLength).map(_.toArray) //一条语句长度大于1000后，将被拆分为多个分组
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val initRandom = new XORShiftRandom(seed)
    if (vocabSize.toLong * vectorSize >= Int.MaxValue) {
      throw new RuntimeException("vocabSize.toLong * vectorSize >= Int.MaxValue, " +
        "Int.MaxValue: " + Int.MaxValue)
    }

    //初始化叶子节点，分词向量随机设置初始值
    val syn0Global = Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    //初始化非叶子结点，参数向量设置初始值为0
    val syn1Global = new Array[Float](vocabSize * vectorSize)
    var alpha = learningRate //学习率

    for (k <- 1 to numIterations){ //对整个语料开始迭代，总共完成numIterations次迭代
    val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)

      //对每条句子进行向量计算：case中idx表示分词的目录，iter表示这条句子的起始地址
      val partial = newSentences.mapPartitionsWithIndex { case (idx, iter) =>
        val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
        val syn0Modify = new Array[Int](vocabSize)
        val syn1Modify = new Array[Int](vocabSize)
        val model = iter.foldLeft((bcSyn0Global.value, bcSyn1Global.value, 0L, 0L)) {
          case ((syn0, syn1, lastWordCount, wordCount), sentence) =>
            var lwc = lastWordCount
            var wc = wordCount
            if (wordCount - lastWordCount > 10000) {
              lwc = wordCount
              // TODO: discount by iteration?
              alpha =
                learningRate * (1 - numPartitions * wordCount.toDouble / (trainWordsCount + 1))
              if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
              //logInfo("wordCount = " + wordCount + ", alpha = " + alpha)
            }
            wc += sentence.length
            var pos = 0
            while (pos < sentence.length) {
              val word = sentence(pos) //这条句子中第pos个分词
              //在window范围内随机取出一个词b    window 表示中心词w上下最大各window个词。
              // 则最多一共2*window个词，即Context(w)的长度最大为2*window
              val b = random.nextInt(window)
              // Train Skip-gram
              var a = b
              while (a < window * 2 + 1 - b) {//此处循环是以pos为中心的skip-gram，即Context(w)中分词的向量计算
                if (a != window) {
                  val c = pos - window + a //c 是以 pos 为中心，所要表征Context(w)中的一个分词
                  if (c >= 0 && c < sentence.length) {
                    val lastWord = sentence(c) //c是通过pos词得到的，即Huffman树的叶子结点，也就是lastWord
                    val l1 = lastWord * vectorSize
                    val neu1e = new Array[Float](vectorSize) //用来存储Context(w)中各分词向量对分词w的贡献向量值

                    // Hierarchical softmax
                    var d = 0
                    //Huffman树中到达单词word，要经过结点数为 codeLen，这里从根节点开始遍历Huffman树
                    while (d < bcVocab.value(word).codeLen) {
                      val inner = bcVocab.value(word).point(d) //经过第d步时的结点
                      val l2 = inner * vectorSize
                      // Propagate hidden -> output
                      var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)//syn0 * syn1 两向量相乘
                      if (f > -MAX_EXP && f < MAX_EXP) {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        f = expTable.value(ind)
                        val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat
                        blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1) //neu1e = g * syn1 + neu1e
                        blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1) //syn1 = g * syn0 + syn1
                        syn1Modify(inner) += 1
                      }
                      d += 1
                    }
                    blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1) //syn0 = 1.0f * neu1e + syn0
                    syn0Modify(lastWord) += 1
                  }
                }
                a += 1
              }
              pos += 1
            }
            (syn0, syn1, lwc, wc)
        }
        val syn0Local = model._1 //syn0 为叶子结点向量，即分词向量
      val syn1Local = model._2 //syn1 为非叶子结点向量，即参数向量

        // Only output modified vectors.
        Iterator.tabulate(vocabSize) { index =>
          if (syn0Modify(index) > 0) {
            Some((index, syn0Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (syn1Modify(index) > 0) {
            Some((index + vocabSize, syn1Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten
      }

      //处理完每条句子的向量后，对所有语句中相同分词所对应的向量相加
      val synAgg = partial.reduceByKey { case (v1, v2) =>
        blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1) //v2 + v1
        v1
      }.collect()

      var i = 0
      while (i < synAgg.length) {
        val index = synAgg(i)._1
        if (index < vocabSize) {
          Array.copy(synAgg(i)._2, 0, syn0Global, index * vectorSize, vectorSize)
        } else {
          Array.copy(synAgg(i)._2, 0, syn1Global, (index - vocabSize) * vectorSize, vectorSize)
        }
        i += 1
      }
      bcSyn0Global.unpersist(false)
      bcSyn1Global.unpersist(false)
    }

    newSentences.unpersist()
    expTable.unpersist()
    bcVocab.unpersist()
    bcVocabHash.unpersist()

    val wordArray = vocab.map(_.word)
    new Word2VecModel(wordArray.zipWithIndex.toMap, syn0Global)
  }
}

class Word2VecModel  (
                       val wordIndex: Map[String, Int],
                       val wordVectors: Array[Float]) extends Serializable
{
  private val numWords = wordIndex.size
  private val vectorSize = wordVectors.length / numWords
  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }
  private val wordVecNorms: Array[Double] = {
    val wordVecNorms = new Array[Double](numWords)
    var i = 0
    while (i < numWords) {
      val vec = wordVectors.slice(i * vectorSize, i * vectorSize + vectorSize)
      wordVecNorms(i) = blas.snrm2(vectorSize, vec, 1)
      i += 1
    }
    wordVecNorms
  }

  def transform(word: String): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = wordVectors.slice(ind * vectorSize, ind * vectorSize + vectorSize)
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num)
  }

  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")
    // TODO: optimize top-k
    val fVector = vector.toArray.map(_.toFloat)
    val cosineVec = Array.fill[Float](numWords)(0)
    val alpha: Float = 1
    val beta: Float = 0
    // Normalize input vector before blas.sgemv to avoid Inf value
    val vecNorm = blas.snrm2(vectorSize, fVector, 1)
    if (vecNorm != 0.0f) {
      blas.sscal(vectorSize, 1 / vecNorm, fVector, 0, 1)
    }
    blas.sgemv(
      "T", vectorSize, numWords, alpha, wordVectors, vectorSize, fVector, 1, beta, cosineVec, 1)

    val cosVec = cosineVec.map(_.toDouble)
    var ind = 0
    while (ind < numWords) {
      val norm = wordVecNorms(ind)
      if (norm == 0.0) {
        cosVec(ind) = 0.0
      } else {
        cosVec(ind) /= norm
      }
      ind += 1
    }

    wordList.zip(cosVec)
      .toSeq
      .sortBy(-_._2)
      .take(num + 1)
      .tail
      .toArray
  }
}

private class XORShiftRandom(init: Long) extends JavaRandom(init) {

  private var seed = hashSeed(init)

  private def hashSeed(seed: Long): Long = {
    val bytes = ByteBuffer.allocate(java.lang.Long.SIZE).putLong(seed).array()
    val lowBits = MurmurHash3.bytesHash(bytes)
    val highBits = MurmurHash3.bytesHash(bytes, lowBits)
    (highBits.toLong << 32) | (lowBits.toLong & 0xFFFFFFFFL)
  }
  // we need to just override next - this will be called by nextInt, nextDouble,
  // nextGaussian, nextLong, etc.
  override protected def next(bits: Int): Int = {
    var nextSeed = seed ^ (seed << 21)
    nextSeed ^= (nextSeed >>> 35)
    nextSeed ^= (nextSeed << 4)
    seed = nextSeed
    (nextSeed & ((1L << bits) -1)).asInstanceOf[Int]
  }
}