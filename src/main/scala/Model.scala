import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

object Model{
  def main(args: Array[String]): Unit = {

    val spark = SparkNLP.start()

    val myDF = spark.read.text("data/text.txt").toDF("text")
    myDF.show(false)

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")


    val token = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val embedding = WordEmbeddingsModel.pretrained("glove_100d", "en")
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("embedding")

    val pos = PerceptronModel.pretrained("pos_anc", "en")
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("pos")

    val ner = NerCrfModel.pretrained("ner_crf", "en")
      .setInputCols(Array("sentence", "token", "pos", "embedding"))
      .setOutputCol("ner")

    val pipeline = new Pipeline().setStages(
      Array
      (
        document,
        sentence,
        token,
        embedding,
        pos,
        ner
      )
    )

    val model = pipeline.fit(myDF)

    val processedDf = model.transform(myDF)
    processedDf.show(false)

    processedDf.select(
      col("pos.result").alias("POS"),
      col("ner.result").alias("NER")
    ).show(false)

    val zippedDf = processedDf.withColumn(
      "pos_ner_pairs",
      expr("arrays_zip(pos.result, ner.result)")
    )

    zippedDf.select(
      col("pos_ner_pairs")
    ).show(10000,false)

    zippedDf.printSchema()

    val explodeDf = zippedDf.select(explode(col("pos_ner_pairs").alias("col")))
    explodeDf.show(100000,false)

    val pairCounts = explodeDf.groupBy("col").count()
    pairCounts.show(10000,false)

    pairCounts.printSchema()

//    val POS_NER_COUNT = pairCounts .withColumn("POS" ,col("col").getItem(0))
//      .withColumn("NER" ,col("col").getItem(1))
//    POS_NER_COUNT.show(10000,false)


    //    exploded_df = df.select(explode(col("pos_ner_pairs")).alias("pair"))
//
//    pair_counts = exploded_df.groupBy("pair").count()
//
//    pair_counts.show(truncate=False)



    //
//    val document = new DocumentAssembler()
//      .setInputCol("text")
//      .setOutputCol("document")
//
//    val token = new Tokenizer()
//      .setInputCols("document")
//      .setOutputCol("token")
//
//    val normalizer = new Normalizer()
//      .setInputCols("token")
//      .setOutputCol("normal")
//
//    val finisher = new Finisher()
//      .setInputCols("normal")
//
//    val ngram = new NGram()
//      .setN(3)
//      .setInputCol("finished_normal")
//      .setOutputCol("3-gram")
//
//    val pipeline = new Pipeline().setStages(Array(document, token, normalizer, finisher, ngram))
//
//    val model = pipeline.fit(myDF)
//
//    val processedDF = model.transform(myDF)
//    processedDF.show(false)

    //    val document = new DocumentAssembler()
//      .setInputCol("text")
//      .setOutputCol("document")
//
//    val token = new Tokenizer()
//      .setInputCols("document")
//      .setOutputCol("token")
//
//    val normalizer = new Normalizer()
//      .setInputCols("token")
//      .setOutputCol("normal")
//
//    val wordEmbeddings = WordEmbeddingsModel.pretrained()
//      .setInputCols("document", "token")
//      .setOutputCol("word_embeddings")
//
//    val ner = NerDLModel.pretrained()
//      .setInputCols("normal", "document", "word_embeddings")
//      .setOutputCol("ner")
//
//    val nerConverter = new NerConverter()
//      .setInputCols("document", "normal", "ner")
//      .setOutputCol("ner_converter")
//
//    val finisher = new Finisher()
//      .setInputCols("ner", "ner_converter")
//      .setIncludeMetadata(true)
//      .setOutputAsArray(false)
//      .setCleanAnnotations(false)
//      .setAnnotationSplitSymbol("@")
//      .setValueSplitSymbol("#")
//
//    val pipeline = new Pipeline().setStages(Array(document, token, normalizer, wordEmbeddings, ner, nerConverter, finisher))
//
//    val model = pipeline.fit(myDF)
//
//    val processedDF = model.transform(myDF)
//
//    processedDF.show(false)

//    val documentAssembler = new DocumentAssembler()
//      .setInputCol("text")
//      .setOutputCol("document")
//      .setCleanupMode("shrink")
//
//    val sentenceDetector = new SentenceDetector()
//      .setInputCols(Array("document"))
//      .setOutputCol("sentence")
//
//    val tokenizer = new Tokenizer()
//      .setInputCols(Array("sentence"))
//      .setOutputCol("tokens")
//      .setExceptions(Array("e-mail"))
//
//    val checker = NorvigSweetingModel.pretrained()
//      .setInputCols(Array("tokens"))
//      .setOutputCol("checked")
//
//    val embedding = WordEmbeddingsModel.pretrained()
//      .setInputCols(Array("sentence","checked"))
//      .setOutputCol("embeddings")
//
//    val ner = NerDLModel.pretrained()
//      .setInputCols(Array("sentence","checked","embeddings"))
//      .setOutputCol("ner")
//
//    val converter = new NerConverter()
//      .setInputCols(Array("sentence","checked","ner"))
//      .setOutputCol("chunk")
//
//    val pipeline = new Pipeline()
//      .setStages(Array(
//        documentAssembler,
//        sentenceDetector,
//        tokenizer,
//        checker,
//        embedding,
//        ner,
//        converter
//      )
//      )
//
//    val model = pipeline.fit(myDF)
//
//    val processedDF = model.transform(myDF)
//    processedDF.show(false)
  }
}
