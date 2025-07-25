import numpy as np
from pyspark.sql.functions import from_unixtime, length, year, col, udf, row_number, collect_list, concat_ws
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, IntegerType, StringType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import BucketedRandomProjectionLSH, Normalizer, HashingTF, MinHashLSH
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, StopWordsCleaner, LemmatizerModel, BertSentenceEmbeddings


# some books have duplicates, with some differences in the title. We only keep one of each
def load_duplicated_titles(file_path = "https://github.com/lucasangio01/amazon-books/blob/master/data/duplicated_titles.txt"):

    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_data(spark, chosen_data):

    duplicated_titles = load_duplicated_titles()
    duplicated_df = spark.createDataFrame([(title,) for title in duplicated_titles], ["book_title"])

    reviews_schema = StructType([
        StructField("Id", IntegerType(), True),
        StructField("Title", StringType(), True),
        StructField("Price", IntegerType(), True),
        StructField("User_id", StringType(), True),
        StructField("profileName", StringType(), True),
        StructField("review/helpfulness", StringType(), True),
        StructField("review/score", FloatType(), True),
        StructField("review/time", IntegerType(), True),
        StructField("review/summary", StringType(), True),
        StructField("review/text", StringType(), True)
    ])

    books_data_schema = StructType([
        StructField("Title", StringType(), True),
        StructField("description", StringType(), True),
        StructField("categories", StringType(), True)
    ])

    original_df = (
        spark.read.csv("../content/drive/MyDrive/Colab Notebooks/Books_rating.csv", header=True, schema=reviews_schema)
        .drop("Price", "profileName", "review/summary")
        .filter(length("review/text").between(300, 600))  # only keeping reviews between 300 and 600 characters
        .dropna()
        .withColumn("review_year", year(from_unixtime("review/time")))
        .drop("review/time")
        .withColumn("review/score", col("review/score").cast(IntegerType()))
        .withColumnRenamed("Id", "book_id")
        .withColumnRenamed("Title", "book_title")
        .withColumnRenamed("User_id", "user_id")
        .withColumnRenamed("review/helpfulness", "review_helpfulness")
        .withColumnRenamed("review/score", "review_score")
        .withColumnRenamed("review/text", "review_text")
        .cache()
        )

    reviews_per_book = 50

    if chosen_data == "big_only_fiction":  # choose if you want to analyze only the "fiction" books or all the categories
        min_reviews = 50
    elif chosen_data == "big_full":  # choose if you want to analyze the books of all the categories
        min_reviews = 105
    elif chosen_data == "trial_full":  # choose if you want to analyze a smaller subset of the data, for faster (but less informative) results
        min_reviews = 260

    famous_titles = original_df.select("book_title").groupBy("book_title").count().filter(
        col("count") > min_reviews).drop("count")  # select the books with a certain number of reviews
    df_reviews = original_df.select("book_title", "review_text").join(famous_titles, on="book_title", how="inner").join(
        duplicated_df, on="book_title", how="left_anti")
    original_df.unpersist()

    """
    Bigger dataset: only fiction (6750 x 135), full (6750 x 135)
    Trial dataset: full(800 x 16)
    """

    if chosen_data == "big_only_fiction":  # choose if you want to analyze only the "fiction" books
        titles_list = df_reviews.select("book_title").distinct()
        books_data_full = spark.read.csv("../content/drive/MyDrive/Colab Notebooks/books_data.csv", inferSchema=True,
                                         header=True).select("Title", "categories", "description").withColumnRenamed(
            "Title", "book_title").dropna().filter(col("categories") == "['Fiction']").join(titles_list,
                                                                                            on="book_title",
                                                                                            how="inner").cache()
        books_with_description = books_data_full.select("book_title", "description").cache()

    window_spec = Window.partitionBy("book_title").orderBy("review_text")
    df_reviews = df_reviews.withColumn("row_num", row_number().over(window_spec)).filter(
        col("row_num") <= reviews_per_book)  # select how many reviews to use for the analysis, for each book

    if chosen_data == "big_only_fiction":
        df_reviews = df_reviews.drop("row_num", "description").join(books_data_full, on="book_title",
                                                                    how="inner").cache()
        df_descriptions = books_data_full.select("book_title", "description").cache()
        return df_reviews, df_descriptions
    else:
        df_reviews = df_reviews.drop("row_num").cache()
        return df_reviews, None


def pretrained_pipeline(dataset):

  if dataset == df_reviews:
    text_column = "review_text"
  else:
    text_column = "description"

  document_assembler = DocumentAssembler().setInputCol(text_column).setOutputCol("document")
  sentence_embeddings = BertSentenceEmbeddings().pretrained("sent_small_bert_L2_128", "en").setInputCols(["document"]).setOutputCol("embeddings")
  pipeline = Pipeline(stages = [document_assembler, sentence_embeddings])

  result = pipeline.fit(dataset).transform(dataset)
  extract_embedding = udf(lambda arr: arr[0].embeddings, ArrayType(FloatType()))
  result = result.withColumn("embedding_vector", extract_embedding("embeddings")).select("book_title", "embedding_vector")

  to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())
  result = result.withColumn("embedding_vector_ml", to_vector_udf("embedding_vector")).select("book_title", "embedding_vector_ml")

  return result


def custom_pipeline(dataset):

  if dataset in (df_reviews, customized_reviews_grouped):
    text_column = "review_text"
  else:
    text_column = "description"

  document_assembler = DocumentAssembler().setInputCol(text_column).setOutputCol("document")
  tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
  stopwords_cleaner = StopWordsCleaner().setInputCols(["token"]).setOutputCol("clean_tokens").setCaseSensitive(False)
  lemmatizer = LemmatizerModel.pretrained().setInputCols(["clean_tokens"]).setOutputCol("lemma")
  finisher = Finisher().setInputCols(["lemma"]).setCleanAnnotations(True)

  pipeline = Pipeline(stages = [document_assembler, tokenizer, stopwords_cleaner, lemmatizer, finisher])
  result = pipeline.fit(dataset).transform(dataset).select("book_title", "finished_lemma")

  return result


def average_vectors(vectors):

    arrays = np.array([v.toArray() for v in vectors])
    avg_arr = np.mean(arrays, axis=0)

    return DenseVector(avg_arr)


def group_vectors(embedded_data):

  avg_udf = udf(average_vectors, VectorUDT())
  book_embeddings = embedded_data.groupBy("book_title").agg(collect_list("embedding_vector_ml").alias("embeddings_list")).withColumn("book_embedding", avg_udf("embeddings_list")).select("book_title", "book_embedding")

  normalizer = Normalizer(inputCol = "book_embedding", outputCol = "norm_embedding", p = 2.0)
  book_embeddings = normalizer.transform(book_embeddings).select("book_title", "norm_embedding")

  return book_embeddings


def group_text(dataset):

  if dataset == df_reviews:
    text_column = "review_text"
  else:
    text_column = "description"

  return (dataset.groupBy("book_title").agg(concat_ws(" ", collect_list(text_column)).alias(text_column)))


def compute_cosine_similarity(grouped_reviews):

  lsh = BucketedRandomProjectionLSH(inputCol = "norm_embedding", outputCol = "hashes", bucketLength = 2.0, numHashTables = 3)
  cosine_similarity_model = lsh.fit(grouped_reviews)
  similar_books_cosine = cosine_similarity_model.approxSimilarityJoin(grouped_reviews, grouped_reviews, threshold = 0.4, distCol = "distance").filter("datasetA.book_title < datasetB.book_title").withColumn("cosine_similarity", (1 - col("distance"))).selectExpr("datasetA.book_title as book1", "datasetB.book_title as book2", "cosine_similarity").orderBy("cosine_similarity", ascending = False)

  return similar_books_cosine


def compute_jaccard_similarity(grouped_reviews):

  hashingTF = HashingTF(inputCol = "finished_lemma", outputCol = "features", numFeatures = 10000)
  featurized = hashingTF.transform(grouped_reviews)
  mh = MinHashLSH(inputCol = "features", outputCol = "hashes", numHashTables = 3)

  jaccard_similarity_model = mh.fit(featurized)
  similar_books_jaccard = jaccard_similarity_model.approxSimilarityJoin(featurized, featurized, threshold = 0.9, distCol="jaccard_distance").filter("datasetA.book_title < datasetB.book_title").withColumn("jaccard_similarity", 1 - col("jaccard_distance")).selectExpr("datasetA.book_title as book1", "datasetB.book_title as book2", "jaccard_similarity").orderBy("jaccard_similarity", ascending = False)

  return similar_books_jaccard