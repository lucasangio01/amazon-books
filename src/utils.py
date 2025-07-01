import numpy as np

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, when, from_unixtime, length, year, col, array_size, udf, avg, collect_list, row_number
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, IntegerType, StringType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, BucketedRandomProjectionLSH, Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector

from sparknlp.pretrained import PretrainedPipeline
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, StopWordsCleaner, LemmatizerModel, SentenceDetectorDLModel, NorvigSweetingModel, BertSentenceEmbeddings


# some books have duplicates, with some differences in the title. We only keep one of each
duplicated_titles = ["Pride & Prejudice (Penguin Classics)", "Pride & Prejudice (New Windmill)",
                     "Hannibal (Hannibal Lecter)", "Jane Eyre (Penguin Classics)", "Jane Eyre (Large Print)",
                     "Jane Eyre (Signet classics)", "Jane Eyre (Simple English)", "Jane Eyre (New Windmill)",
                     "Jane Eyre (Everyman's Classics)", "Jane Eyre: Complete and Unabridged (Puffin Classics)",
                     "Of Mice and Men (Penguin Audiobooks)",
                     "Me Talk Pretty One Day (Turtleback School & Library Binding Edition)", "Me Talk Pretty One Day C",
                     "Wuthering Heights (Riverside editions)",
                     "Wuthering Heights (Penguin Audiobooks)", "Wuthering Heights (New Windmill)",
                     "A Tale of Two Cities, Literary Touchstone Edition",
                     "A Christmas Carol, in Prose: Being a Ghost Story of Christmas (Collected Works of Charles Dickens)",
                     "Christmas Carol (Ladybird Classics)", "Frankenstein (Running Press classics)",
                     "Signature Classics - Great Expectations (Signature Classics Series)",
                     "The Scarlet Letter (Lake Illustrated Classics, Collection 2)",
                     "The Adventures of Huckleberry Finn (Courage Literary Classics)",
                     "The Picture of Dorian Gray (The Classic Collection)", "Picture of Dorian Gray",
                     "Little Women (Courage giant classics)", "Down Under; Abridged",
                     "Heart of Darkness (Everyman Classics)",
                     "The Secret Garden (Worlds Classics)", "everything on this page is for Treasure Island",
                     "In the Heart of the Sea", "the Picture of Dorian Gray", "Wuthering Heights.",
                     "Daughter of Fortune CD",
                     "Adventures of Huckleberry Finn (Simple English)",
                     "Awakening: Kate Chopin Pb (Case Studies in Contemporary)",
                     "Tess of the D'Urbervilles (New Wessex editions)",
                     "The Awakening: Complete, Authoritative Text With Biographical and Historical Contexts, Critical History, and Essays from Five Contemporary Critical Perspectives (Case Studies in Contemporary Criticism)",
                     "Tess of the d'Urbervilles (Cambridge Literature)",
                     "Tess of Th D'Urbervilles (Pbk)(Oop) (Bloom's Notes)", "Tess of the Durbervilles",
                     "Tess of the D'urbervilles (Summer Classics)",
                     "The Heart Is a Lonely Hunter",
                     "Stone Of Tears (Turtleback School & Library Binding Edition) (Sword of Truth)", "Lucky Man",
                     "The Call of the Wild (Dover Large Print Classics)",
                     "Sense And Sensibility (CH) (Jane Austen Collection)",
                     "Sense and Sensibility (Wordsworth Hardback Library)", "Sense and sensibility",
                     "Emma (Signet classics)", "Emma (Riverside Editions)",
                     "Emma (CH) (Jane Austen Collection)", "Emma (Progress English)", "Emma (The World's Classics)",
                     "Emma (Radio Collection)", "Emma (Summer Classics)", "Persuasion (World's Classics)",
                     "Bet Me (Brilliance Audio on Compact Disc)", "Sense & Sensibility Cds (Penguin Classics)",
                     "Tess of the D'Urbervilles", "Wuthering Heights (Signet classics)", "Alice in Wonderland",
                     "Alice in Wonderland (Tell tales)",
                     "The Picture of Dorian Gray (Classic Collection (Brilliance Audio))",
                     "Great Expectations (Signet classics)", "Heart of Darkness and the Secret Sharer",
                     "Alice's Adventures in Wonderland and Through the Looking Glass (Classic Collection (Brilliance Audio))",
                     "Treasure Island (Classic Illustrated)",
                     "Under and Alone: The True Story of the Undercover Agent Who Infiltrated America's Most Violent Outlaw Motorcycle Gang",
                     "This Present Darkness (Turtleback School & Library Binding Edition)",
                     "Angela's Ashes (Turtleback School & Library Binding Edition)",
                     "The Awakening: Complete, Authoritative Text With Biographical & Historical Contexts, Critical History, & Essays from Five Contemporary Critica. Perspectives (Case Studies in Contemporary Criticism)",
                     "Casting the First Stone", "The Hound of the Baskervilles (Signet Classics)",
                     "Hound of the Baskervilles (Lrs Large Print Heritage Series)"]


def load_data(spark, chosen_data):

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


def embedding_pipeline(dataset):
    if dataset == df_reviews:
        text_column = "review_text"
    else:
        text_column = "description"

    document_assembler = DocumentAssembler().setInputCol(text_column).setOutputCol("document")
    sentence_embeddings = BertSentenceEmbeddings().pretrained("sent_small_bert_L2_128", "en").setInputCols(
        ["document"]).setOutputCol("embeddings")
    pipeline = Pipeline(stages=[document_assembler, sentence_embeddings])

    result = pipeline.fit(dataset).transform(dataset)
    extract_embedding = udf(lambda arr: arr[0].embeddings, ArrayType(FloatType()))
    result = result.withColumn("embedding_vector", extract_embedding("embeddings")).select("book_title",
                                                                                           "embedding_vector")

    to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())
    result = result.withColumn("embedding_vector_ml", to_vector_udf("embedding_vector")).select("book_title",
                                                                                                "embedding_vector_ml")

    return result


def average_vectors(vectors):
    arrays = np.array([v.toArray() for v in vectors])
    avg_arr = np.mean(arrays, axis=0)

    return DenseVector(avg_arr)


def group_vectors(embedded_data):
    avg_udf = udf(average_vectors, VectorUDT())
    book_embeddings = embedded_data.groupBy("book_title").agg(
        collect_list("embedding_vector_ml").alias("embeddings_list")).withColumn("book_embedding",
                                                                                 avg_udf("embeddings_list")).select(
        "book_title", "book_embedding")

    normalizer = Normalizer(inputCol="book_embedding", outputCol="norm_embedding", p=2.0)
    book_embeddings = normalizer.transform(book_embeddings).select("book_title", "norm_embedding")

    return book_embeddings


def compute_similarity(grouped_reviews):
    lsh = BucketedRandomProjectionLSH(inputCol="norm_embedding", outputCol="hashes", bucketLength=2.0, numHashTables=3)
    similarity_model = lsh.fit(grouped_reviews)
    similar_books = similarity_model.approxSimilarityJoin(grouped_reviews, grouped_reviews, threshold=0.3,
                                                          distCol="distance").filter(
        "datasetA.book_title < datasetB.book_title").withColumn("cosine_similarity", (1 - col("distance"))).selectExpr(
        "datasetA.book_title as book1", "datasetB.book_title as book2", "cosine_similarity").orderBy(
        "cosine_similarity", ascending=False)

    return similar_books