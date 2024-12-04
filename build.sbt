ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.15"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.1",
  "org.apache.spark" %% "spark-sql" % "3.5.0",
  "com.johnsnowlabs.nlp" %% "spark-nlp" % "5.5.1",
  "org.tensorflow" % "tensorflow-core-platform" % "0.5.0",
  "org.apache.spark" %% "spark-mllib" % "3.5.1"
)

lazy val root = (project in file("."))
  .settings(
    name := "spark-NLP-Exercise"
  )
