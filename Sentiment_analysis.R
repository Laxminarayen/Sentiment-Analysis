setwd("C:\\Users\\AADHI\\Documents\\Laxmi")
#install.packages("keras")
library(keras)
#install_keras()
library(Rcpp)
install.packages("XML")
library(XML)
install.packages("htmltidy")
library(htmltidy)
library(tidyverse)
library(stringi)
install.packages("xgboost")
library(xgboost)
library(magrittr)
library(rvest)
library(httr)
# Define maximum number of input features
max_features <- 20000
# Cut texts after this number of words
# (among top max_features most common words)
maxlen <- 100
batch_size <- 32

train <- read_csv("train_2kmZucJ.csv")
test <- read_csv("test_oJQbWVk.csv")

y <- train$label
tr_te <- train %>% select(-label) %>% bind_rows(test)
conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")
install.packages("text2vec")
library(text2vec)
library(stringi)
library(stringr)
install.packages("purrrlyr")
library(purrrlyr)
tri <- 1:nrow(train)
head(tweet,20)
rep <- function(x) gsub("\\$", "Dollar", x)
tweet <-  as.data.frame(apply(tr_te[,2], 2,rep ), stringsAsFactors = FALSE)
rep <- function(x) gsub("Dollar\\&", "DollarAnd", x)
tweet <-  as.data.frame(apply(tweet, 2,rep ), stringsAsFactors = FALSE)
rep <- function(x) gsub("DollarAnd\\@", "DollarAt", x)
tweet <-  as.data.frame(apply(tweet, 2,rep ), stringsAsFactors = FALSE)
rep <- function(x) gsub("DollarAndAt\\*", "DollarAndAtStar", x)
tweet <-  as.data.frame(apply(tweet, 2,rep ), stringsAsFactors = FALSE)
rep <- function(x) gsub("DollarAndAtStar\\#", "Profanity", x)
tweet <-  as.data.frame(apply(tweet, 2,rep ), stringsAsFactors = FALSE)
tr_te$tweet <- tweet$tweet
tr_te <- tr_te %>% dmap_at('tweet', conv_fun)
cat('x_train shape:', dim(tr_te), '\n')
tr <- tr_te[tri,]
te <- tr_te[-tri,]
prep_fun <- tolower
tok_fun <- text_tokenizer(num_words = 40,
                          filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n", lower = TRUE,
                          split = " ", char_level = TRUE, oov_token = NULL)
tok_fun <- fit_text_tokenizer(tok_fun,tr_te$tweet)
it_complete <- itoken(tr_te$tweet, 
                      preprocessor = prep_fun, 
                      tokenizer = tok_fun,
                      ids = tr_te$id,
                      progressbar = TRUE)
it_tr <- itoken(tr$tweet, 
                preprocessor = prep_fun, 
                tokenizer = tok_fun,
                ids = tr$id,
                progressbar = TRUE)
it_test <- itoken(te$tweet, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun,
                  ids = te$id,
                  progressbar = TRUE)
sequencetrain <- texts_to_sequences(tok_fun, tr$tweet)
sequencecomplete <- texts_to_sequences(tok_fun, tr_te$tweet)
sequencetest <- texts_to_sequences(tok_fun, te$tweet)
train_data <- pad_sequences(
  sequencetrain
)
test_data <- pad_sequences(
  sequencetest
)
completeddata <- pad_sequences(
  sequencecomplete
)
x_train <- completeddata[tri,]
x_test <- completeddata[-tri,]
cat('x_train shape:', dim(completeddata), '\n')
cat('x_train shape:', dim(train_data), '\n')
cat('x_test shape:', dim(test_data), '\n')
model <- keras_model_sequential()
model %>%
  # Creates dense embedding layer; outputs 3D tensor
  # with shape (batch_size, sequence_length, output_dim)
  layer_embedding(input_dim = ncol(x_train), 
                  output_dim = 128, 
                  input_length = 377) %>%
  layer_dropout(rate = 0.5) %>%
  bidirectional(layer_lstm(units = 64)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = 'sigmoid')
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)
cat('Train...\n')
model %>%keras::fit(
  x_train, y,epochs = 40, batch_size = 512,
  callbacks = callback_early_stopping(patience = 10, monitor = 'acc'),
  validation_split = 0.05
)
train$label <- predict(model,x_test)