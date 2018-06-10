##data mining code-model 3
##combination of logistic + xgboost



# Fork of kxx's kernel at
# https://www.kaggle.com/kailex/tidy-xgboost-glmnet-text2vec-lsa?scriptVersionId=2713975
# Just added KFold CV + OOF for stacking


##ojb-


library(tidyverse)
library(magrittr)
library(text2vec)
library(tokenizers)
library(cvTools)
library(xgboost)
library(glmnet)
library(doParallel)
registerDoParallel(4)

train <- read_csv("C:/data mining/Data Mining Project/train.csv/train.csv") 
test <- read_csv("C:/data mining/Data Mining Project/test.csv/test.csv") 
##subm <- read_csv("../input/sample_submission.csv") 

tri <- 1:nrow(train)
targets <- c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")

#---------------------------
cat("Basic preprocessing & stats...\n")
tr_te <- train %>% 
  select(-one_of(targets)) %>% 
  bind_rows(test) %>% 
  mutate(length = str_length(comment_text),
         ncap = str_count(comment_text, "[A-Z]"),
         ncap_len = ncap / length,
         nexcl = str_count(comment_text, fixed("!")),
         nquest = str_count(comment_text, fixed("?")),
         npunct = str_count(comment_text, "[[:punct:]]"),
         nword = str_count(comment_text, "\\w+"),
         nsymb = str_count(comment_text, "&|@|#|\\$|%|\\*|\\^"),
         nsmile = str_count(comment_text, "((?::|;|=)(?:-)?(?:\\)|D|P))")) %>% 
  select(-id) %T>% 
  glimpse()

#---------------------------
cat("Parsing comments...\n")
it <- tr_te %$%
  str_to_lower(comment_text) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  itoken(tokenizer = tokenize_word_stems)



##adding more libraries

library(tm)
library(caret)
library(RColorBrewer)
library(wordcloud)

vectorizer <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("en")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.5, vocab_term_max = 4000) %>%
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <- create_dtm(it, vectorizer) %>%
  fit_transform(m_tfidf)  
  
m_lsa <- LSA$new(n_topics = 25, method = "randomized")
lsa <- fit_transform(tfidf, m_lsa)

#---------------------------
cat("Preparing data for glmnet...\n")
X <- tr_te %>% 
  select(-comment_text) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(tfidf,lsa)


X_test <- X[-tri, ]
X <- X[tri, ]





##rm(tr_te, test, tri, it, vectorizer, m_lsa, lsa); gc()

#---------------------------
cat("Training & predicting...\n")

p <- list(objective = "binary:logistic", 
          booster = "gbtree", 
          eval_metric = "auc", 
          nthread = 4, 
          eta = 0.2, 
          max_depth = 3,
          min_child_weight = 4,
          subsample = 0.7,
          colsample_bytree = 0.7)

k <- 5
oofs_cols <- c("id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
oofs <- subset(train, select = oofs_cols)

for (target in targets) {
  cat("\nFitting", target, "...\n")
  y <- train[[target]]
  dataset <- cbind(X, y)
  folds <- cvFolds(nrow(dataset), K=k)
  
  ts_pred = list()
  ts_fpred = list()

  for (i in 1:k) {
    cat("\nFitting", i, "Fold\n")
    trainx <- dataset[folds$subsets[folds$which != i], ]
    validx <- dataset[folds$subsets[folds$which == i], ]

    m_xgb <- xgboost(trainx[,-ncol(trainx)], trainx[,ncol(trainx)],
                                             params = p, print_every_n = 100, nrounds = 500)
    m_glm <- glmnet(trainx[,-ncol(trainx)], factor(trainx[,ncol(trainx)]),
                                            alpha = 0, family = "binomial", nlambda = 50)
    
    xgb_val_pred <- predict(m_xgb, validx[,-ncol(validx)])
    glm_val_pred <- predict(m_glm, validx[,-ncol(validx)], type = "response", s = 0.001)
    oofs[folds$subsets[folds$which == i], target] <- (xgb_val_pred + glm_val_pred) / 2
    
    ts_y_pred <- (predict(m_xgb, X_test) + predict(m_glm, X_test, type = "response", s = 0.001)) / 2
    if (i > 1) {
      ts_fpred = ts_pred + ts_y_pred
    } else {
      ts_fpred = ts_y_pred
    }
    ts_pred = ts_fpred
  }
  
  subm[[target]] <- ts_pred / k
}


###IGNORE this
#---------------------------
cat("Creating submission file...\n")
write_csv(subm, "subm_xgb_glm.csv")
write_csv(oofs, "oofs_xgb_glm.csv")