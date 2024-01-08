library(rtweet)
library(ggplot2)
library(dplyr)
library(tidytext)
library(tidyverse)
library(wordcloud)
library(textdata)
library(reshape2)
library(tm)
library(sentimentr)

# Connection set up
appname <- "insert_App_name_here"

# API key
key <- "insert_API_key_here"

# API secret
secret <- "insert_API_secret_here"

# Create token
twitter_token <- create_token(
  app = appname,
  consumer_key = key,
  consumer_secret = secret,
  access_token = 'insert_token_here',
  access_secret = 'insert_secret_here')

# Create empty list to collect data
store_maxnode <- list()

# Create list of tweets keywords
terms <- list('airbnb'
              ,'@airbnb'
              ,'#airbnb'
              ,'Airbnb'
              ,'@Airbnb'
              ,'#Airbnb'
              ,'Air BnB'
              ,'Air B N B'
              ,'@AirB&B'
              ,'#AirBnb'
              ,'Air-bnb'
              ,'Air_bnb'
)

# Store tweets that meet keywords
for (term in terms) {
  rf_maxnode <- search_tweets(q = term,
                              nclude_rts = FALSE,
                              n=1000,
                              lang = "en",
                              geocode = "40.712776, -74.005974, 10km")
  current_iteration <- term
  store_maxnode[[current_iteration]] <- rf_maxnode
}

store_maxnode
data <- do.call("rbind", store_maxnode)

# Remove duplicates and standardize texts
data <- data %>% distinct()
data['text'] <- data %>% select(text) %>% lapply(replace_emoji)
data['text'] <- data %>% select(text) %>% lapply(replace_emoticon)
data['text'] <- data %>% select(text) %>% lapply(tolower)
data['text'] <- data %>% select(text) %>% lapply(str_trim)

# Get % of Positive, Neutral and Negative tweets
sentences <- get_sentences(data)
sentiments <- sentiment(sentences)
sentiments <- sentiments %>% mutate(label=ifelse(sentiments$sentiment>0, 'Positive',
                                   ifelse(sentiments$sentiment<0, 'Negative',
                                          'Neutral'))) %>% distinct()

sentiments %>% ggplot(aes(label))+
  geom_bar()+
  xlab('Sentiment')+
  ylab('# Tweets')+
  theme_bw()+
  ggtitle('Tweets vs Sentiment')

# Get percentages
prop.table(table(sentiments$label))

sentiments %>% filter(label!='Neutral') %>% ggplot(aes(label, sentiment))+
  geom_boxplot()+
  theme_bw()+
  xlab('Tweet type')+
  ylab('Sentiment score')+
  ggtitle('Scores vs Tweet type')


# Get emotion for each sentence
emotions <- emotion(sentences)
emo <- c('anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust')
emodf <- emotions %>% filter(emotion_count!=0) %>% 
  filter(emotion_type %in% emo) %>% 
  select(emotion_type, emotion_count) %>% 
  group_by(emotion_type) %>% 
  summarise(tot=n())

emodf$emotion_type <- factor(emodf$emotion_type, levels = emodf$emotion_type[order(-emodf$tot)])

emodf %>% ggplot(aes(emotion_type, tot))+
  geom_col()+
  xlab('Emotion')+
  ylab('# Sentences')+
  theme_bw()+
  ggtitle('Sentences vs Emotions')+
  scale_x_discrete(labels=c('Trust', 'Anticipation', 'Anger', 'Fear', 'Joy', 'Sadness', 'Disgust', 'Surprise'))


# Get only tweets then tokens
dat <- as_tibble(data %>% select('text'))
tokens <- dat %>% unnest_tokens(word, text)

# See 10 most frequently used words
tokens %>% count(word, sort = T)

# Eliminate numbers
tokens <- tokens %>% filter(!grepl('[0-9]', word))

# Eliminate stop words
data("stop_words")
corpus <- tokens %>% anti_join(stop_words)

# See 10 most frequently used words
corpus %>% count(word, sort = T)

# Create additional list of words to eliminate:
my_stopwords <- stop_words %>% 
  select(-lexicon) %>%
  bind_rows(data.frame(word = c('t.co', 'https')))

# Exclude from corpus
corpus <- corpus %>% anti_join(my_stopwords)
corpus %>% count(word, sort = T)


######## BING #########
# Get "Bing" lexicon and extract positive and negative words in a ranking:
bing <- get_sentiments("bing")

# Word cloud Negative vs Positive:
corpus %>%
  inner_join(bing) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("#F8766D", "#00BFC4"),
                   max.words = 100)

bing_word_counts <- corpus %>%
  inner_join(bing) %>%
  count(word, sentiment, sort = TRUE) %>% print(n=10)

#Get "Bing" positive words:
positive <- get_sentiments("bing") %>%
  filter(sentiment == "positive")

#Find positive words in corpus:
positive_words <- corpus %>%
  semi_join(positive) %>%
  count(word, sort = TRUE)

#Get "Bing" negative words:
negative <- get_sentiments("bing") %>%
  filter(sentiment == "negative")

#Find negative words in corpus:
negative_words <- corpus %>%
  semi_join(negative) %>%
  count(word, sort = TRUE)

#Plot Positive vs Negative words occurring > 30 checking contribution to sentiment:
bing_word_counts %>%
  filter(n > 30) %>%
  rename('Sentiment'='sentiment') %>%
  mutate(n = ifelse(Sentiment == "negative", -n, n)) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = Sentiment)) +
  geom_col() +
  coord_flip() +
  theme_bw()+
  ggtitle('Word vs Contribution to Sentiment')+
  ylab('Score')+
  xlab('Word')+
  scale_fill_discrete(labels=c('Negative', 'Positive'))


######## LOUGHRAN #########
loughran <- get_sentiments("loughran")
loughran_word_counts <- corpus %>%
  inner_join(loughran) %>%
  count(word, sentiment, sort = TRUE)

loughran_word_counts %>%
  filter(n > 30) %>%
  rename('Sentiment'='sentiment') %>%
  mutate(n = ifelse(Sentiment == "negative", -n, n)) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = Sentiment)) +
  geom_col() +
  coord_flip()+
  theme_bw()+
  ggtitle('Word vs Contribution to Sentiment')+
  ylab('Score')+
  xlab('Word')+
  scale_fill_discrete(labels=c('Litigious', 'Negative', 'Positive'))


######## AFINN #########
afinn <- get_sentiments("afinn")
afinn_word_counts <- corpus %>%
  inner_join(afinn) %>%
  count(word, value, sort = TRUE)

afinn_word_counts %>%
  filter(n > 30) %>%
  rename('Value'='value') %>%
  mutate(n = ifelse(Value == "negative", -n, n)) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = Value)) +
  geom_col() +
  coord_flip() +
  theme_bw()+
  ggtitle('Word vs Contribution to Sentiment')+
  ylab('Score')+
  xlab('Word')+
  scale_fill_continuous(labels=c('-3', '-2', '-1', '0', '+1', '+2', '+3'))

