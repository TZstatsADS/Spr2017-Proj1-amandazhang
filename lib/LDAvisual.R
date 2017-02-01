library("rvest")
library("tibble")
library("qdap")
library("sentimentr")
library("gplots")
library("dplyr")
library("tm")
library("syuzhet")
library("factoextra")
library("beeswarm")
library("scales")
library("RColorBrewer")
library("RANN")
library("tm")
library("topicmodels")
source("../lib/plotstacked.R")
source("../lib/speechFuncs.R")


#Inauguaral speeches
main.page <- read_html(x = "http://www.presidency.ucsb.edu/inaugurals.php")
# Get link URLs
# f.speechlinks is a function for extracting links from the list of speeches. 
inaug=f.speechlinks(main.page)
#head(inaug)
as.Date(inaug[,1], format="%B %e, %Y")
inaug=inaug[-nrow(inaug),] # remove the last line, irrelevant due to error.
# The inauglist contains information of 45 presidents' inauguration address length,
# term and date
inaug.list=read.csv("../data/inauglist.csv", stringsAsFactors = FALSE)
inaug.list=cbind(inaug.list, inaug)

inaug.list$fulltext=NA
for(i in seq(nrow(inaug.list))) {
  text <- read_html(inaug.list$urls[i]) %>% # load the page
    html_nodes(".displaytext") %>% # isloate the text
    html_text() # get the text
  inaug.list$fulltext[i]=text
  # Create the file name
  filename <- paste0("../data/fulltext/", 
                     inaug.list$type[i],
                     inaug.list$File[i], "-", 
                     inaug.list$Term[i], ".txt")
  sink(file = filename) %>% # open file to write 
    cat(text)  # write the file
  sink() # close the file
}

inauguration <- character()
for (i in 1:nrow(inaug.list)) {
  inauguration[i] <- inaug.list$fulltext[i]
}


library(tm)
stop_words <- stopwords("SMART")

# pre-processing:
inauguration <- gsub("'", "", inauguration)  # remove apostrophes
inauguration <- gsub("[[:punct:]]", " ", inauguration)  # replace punctuation with space
inauguration <- gsub("[[:cntrl:]]", " ", inauguration)  # replace control characters with space
inauguration <- gsub("^[[:space:]]+", "", inauguration) # remove whitespace at beginning of documents
inauguration <- gsub("[[:space:]]+$", "", inauguration) # remove whitespace at end of documents
inauguration <- tolower(inauguration)  # force to lowercase

# tokenize on space and output as a list:
doc.list <- strsplit(inauguration, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)
#Using the R package 'lda' for model fitting

#The object documents is a length-2000 list where each element represents one document, according to the specifications of the lda package. After creating this list, we compute a few statistics about the corpus:

# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (2,000)
W <- length(vocab)  # number of terms in the vocab (14,568)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (546,827)
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus [8939, 5544, 2411, 2410, 2143, ...]



# MCMC and model tuning parameters:
K <- 15
G <- 5000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  # about 24 minutes on laptop
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))
InaugReviews <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)
library(LDAvis)
json <- createJSON(phi = InaugReviews$phi, 
                   theta = InaugReviews$theta, 
                   doc.length = InaugReviews$doc.length, 
                   vocab = InaugReviews$vocab, 
                   term.frequency = InaugReviews$term.frequency)
serVis(json, out.dir = 'vis', open.browser = FALSE)




library(shiny)

ui <- shinyUI(
  fluidPage(
    sliderInput("nTerms", "Number of terms to display", min = 20, max = 40, value = 30),
    visOutput('myChart')
  )
)

server <- shinyServer(function(input, output, session) {
  output$myChart <- renderVis({
    if(!is.null(input$nTerms)){
      with(InaugReviews, 
           createJSON(phi, theta, doc.length, vocab, term.frequency, 
                      R = input$nTerms))
    } 
  })
})

shinyApp(ui = ui, server = server)





