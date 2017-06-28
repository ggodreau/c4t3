# imports
#library(tidyr)
library(dplyr)
library(caret)
library(corrplot) # untested

data <- read.table(
  './trainingData.csv',
  sep=",",
  header=TRUE,
  as.is=TRUE,
  nrows=100
)

# select sum of row counts, groupby column, where 
# value = NULL (na) and count > 0. Then drop any na's
naCount <- data.frame(
  sapply(
    data,
    function(y) sum(
      length(
        which(
          is.na(y)
        )
      )
    )
  )
)

subset(naCount, naCount[1] > 0)
data <- na.omit(data)
naCount <- NULL

# generate target Y label from distinct factor interactions of
# FLOOR, BUILDINGID, SPACEID, and RELATIVEPOSITION (FBSR)
# FLOOR, BUILDING, SPACEID (FBS)
# FLOOR, BUILDING (FB)
# BUILDING (B)
data <- cbind(data,interaction(data[,523:526], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "FBSR"

data <- cbind(data,interaction(data[,523:525], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "FBS"

data <- cbind(data,interaction(data[,523:524], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "FB"

data <- cbind(data,interaction(data[,523], sep="-", drop=TRUE))
colnames(data)[ncol(data)] <- "B"

# drop source FLOOR, BUILDINGID, SPACEID, and RELATIVEPOS cols
data <- data[,-c(523:526)]

# linearize logarithmic RSSI values and set 100 RSSI values
# (no signal) to zero. This transforms the feature space 
# from a log feature space to a linearly separable one
# WARNING: THIS IS A HORRIBLE HACK!! Please send me your PR!!

# stack all WAP factors in one column, this allows us to
# transform values without looping over column indexes,
# something not really possible in R
wapStack <- stack(
  dplyr::select(
    data, dplyr::starts_with("WAP")
    )
  )

# move the row.names df index into the dataframe itself
# this is needed because the dplyr transformations will 
# lose the original df index which is needed to reconstruct
# the wapStack when we rejoin the dplyr data 'tibbles'
wapStack <- cbind(wapStack, row.names(wapStack))
colnames(wapStack)[ncol(wapStack)] <- "id"
wapStack <- wapStack[,c(ncol(wapStack), 1:(ncol(wapStack)-1))]

# change values of 100 RSSI to zero (no signal)
wapNulls <- dplyr::filter(wapStack, values==100) %>%
  dplyr::transmute(id, values = 0, ind)

# convert remaining RSSI values to linear scale between 0-1
wapVals <- dplyr::filter(wapStack, values!=100) %>%
  dplyr::transmute(id, values = values + 5, ind)

# join the 'no signal' and 'signal' frames by rows
wapJoined <- rbind(wapNulls, wapVals)

# overwrite the df idx with the 'id' field
row.names(wapJoined) <- wapJoined[,1]

# sort the joined df asc by the id field
# note that id must be converted from factor to num
wapJoined[,1] <- as.numeric(wapJoined[,1])
wapJoined <- dplyr::arrange(wapJoined, id) %>%
  dplyr::select(-id) # drop the id column

# finally, unstack the joined set to get our factor 
# matrix back with no '100' values and linear RSSIs
wapClean <- unstack(wapJoined)


# merge the wapClean df back into the original data df
data <- cbind(
  wapClean, 
  dplyr::select(data, -(starts_with("WAP")))
)

# clean up intermediary tables
wapStack <- NULL
wapNulls <- NULL
wapVals <- NULL
wapJoined <- NULL
wapClean <- NULL


# reduce feature space - find near zero variance columns in 
# the WAP feature space (1:521) and drop them. Tuned 
# freq/uniqueCut args to compensate for the sparse nature 
# of sampling (somewhat)
data <- data[,-c(
  caret::nearZeroVar(
    data[,1:520],
    # saveMetrics=TRUE, # valuable info for tuning
    freqCut = 50,
    uniqueCut = 1,
    names = FALSE
  )
)]

# visualize covariance matrix
# note that you can't draw cor to Y because Y is a factor
corrplot(cor(X)) # this should work, but is untested

# create X and Y (factors and target) matricies
X <- dplyr::select(data, (starts_with("WAP")))
# scaling needed? not sure
# X <- scale(X, center=TRUE)
# FLOOR.BLDG.SPACEID, Y target
Y <- dplyr::select(data, "FBS")

# note that there are only 76 labels for 100 instances (rows)
# this is 1.3 samples per class (not that great)

# create trainControl object
set.seed(1337)
ctrl <- trainControl(
  method="cv",
  repeats=5
)

# train dat model (need e1071 installed)
modKnn <- train(
  X,
  Y,
  preProcess = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl
)

# helpful commands
data[1:5,((ncol(data)-6):ncol(data))] #select last 6 cols

library(dplyr)
dplyr::select(data, WAP008:WAP012) %>%
  dplyr::filter(WAP008==100)

# playground
foo <- stack(data[,1:2])
foo <- cbind(foo, row.names(foo))
colnames(foo)[ncol(foo)] <- "id"
foo <- foo[,c(ncol(foo), 1:(ncol(foo)-1))]

dplyr::transmute(foo, values = values + 5)
dplyr::select(data, starts_with("WAP"))

fooNulls <- dplyr::filter(foo, values==100) %>%
  dplyr::transmute(id, values = 0, ind)

fooVals <- dplyr::filter(foo, values!=100) %>%
  dplyr::transmute(id, values = values + 5, ind)

fooJoined <- rbind(fooNulls, fooVals)
row.names(fooJoined) <- fooJoined[,1]

fooJoined[,1] <- as.numeric(fooJoined[,1])
fooJoined <- dplyr::arrange(fooJoined, id) %>%
  dplyr::select(-id)

unstack(fooJoined)
