# imports
library(dplyr)
library(caret)

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
  dplyr::transmute(id, values = 10^(values/10)/1000*10^12, ind)

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

# create X and Y (factors and target) matricies
X <- dplyr::select(data, starts_with("WAP"))
# scaling needed? need to investigate if train() does
# a good job. If not, use X <- scale(X, center=TRUE)

# create class label Y matrix, FLOOR.BLDG.SPACEID
Y_FBS <- dplyr::select(data, ends_with("FBS"))
Y_FBS <- Y_FBS[,1] # caret accepts only vector, not data.frame

# create class label Y matrix, FLOOR.BLDG (no SPACEID)
Y_FB <- dplyr::select(data, ends_with("FB"))
Y_FB <- Y_FB[,1] # caret accepts only vector, not data.frame

# create trainControl object
set.seed(1337)
ctrl <- trainControl(
  method="cv",
  repeats=1
)

# train knn model
modKnn <- train(
  X,
  Y_FB, #note that FBS or FBSR will probably not work here
  method = "knn",
  preProcess = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl
)

modKnn

## train svm model
modSvm <- train(
  X,
  Y_FB, #note that FBS or FBSR will probably not work here
  method = 'svmLinear3',
  preProcess = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl
)

modSvm
