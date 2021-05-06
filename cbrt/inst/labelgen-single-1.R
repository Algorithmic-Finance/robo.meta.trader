### Library

library(cbrt) # https://github.com/Algorithmic-Finance/robo.meta.trader/tree/main/cbrt
library(quantmod)

# Data

getSymbols("AAPL")
aapl.ohlc <- AAPL[,1:4]
names(aapl.ohlc) <- c("o", "h", "l", "c")

# Parameter

data.ohlc <- aapl.ohlc["2020"]
img.prefix <- "aapl"
block.size <- 20
path <- tempdir()
img.type <- "line" # dotline, candle
label.type <- "binary" # eps.gap

# Image and Label Generation

if(!dir.exists(paste0(path, "/img"))) { dir.create(paste0(path, "/img")) }

range <- (block.size+1):(nrow(data.ohlc)-1)

label.vector.size <- nrow(data.ohlc) - block.size - 1
labels <- rep(NA, nrow(data.ohlc) - block.size - 1)

for(pos in range) {
  
  # Image
  image_ohlc_single(data.ohlc[(pos-block.size):(pos-1)],
                    type = img.type,
                    path = paste0(path, "/img"),
                    file = paste0(tolower(paste0(img.prefix, collapse = "-")), pos-block.size) )
  
  # Label
  label <- NA
  if(label.type == "binary") {
    label <- 0
    if (as.numeric(data.ohlc$c[pos]) > as.numeric(data.ohlc$c[pos-1])) { label <- 1 }
  }
  labels[pos-block.size] <- label

}

save(labels, file = paste0(path, "/", img.prefix, "-labels.rda"))
write.table(labels, file = paste0(path, "/", img.prefix, "-labels.csv"), 
            col.names = FALSE, row.names = FALSE)
