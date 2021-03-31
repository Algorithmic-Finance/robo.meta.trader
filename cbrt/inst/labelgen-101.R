### Library

library(xts)

### Data

load(url("https://cdn.quantlab.cloud/finance/data/djia30.rda"))
time.series <- djia30
ticker.list <- names(time.series)

### Parameters

pair.ticker <- c("AAPL", "MSFT")
block.size <- 20

### Image and Label Generation

pair.returns <- lapply(1:length(pair.ticker), 
                       function(x) { cumprod(c(1, 1+time.series[, which(ticker.list == pair.ticker[x])])) })

label.vector.size <- nrow(djia30) - block.size - 1
labels <- rep(NA, nrow(djia30) - block.size - 1)

for(current.start in c(1:label.vector.size)) {
  current.block <- current.start:(current.start+block.size)
  image.gen(pair.returns[[1]][current.block], pair.returns[[2]][current.block],
            file = paste0(tolower(paste0(pair.ticker, collapse = "-")), current.start) )
  labels[current.start] <- ifelse((pair.returns[[1]][current.start + block.size + 1] / pair.returns[[1]][current.start + block.size]) > 1, 1, 0)
}

save(labels, file = paste0(tolower(paste0(pair.ticker, collapse = "-")), "-labels.rda"))
