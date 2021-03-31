### Library

library(xts)

### Data

load(url("https://cdn.quantlab.cloud/finance/data/djia30.rda"))

### Image and Label Generation

# Test: AAPL & MSFT
pair.ticker <- c("AAPL", "MSFT")
pair.returns <- list(cumprod(c(1, 1+djia30$AAPL)), cumprod(c(1, 1+djia30$MSFT)))

block.size <- 20

label.vector.size <- nrow(djia30) - block.size - 1
labels <- rep(NA, nrow(djia30) - block.size - 1)

for(current.start in c(1:label.vector.size)) {
  current.block <- current.start:(current.start+block.size)
  image.gen(pair.returns[[1]][current.block], pair.returns[[2]][current.block],
            file = paste0(tolower(paste0(pair.ticker, collapse = "-")), current.start) )
  labels[current.start] <- ifelse((pair.returns[[1]][current.start + block.size + 1] / pair.returns[[1]][current.start + block.size]) > 1, 1, 0)
}

save(labels, file = paste0(tolower(paste0(pair.ticker, collapse = "-")), "-labels.rda"))
