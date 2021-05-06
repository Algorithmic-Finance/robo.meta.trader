#' @title Create an image out of two pre-processed (numeric) time-series
#' 
#' @description
#' \code{image_ohlc_single} creates an image for chart-based Deep Learning
#'
#' @param ohlc ohlc xts of time series
#' @param type plot type: line, line-macross, dotline, candle, candle-line
#' @param path path where to save the image to
#' @param filename file name of the image
#' @param width the width of the image in pixels
#' @param height the height of the image in pixels
#' @param colors a vector of colors to be used in sequence (currently 4: dark, up, down, light)
#' 
#' @return currently TRUE in every case
#' 
#' @author Ronald Hochreiter, \email{ronald@@algorithmic.finance}
#'
#' @export
image_ohlc_single <- function(ohlc, type = "line",
                              path = tempdir(), filename = "img", 
                              width = 200, height = 200,
                              colors = c("black", "red", "green", "lightgrey")) {

  ret.pos <- which(dailyReturn(ohlc$c) > 0)
  ret.neg <- which(dailyReturn(ohlc$c) < 0)

  png(paste0(path, "/", filename, ".png"), width = width, height = height, units = "px")
  op <- par(mar = rep(0, 4))
  
  # line
  if(type == "line") {
    plot(as.numeric(ohlc$c), lwd = 2, type = "l", col = colors[1], 
         ylab = "", xlab = "", xaxt = "n", yaxt = "n", bty = "n")
  }

  # line-macross
  if(type == "line-macross") {
    plot(as.numeric(ohlc$c), lwd = 2, type = "l", col = colors[1], 
         ylab = "", xlab = "", xaxt = "n", yaxt = "n", bty = "n",
         ylim = c(min(ohlc$c, ohlc$ma.fast, ohlc$ma.slow), max(ohlc$c, ohlc$ma.fast, ohlc$ma.slow)))
    lines(as.numeric(ohlc$ma.fast), lwd = 1, col = colors[2])
    lines(as.numeric(ohlc$ma.slow), lwd = 1, type="l", col = colors[3])
  }
  
  # dotline
  if(type == "dotline") {
    plot(as.numeric(ohlc$c), pch = 16, col = colors[3], 
         ylab = "", xlab = "", xaxt = "n", yaxt = "n", bty = "n")
    lines(as.numeric(ohlc$c), type="l", col = colors[4])
    points(ret.neg, ohlc$c[ret.neg], pch = 16, col = colors[2])
    points(1, ohlc$c[1], pch = 16, col = colors[1])
  }
  
  # candle
  if(type == "candle" | type == "candle-line") {
    plot(as.numeric(ohlc$c), pch = 16, col = colors[4], 
         ylab = "", xlab = "", xaxt = "n", yaxt = "n", bty = "n")
    
    for(i in 1:nrow(ohlc)) { rect(i, ohlc$l[i], i, ohlc$h[i]) }
    for(i in ret.pos) { rect(i, ohlc$l[i], i, ohlc$h[i], col = colors[2], border = colors[2]) }
    for(i in ret.neg) { rect(i, ohlc$l[i], i, ohlc$h[i], col = colors[3], border = colors[3]) }

    points(as.numeric(ohlc$c), pch = 16, col = colors[4])
    
    if(type == "candle-line") {
      lines(as.numeric(ohlc$c), lwd = 1, type="l", col = colors[4])
    }
  }  

  par(op)
  dev.off()
}