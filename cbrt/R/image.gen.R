#' @title Create an image out of two pre-processed (numeric) time-series
#' 
#' @description
#' \code{image.gen} creates an image for chart-based Deep Learning
#'
#' @param series1 numeric vector of values of time-series 1
#' @param series2 numeric vector of values of time-series 2 
#' @param path path where to save the image to
#' @param filename file name of the image
#' @param width the width of the image in pixels
#' @param height the height of the image in pixels
#' @param colors a vector of colors to be used in sequence (currently 2)
#' 
#' @return currently TRUE in every case
#' 
#' @author Ronald Hochreiter, \email{ronald@@algorithmic.finance}
#'
#' @export
image.gen <- function(series1, series2, 
                      path = tempdir(), filename = "img", 
                      width = 200, height = 200,
                      colors = c("red", "blue"), ...) {
  
  series1 <- series1 / series1[1]
  series2 <- series2 / series2[1]
  
  png(paste0(path, "/", filename, ".png"), width = width, height = height, units = "px")
  op <- par(mar = rep(0, 4))
  plot(series1, lwd = 2, type = "l", col = colors[1], 
       ylab = "", xlab = "", xaxt = "n", yaxt = "n", bty = "n",
       ylim = c(min(series1, series2), max(series1, series2)))
  lines(series2, lwd = 2, col = colors[2])
  par(op)
  dev.off()
  
  return(TRUE)
}
