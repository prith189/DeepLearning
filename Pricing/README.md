# Exploring Pricing Optimization with Machine Learning

In this notebook, we will implement a model to understand the mechanics of pricing optimization using machine learning. Google published a [blog post](https://cloud.google.com/blog/products/ai-machine-learning/price-optimization-using-vertex-ai-forecast) where they used a synthetic dataset with prices and sales for various items. However, their demo focused on the AutoML platform, and they did not share any details on the model itself, only the results from the model were shared.

Our goal is to reproduce the results from Google using a **Long Short-Term Memory (LSTM) based recurrent neural network**.


## High-Level Overview

1. **Download and ingest data**
2. **Frame the problem**:
    - Given price history and sales history for the last 28 days, can we estimate the sales for the next 14 days, for various price ranges (i.e., given the history, if I set the price as X, what is the expected demand)?
3. **Build an encoder-decoder LSTM based architecture** where the decoder takes in a price and outputs the demand (i.e., sales for the next 14 days).
4. **Compute sales for the next 14 days** for a product, given the price that we want to set (i.e., estimate the demand for our price).
5. **Estimate the optimal price** that maximizes profit.
