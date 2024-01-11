# **Sentiment Analysis of Amazon Reviews for Cell Phones and Accessories**

## **Overview**

This repository explores various machine learning and deep learning techniques for sentiment analysis of Amazon reviews in the cell phones and accessories category. It focuses on predicting review ratings through multi-class classification, comparing the performance of different approaches.

## **Key Features**

- **Models Explored:**
    - Traditional Machine Learning with TF-IDF:
        - Logistic Regression
        - SVM with Linear Kernel
    - Word Embeddings:
        - Gensim Word2Vec with various hyperparameters
        - Pre-trained Google Word2Vec
        - Pre-trained GloVe (50D, 100D, 200D)
    - Neural Networks:
        - RNN, LSTM, BiLSTM with various architectures
    - BERT (pre-trained language model)
- **Dataset:** Amazon reviews for cell phones and accessories (194,439 reviews)
- **Evaluation Metric:** Accuracy

## **Key Findings**

- **Best Performing Model:** Gensim Word2Vec (size=200, window=7, min_count=2,5) with Logistic Regression (Accuracy: 0.6053)
- **Impact of Data Size:** Increasing data from 5% to 25% improved model performance
- **Model Complexity vs. Performance:** Simpler models (logistic regression, SVM) outperformed complex models (BERT) in this case, suggesting potential model overkill or dataset simplicity
- **Importance of Pre-trained Embeddings:** Pre-trained embeddings (Word2Vec, GloVe) often yielded good results
- **Computational Resources:** Limited resources for BERT training (2.5% of data, 5 epochs) potentially impacted results

## **Conclusion**

The experiments highlight the importance of:

- Careful model selection based on dataset characteristics and computational resources
- Exploring traditional ML techniques alongside deep learning methods
- Leveraging pre-trained embeddings for potential performance gains
- Balancing model complexity with available resources
- Considering data size and hyperparameter tuning for optimal results

## **Further Exploration**

- Experiment with more extensive hyperparameter tuning
- Utilize larger data samples for training
- Explore alternative model architectures and techniques
- Investigate different evaluation metrics beyond accuracy
