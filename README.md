# Advanced RAG using Amazon Bedrock and Amazon SageMaker

This repository contains a comprehensive workshop demonstrating advanced Retrieval-Augmented Generation (RAG) techniques using AWS services. The labs provide hands-on experience with Amazon Bedrock for foundation models and knowledge bases, and Amazon SageMaker for custom model deployment and inference.

## Overview

Retrieval-Augmented Generation (RAG) enhances large language model (LLM) outputs by incorporating relevant information from external knowledge sources. This workshop explores advanced RAG techniques to improve accuracy, relevance, and trustworthiness of AI-generated responses.

## Labs Overview

This workshop is organized into three main lab sections:

### Lab 1: Knowledge Base Setup and Basic RAG Implementation

- **1.1 Prerequisites**: Set up the environment and prepare documents
- **1.2 Knowledge Base with Fixed Chunking**: Create a knowledge base using fixed-size chunking
- **1.3 Knowledge Base with Semantic Chunking**: Implement semantic chunking for better context preservation
- **1.4 Knowledge Base with Hierarchical Chunking**: Explore hierarchical document organization
- **1.5 Knowledge Base with Custom Chunking**: Implement custom chunking with Lambda functions
- **1.6 Retrieval and Generation using Bedrock FMs**: Basic RAG with Bedrock foundation models
- **1.7 Retrieval and Generation using SageMaker Endpoint**: Integrate with custom SageMaker models
- **1.8 Retrieval and Generation with Query Decomposition**: Enhance RAG with query decomposition

### Lab 2: Advanced RAG Techniques

- **2.1 Create Amazon Bedrock Guardrails**: Implement guardrails for safer AI interactions
- **2.2 Bedrock Inference with Metadata Filtering**: Filter retrieval results using metadata
- **2.3 Bedrock Inference with Guardrails**: Apply guardrails to control generation
- **2.4 Bedrock Inference with Reranking**: Improve search relevance with reranking
- **2.5 SageMaker Inference with Metadata Filtering**: Apply metadata filtering with SageMaker
- **2.6 SageMaker Inference with Guardrails**: Integrate guardrails with SageMaker
- **2.7 SageMaker Inference with Reranking**: Leverage reranking with SageMaker models

### Lab 3: Text-to-SQL Integration

- **3.1 Prerequisites - Set up Database & Crawler**: Configure Amazon Athena and AWS Glue
- **3.2 Text to SQL**: Implement natural language to SQL query capability

### Lab 4: How to Experiment Rapidly with FloTorch

- **4.1 Prerequisites for FloTorch**: Configurations for the FloTorch experiments
- **4.2 Retrieval and Generation - Fixed vs Semantic KB**: Comparing Chunking Methods in RAG Knowledge Bases
- **4.3 Ragas Evaluation - Fixed vs Semantic KB**: Measuring RAG Quality with FloTorch Metrics
- **4.4 Custom Evaluation - Fixed vs Semantic KB**: Custom Evaluation of RAG Knowledge Bases
- **4.5 Retrieval and Generation - Multiple Models**: Evaluating Different Models for RAG Generation
- **4.6 Ragas Evaluation - Multiple Models**: FloTorch RAG Evaluation: Multi-Model Comparison
- **4.7 Custom Evaluation - Multiple Models**: Custom Evaluation Framework for Multiple Models

## Prerequisites

Before starting the labs, ensure you have:

1. An AWS account with appropriate permissions
2. Access to AWS Bedrock (you may need to request access)
3. Python 3.7+ environment
4. Access to Amazon SageMaker Studio or SageMaker notebook instances
5. Required Python libraries (automatically installed in notebooks with `pip install` commands)

## Getting Started

### Setup

1. Clone this repository:
```bash
git clone https://github.com/aws-samples/sample-advanced-rag-using-bedrock-and-sagemaker.git
cd sample-advanced-rag-using-bedrock-and-sagemaker
```

2. Start with Lab 1/1.1 Prerequisites.ipynb to set up the environment.

3. Follow the notebooks in sequence for the best learning experience.

## Key Concepts

Throughout these labs, you'll work with several important concepts:

- **Chunking Strategies**: Learn different approaches to splitting documents (fixed, semantic, hierarchical, custom)
- **Vector Embeddings**: Transform text into numerical vectors for semantic search
- **Metadata Filtering**: Filter search results based on document attributes
- **Guardrails**: Implement safety mechanisms for responsible AI
- **Reranking**: Improve relevance of search results
- **Query Decomposition**: Break complex queries into simpler components
- **Text-to-SQL**: Convert natural language to database queries

## Architecture

The workshop demonstrates a modular RAG architecture with these components:

1. **Document Processing**: Ingest, chunk, and embed documents
2. **Vector Storage**: Store and search embeddings in OpenSearch Serverless
3. **Retrieval**: Fetch relevant documents based on queries
4. **Generation**: Create responses using retrieved context
5. **Enhancement**: Apply techniques like reranking, query decomposition, and guardrails

## Additional Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [OpenSearch Serverless Documentation](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html)
- [AWS Glue Documentation](https://docs.aws.amazon.com/glue/)
- [Amazon Athena Documentation](https://docs.aws.amazon.com/athena/)
