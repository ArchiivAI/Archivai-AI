# ArchivAI Project

## Project Overview
ArchivAI is a sophisticated AI-powered project designed to process, analyze, and extract metadata from various document types. The project integrates multiple modules for Optical Character Recognition (OCR), metadata extraction, inference, and training, leveraging Azure services for deployment and scalability.

## AI Features

ArchivAI’s AI core features include:
### Optical Character Recognition (OCR)
- An AI-driven OCR module converts scanned, non-digital documents into searchable text.
- Implemented using Azure OpenAI GPT-4 API for advanced text recognition.

### Supervised Document Classification
- The system uses supervised learning with labeled data to identify patterns, allowing it to intelligently categorize new documents without manual intervention.
- Fine-tuned Jina embedding v3 model using Transformers and PyTorch library.
- Achieved 92% accuracy on the Tobacco 3482 Dataset.

### Semantic Search & RAG
- Unlike traditional keyword-based searches, our semantic search understands the context behind queries, providing more accurate and relevant results.
- Powered by Cohere embeddings deployed on Azure Machine Learning Studio.
- Utilizes ChromaDB as the vector database for storing and querying document embeddings.
- Generation model is GPT-4o, implemented using LangChain for seamless integration and advanced response generation.

### Auto Metadata Extraction
- Extract any metadata about your documents like `name` or `email` in your emails folder, for example.
- Utilizes Azure OpenAI GPT-4 API for intelligent metadata extraction.

---

## Project Structure

### 1. `app/`
- **Purpose**: Contains the main application logic for metadata extraction and OCR.
- **Key Files**:
  - `main.py`: Entry point for the application.
  - `metadata_extractor.py`: Handles metadata extraction from documents.
  - `ocr_functions.py`: Implements OCR functionalities.
  - `rag_service.py`: Provides retrieval-augmented generation (RAG) services.
  - `API_Docs.md`: Documents the API endpoints for the main application.
- **Integration**: This folder integrates with the `rag/chroma_langchain_db/` for database operations and `train_app/` for model training.

### 2. `app/train_app/`
- **Purpose**: Contains scripts and configurations for training AI models.
- **Key Files**:
  - `main.py`: Entry point for training workflows.
  - `config.py`: Configuration settings for training.
  - `data_preprocessing.py`: Prepares data for training.
  - `model_saving.py`: Handles saving trained models.
  - `train_classes.py`: Defines training classes.
  - `train_script.py`: Executes the training process.
  - `api-docs.md`: Documents the API endpoints for training workflows.
- **Integration**: This folder interacts with `app/` for deploying trained models.


## Integration Workflow
1. **Metadata Extraction**:
   - `app/main.py` serves as the entry point for extracting metadata using `metadata_extractor.py` and `ocr_functions.py`.

2. **RAG Services**:
   - `app/rag_service.py` provides retrieval-augmented generation (RAG) capabilities.
   - It uses Cohere embeddings and GPT-4o API to retrieve relevant documents and generate responses.
   - The vector database (ChromaDB) is used for storing and querying document embeddings.

3. **Training**:
   - `train_app/main.py` initiates the training process, leveraging `data_preprocessing.py` and `train_script.py`.

4. **End-to-End Flow**:
   - The application integrates all modules to provide a seamless workflow for document processing and AI model training.

## Running the Project Locally

### Prerequisites
1. Install Python dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
2. Set up the following environment variables:
   ```cmd
   set AZURE_CLIENT_ID=<your-client-id>
   set AZURE_TENANT_ID=<your-tenant-id>
   set AZURE_CLIENT_SECRET=<your-client-secret>

   set AZURE_API_KEY=<your-api-key>
   set AZURE_INFERENCE_ENDPOINT="https://{yourworkspace}service.services.ai.azure.com/models"
   set AZURE_INFERENCE_CREDENTIAL=<your-secret-key>

   set AZURE_ML_COHERE_EMBED_ENDPOINT="https://{your-model-deployment-on-azure-ml}.eastus2.models.ai.azure.com"
   set AZURE_ML_COHERE_EMBED_CREDENTIAL=<your-secret-key>
   ```

### Steps to Run
1. **Run the Application**:
   - Navigate to the `app/` folder and execute:
     ```cmd
     python main.py
     ```
2. **Run the Training Module**:
   - Navigate to the `train_app/` folder and execute:
     ```cmd
     python main.py
     ```

## Running the Project Using Docker

### Steps to Run
1. **Create an `.env` File**:
   - In the project root directory, create a file named `.env` and add the following environment variables:
     ```env
     AZURE_CLIENT_ID=<your-client-id>
     AZURE_TENANT_ID=<your-tenant-id>
     AZURE_CLIENT_SECRET=<your-client-secret>

     AZURE_API_KEY=<your-api-key>
     AZURE_INFERENCE_ENDPOINT="https://{yourworkspace}service.services.ai.azure.com/models"
     AZURE_INFERENCE_CREDENTIAL=<your-secret-key>

     AZURE_ML_COHERE_EMBED_ENDPOINT="https://{your-model-deployment-on-azure-ml}.eastus2.models.ai.azure.com"
     AZURE_ML_COHERE_EMBED_CREDENTIAL=<your-secret-key>
     ```

2. **Build the Docker Image**:
   - Navigate to the project root directory and execute:
     ```cmd
     docker build -t archivai-app .
     ```

3. **Run the Docker Container**:
   - Execute the following command to start the container using the `.env` file:
     ```cmd
     docker run -p 8000:8000 --env-file .env archivai-app
     ```

4. **Access the Application**:
   - Open your browser and navigate to `http://localhost:8000` to access the application.



## Azure Services Used for Production

### Azure App Service
- Used to deploy the main application located in `app/main.py`.
- Provides a scalable and reliable platform for hosting web applications.

### Azure Virtual Machine (VM)
- Hosts the PostgreSQL database and the vector database (ChromaDB).
- Ensures high availability and performance for database operations.

### Azure Container App
- Used to deploy the training script located in `app/train_app`.
- Runs on an NCA100 VM with A100 GPU for efficient model training.

### Azure OpenAI
- Provides the GPT-4 API for metadata extraction and OCR functionalities.
- Enables advanced AI capabilities for document processing.

### Azure Machine Learning Studio
- Hosts the Cohere embedding model used in RAG services.
- Facilitates seamless integration with the vector database.

### Azure Key Vault
- Stores all secrets and sensitive information securely.
- Ensures private and secure access to credentials and API keys.


## Additional Notes

### Vector Database for RAGService
- The `RAGService` class in `app/rag_service.py` requires:
  - A server for the vector database (e.g., Chroma).
  - An OpenAI key or any embedding model to configure the embeddings.
- Ensure these dependencies are properly set up to initialize the service.

### SQL Database for Data Preprocessing
- The `fetch_data_from_db` function in `app/train_app/data_preprocessing.py` requires:
  - A server for a SQL database to extract data.
  - Alternatively, you can use your own data from `pandas.DataFrame` or Hugging Face Datasets.
- Modify the function as needed to suit your data source.

### CI/CD Workflow
- The CI/CD workflow is defined in `.github/workflows/main_archivai-ai.yml`.
- **Workflow Steps**:
  1. **Build Docker Image**:
     - On every push to the `main` branch, the workflow builds a Docker image.
     - Tags the image based on the commit message.
  2. **Push to Azure Container Registry**:
     - The Docker image is pushed to Azure Container Registry.
  3. **Deploy to Azure Web App**:
     - The image is deployed to Azure Web App using the `azure/webapps-deploy` action.
  4. **Training Script Deployment**:
     - Training scripts in `app/train_app` are deployed on Azure Container App serverless with NCA100 GPU.
- This workflow ensures automated deployment and scalability for the application and training modules.


