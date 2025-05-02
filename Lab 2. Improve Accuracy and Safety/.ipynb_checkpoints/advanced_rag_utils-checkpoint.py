# Metadata filtering utility functions for advanced_rag_utils.py

import boto3
import json
from typing import Dict, List, Any, Optional, Union, Tuple

def load_variables(file_path="../variables.json"):
    """
    Load configuration variables from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Loaded variables
    """
    with open(file_path, "r") as f:
        variables = json.load(f)
    return variables

def setup_bedrock_client(region_name="us-west-2"):
    """
    Set up a Bedrock agent runtime client.
    
    Args:
        region_name (str): AWS region name
        
    Returns:
        object: Bedrock agent runtime client
    """
    return boto3.client('bedrock-agent-runtime', region_name=region_name)

def get_model_arn(account_number, model_id="us.amazon.nova-pro-v1:0", region="us-west-2"):
    """
    Get the ARN for a Bedrock model.
    
    Args:
        account_number (str): AWS account number
        model_id (str): Bedrock model ID
        region (str): AWS region
        
    Returns:
        str: Model ARN
    """
    return f"arn:aws:bedrock:{region}:{account_number}:inference-profile/{model_id}"

def retrieve_and_generate_without_filter(
    query: str, 
    knowledge_base_id: str, 
    model_arn: str,
    bedrock_client=None,
    num_results: int = 5,
    region_name: str = "us-west-2"
) -> Dict[str, Any]:
    """
    Retrieves and generates a response based on the given query without metadata filtering.

    Args:
        query (str): The input query
        knowledge_base_id (str): The ID of the knowledge base
        model_arn (str): The ARN of the model
        bedrock_client: Bedrock agent runtime client (optional)
        num_results (int): Number of results to retrieve
        region_name (str): AWS region name
        
    Returns:
        dict: Response from the retrieve_and_generate method
    """
    if bedrock_client is None:
        bedrock_client = boto3.client("bedrock-agent-runtime", region_name=region_name)
        
    response = bedrock_client.retrieve_and_generate(
        input={
            "text": query
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                'knowledgeBaseId': knowledge_base_id,
                "modelArn": model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": num_results
                    }
                }
            }
        }
    )
    return response

def retrieve_and_generate_with_filter(
    query: str, 
    knowledge_base_id: str, 
    model_arn: str, 
    metadata_filter: Dict[str, Any],
    bedrock_client=None,
    num_results: int = 5,
    region_name: str = "us-west-2"
) -> Dict[str, Any]:
    """
    Retrieves and generates a response based on the given query with metadata filtering.

    Args:
        query (str): The input query
        knowledge_base_id (str): The ID of the knowledge base
        model_arn (str): The ARN of the model
        metadata_filter (dict): Metadata filter configuration
        bedrock_client: Bedrock agent runtime client (optional)
        num_results (int): Number of results to retrieve
        region_name (str): AWS region name
        
    Returns:
        dict: Response from the retrieve_and_generate method
    """
    if bedrock_client is None:
        bedrock_client = boto3.client("bedrock-agent-runtime", region_name=region_name)
        
    response = bedrock_client.retrieve_and_generate(
        input={
            "text": query
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                'knowledgeBaseId': knowledge_base_id,
                "modelArn": model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": num_results,
                        "filter": metadata_filter
                    }
                }
            }
        }
    )
    return response

def display_rag_response(response: Dict[str, Any]):
    """
    Display the text response from a RAG query.
    
    Args:
        response (dict): Response from retrieve_and_generate
    """
    print(response['output']['text'])

def extract_citations(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract citations from a RAG response.
    
    Args:
        response (dict): Response from retrieve_and_generate
        
    Returns:
        list: List of citation references
    """
    response_ret = response['citations'][0]['retrievedReferences']
    print("# of citations or chunks used to generate the response: ", len(response_ret))
    return response_ret

def citations_rag_print(response_ret: List[Dict[str, Any]]):
    """
    Print citations or chunks of text retrieved.
    
    Args:
        response_ret (list): Retrieved references
    """
    for num, chunk in enumerate(response_ret, 1):
        print(f'Chunk {num}: ', chunk['content']['text'], end='\n'*2)
        print(f'Chunk {num} Location: ', chunk['location'], end='\n'*2)
        print(f'Chunk {num} Metadata: ', chunk['metadata'], end='\n'*2)

def create_dynamic_filter(
    company: Optional[Union[str, List[str]]] = None, 
    year: Optional[Union[int, List[int]]] = None, 
    docType: Optional[str] = None, 
    min_page: Optional[int] = None, 
    max_page: Optional[int] = None, 
    s3_prefix: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Creates a dynamic metadata filter for Amazon Bedrock Knowledge Base queries.
    
    Args:
        company (str or list): Filter by company name (e.g., 'Amazon')
        year (int or list): Filter by year or list of years
        docType (str): Filter by document type (e.g., '10K Report')
        min_page (int): Filter for pages greater than or equal to this number
        max_page (int): Filter for pages less than or equal to this number
        s3_prefix (str): Filter by S3 URI prefix
    
    Returns:
        dict: A metadata filter configuration or None if no valid filters
    """
    filter_conditions = []
    
    # Add company filter if specified and not empty
    if company:
        if isinstance(company, list):
            # Filter out empty strings and check if we have any values left
            company_list = [c for c in company if c]
            if len(company_list) >= 2:
                # If we have at least 2 valid values, use orAll
                company_conditions = []
                for c in company_list:
                    company_conditions.append({
                        "equals": {
                            "key": "company",
                            "value": c
                        }
                    })
                filter_conditions.append({"orAll": company_conditions})
            elif len(company_list) == 1:
                # If only one valid company, use a direct equals condition
                filter_conditions.append({
                    "equals": {
                        "key": "company",
                        "value": company_list[0]
                    }
                })
        elif isinstance(company, str) and company.strip():  # Check if string is not just whitespace
            filter_conditions.append({
                "equals": {
                    "key": "company",
                    "value": company
                }
            })
    
    # Add year filter (single year or multiple years)
    if year:
        if isinstance(year, list):
            # Filter out empty values and check if we have any values left
            year_list = [y for y in year if y]
            if len(year_list) >= 2:
                # If we have at least 2 valid values, use orAll
                year_conditions = []
                for y in year_list:
                    year_conditions.append({
                        "equals": {
                            "key": "year",
                            "value": y
                        }
                    })
                filter_conditions.append({"orAll": year_conditions})
            elif len(year_list) == 1:
                # If only one valid year, use a direct equals condition
                filter_conditions.append({
                    "equals": {
                        "key": "year",
                        "value": year_list[0]
                    }
                })
        elif str(year).strip():  # Convert to string and check if not just whitespace
            filter_conditions.append({
                "equals": {
                    "key": "year",
                    "value": year
                }
            })
    
    # Add document type filter if specified and not empty
    if docType and (not isinstance(docType, str) or docType.strip()):
        filter_conditions.append({
            "equals": {
                "key": "docType",
                "value": docType
            }
        })
    
    # Add minimum page filter if specified
    if min_page is not None:
        filter_conditions.append({
            "greaterThanOrEquals": {
                "key": "x-amz-bedrock-kb-document-page-number",
                "value": min_page
            }
        })
    
    # Add maximum page filter if specified
    if max_page is not None:
        filter_conditions.append({
            "lessThanOrEquals": {
                "key": "x-amz-bedrock-kb-document-page-number",
                "value": max_page
            }
        })

    # Add S3 prefix filter if specified and not empty
    if s3_prefix and (not isinstance(s3_prefix, str) or s3_prefix.strip()):
        filter_conditions.append({
            "startsWith": {
                "key": "x-amz-bedrock-kb-source-uri",
                "value": s3_prefix
            }
        })
    
    # Return the complete filter only if we have TWO OR MORE conditions
    # The API requires at least 2 conditions for andAll
    if len(filter_conditions) >= 2:
        return {"andAll": filter_conditions}
    # If we have exactly ONE condition, return it directly without andAll
    elif len(filter_conditions) == 1:
        return filter_conditions[0]
    else:
        # Return None if no valid conditions
        return None

def create_standard_filter(docType: str, year: int) -> Dict[str, Any]:
    """
    Create a standard filter for document type and year.
    
    Args:
        docType (str): Document type to filter by
        year (int): Year to filter by
        
    Returns:
        dict: Filter configuration
    """
    return {
        "andAll": [
            {
                "equals": {
                    "key": "docType",
                    "value": docType
                }
            },
            {
                "equals": {
                    "key": "year",
                    "value": year
                }
            }
        ]
    }

def query_financial_data(
    query_text: str, 
    kb_id: str, 
    model_arn: str, 
    bedrock_client=None,
    num_results: int = 5,
    region_name: str = "us-west-2",
    **filter_params
) -> Dict[str, Any]:
    """
    Perform a query against financial data with dynamic filtering.
    
    Args:
        query_text (str): The natural language query
        kb_id (str): Knowledge base ID
        model_arn (str): Model ARN
        bedrock_client: Bedrock agent runtime client (optional)
        num_results (int): Number of results to retrieve
        region_name (str): AWS region name
        **filter_params: Parameters to pass to create_dynamic_filter
        
    Returns:
        dict: Response from Bedrock
    """
    if bedrock_client is None:
        bedrock_client = boto3.client("bedrock-agent-runtime", region_name=region_name)
    
    # Create the filter
    filter_config = create_dynamic_filter(**filter_params)
    
    # Log the filter for debugging
    print("Using filter configuration:")
    print(json.dumps(filter_config, indent=2) if filter_config else "No filter applied")
    
    # Run the query with or without filter based on whether we have a valid filter
    if filter_config is not None:
        response = retrieve_and_generate_with_filter(
            query_text, kb_id, model_arn, filter_config, 
            bedrock_client=bedrock_client, num_results=num_results
        )
    else:
        # If no filter conditions, call the function without filter
        response = retrieve_and_generate_without_filter(
            query_text, kb_id, model_arn,
            bedrock_client=bedrock_client, num_results=num_results
        )
    
    return response

def get_value_by_key_path(d: Dict[str, Any], path: List[str]) -> Any:
    """
    Retrieve a value from a nested dictionary using a key path.

    Args:
        d (dict): The dictionary to search
        path (list): List of keys forming the path to the desired value

    Returns:
        The value at the specified path, or None if not found
    """
    current = d
    for key in path:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return None  # Return None if the path is invalid
    return current

def invoke_converse(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    region_name: str = "us-west-2",
    temperature: float = 0.2,
    max_tokens: int = 4000
) -> Optional[str]:
    """
    Chat with a Bedrock model using the Converse API.
    
    Args:
        system_prompt (str): System instructions/context
        user_prompt (str): User's input/question
        model_id (str): Bedrock model ID
        region_name (str): AWS region name
        temperature (float): Controls randomness (0.0 to 1.0)
        max_tokens (int): Maximum tokens in response
        
    Returns:
        Optional[str]: Model's response or None if error
    """
    try:
        # Initialize Bedrock Runtime client
        client = boto3.client('bedrock-runtime', region_name=region_name)
        
        # Prepare the system prompt
        system_prompt = [{'text': system_prompt}]
        messages = []

        # Format the user's question as a message
        message = {
            "role": "user", 
            "content": [            
                {
                    "text": f"{user_prompt}"
                }
            ]
        }

        # Set inference configuration
        messages.append(message)
        inferenceConfig = {
            "maxTokens": max_tokens,
            "temperature": temperature
        }
        
        # Invoke the API
        response = client.converse(
            modelId=model_id, 
            messages=messages,
            system=system_prompt,
            inferenceConfig=inferenceConfig
        )
        
        # Process the response
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            # Extract and concatenate the content from the response 
            content_list = get_value_by_key_path(response, ['output', 'message', 'content'])
            answer = ""
            for content in content_list:
                text = content.get('text')
                if text:  # Concatenate only if text is not None
                    answer += text
        else:
            # Format an error message if the request was unsuccessful
            answer = f"Error: {response['ResponseMetadata']['HTTPStatusCode']} - {response['Error']['Message']}"
        return answer

    except Exception as e:
        print(f"Error in invoke_converse: {str(e)}")
        return None

def extract_entities_from_query(
    user_prompt: str,
    model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
    region_name: str = "us-west-2"
) -> Tuple[List[int], List[str]]:
    """
    Extract entities (company names and years) from a query.
    
    Args:
        user_prompt (str): The user's query
        model_id (str): Bedrock model ID for entity extraction
        region_name (str): AWS region name
        
    Returns:
        tuple: (years, companies) - Lists of extracted entities
    """
    # System prompt for entity extraction
    system_prompt = """
    You are an expert in extracting entity from queries so that those entities can be used as filters.

    Model Instructions:
    - If company name and year is mentioned in the Query, extract them as entities.
    - Return the information strictly in JSON format where company and year are keys and their corresponding values are an array of strings.
    - If you are not sure if company name and year is mentioned in the query, please return an empty list for the corresponding entity.
    - Please do not return any explanation.
    
    $Query$
    """
    
    # Get response from model
    response = invoke_converse(system_prompt, user_prompt, model_id, region_name)
    
    # Parse JSON response
    try:
        data = json.loads(response)
        
        # Extract years as a list of integers
        if 'year' in data:
            years = data['year']
            years = [int(x) for x in years]
        else:
            years = []
            
        # Extract company names as a list of strings
        if 'company' in data:
            companies = data['company']
        else:
            companies = []
            
        return years, companies
        
    except json.JSONDecodeError:
        print("Error parsing JSON response")
        return [], []
    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
        return [], []

def print_citations(response: Dict[str, Any]) -> None:
    """
    Extract and print citations used in the generated response
    
    Parameters:
    - response (Dict[str, Any]): Response from Bedrock retrieve_and_generate
    """
    try:
        # Extract the response text
        output_text = response['output']['text']
        print("\nGenerated Response:")
        print("=" * 80)
        print(output_text)
        print("=" * 80)
        
        # Extract citations
        if 'citations' in response and response['citations']:
            retrieved_references = response['citations'][0]['retrievedReferences']
            print(f"\nNumber of citations: {len(retrieved_references)}")
            
            for i, reference in enumerate(retrieved_references, 1):
                print(f"\nCitation {i}:")
                print("-" * 40)
                
                # Print content
                if 'content' in reference and 'text' in reference['content']:
                    text = reference['content']['text']
                    print(f"Content: {text[:300]}..." if len(text) > 300 else f"Content: {text}")
                
                # Print location
                if 'location' in reference:
                    print(f"Source: {reference['location']}")
                
                # Print metadata
                if 'metadata' in reference:
                    print("Metadata:")
                    for key, value in reference['metadata'].items():
                        print(f"  - {key}: {value}")
        else:
            print("\nNo citations available in the response.")
    
    except Exception as e:
        print(f"Error extracting citations: {str(e)}")

####################################################################
Notebook 2.2
####################################################################
def retrieve_from_bedrock_with_filter(
    query: str,
    knowledge_base_id: str,
    metadata_filter: dict,
    bedrock_client=None,
    num_results: int = 3,
    region_name: str = "us-west-2"
):
    """
    Retrieve relevant context from Bedrock Knowledge Base with metadata filtering.
    
    Args:
        query (str): The query text to search in the knowledge base
        knowledge_base_id (str): ID of the Bedrock Knowledge Base
        metadata_filter (dict): Metadata filter configuration
        bedrock_client: Bedrock agent runtime client (optional)
        num_results (int): Number of results to retrieve
        region_name (str): AWS region name
        
    Returns:
        list: List of texts from retrieval results
    """
    if bedrock_client is None:
        bedrock_client = setup_bedrock_client(region_name)
        
    try:
        response = bedrock_client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': num_results,
                    "filter": metadata_filter
                }
            }
        )
        return [result['content']['text'] for result in response['retrievalResults']]
    except Exception as e:
        raise RuntimeError(f"Bedrock retrieval failed: {str(e)}")

def format_llama3_prompt(query: str, context: list):
    """
    Format prompt for Llama 3 using retrieved context.
    
    Args:
        query (str): The user's query
        context (list): List of retrieved context texts
        
    Returns:
        str: Formatted prompt for Llama 3
    """
    system_prompt = f"""Use the following context to answer the question. If you don't know the answer, say 'I don't know'.
        Context:
        {" ".join(context)}"
    """

    return f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|start_header_id|>user<|end_header_id|>
        Question: {query}
        <|start_header_id|>assistant<|end_header_id|>
        """.strip()

def generate_sagemaker_response(
    prompt: str, 
    endpoint_name: str,
    generation_config: dict = None
):
    """
    Generate response using SageMaker endpoint.
    
    Args:
        prompt (str): The formatted prompt
        endpoint_name (str): SageMaker endpoint name
        generation_config (dict): Generation configuration parameters
        
    Returns:
        str: Generated response
    """
    import boto3
    import json
    
    if generation_config is None:
        generation_config = {
            "temperature": 0,
            "top_k": 10,
            "max_new_tokens": 5000,
            "stop": "<|eot_id|>"
        }
    
    runtime = boto3.client('sagemaker-runtime')
    
    payload = {
        "inputs": prompt,
        "parameters": generation_config
    }
    
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        result = json.loads(response['Body'].read().decode("utf-8"))
        
        if isinstance(result, list):
            return result[0]['generated_text']
        elif 'generated_text' in result:
            return result['generated_text']
        elif 'generation' in result:
            return result['generation']
        else:
            raise RuntimeError("Unexpected response format")
            
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")

def retrieve_and_generate_with_sagemaker(
    query: str,
    knowledge_base_id: str,
    sagemaker_endpoint: str,
    metadata_filter: dict = None,
    generation_config: dict = None,
    bedrock_client=None,
    num_results: int = 3,
    region_name: str = "us-west-2"
):
    """
    Combine retrieval from Bedrock KB and generation with SageMaker.
    
    Args:
        query (str): User query
        knowledge_base_id (str): Bedrock Knowledge Base ID
        sagemaker_endpoint (str): SageMaker endpoint name
        metadata_filter (dict): Metadata filter for retrieval
        generation_config (dict): Generation parameters
        bedrock_client: Bedrock client (optional)
        num_results (int): Number of results to retrieve
        region_name (str): AWS region
        
    Returns:
        tuple: (response, context) - Generated response and retrieved context
    """
    # Set default generation config if not provided
    if generation_config is None:
        generation_config = {
            "temperature": 0,
            "top_k": 10,
            "max_new_tokens": 5000,
            "stop": "<|eot_id|>"
        }
    
    # Retrieve context from Bedrock KB
    context = retrieve_from_bedrock_with_filter(
        query, 
        knowledge_base_id, 
        metadata_filter,
        bedrock_client,
        num_results,
        region_name
    )
    
    # Format prompt for Llama 3
    prompt = format_llama3_prompt(query, context)
    
    # Generate response using SageMaker
    response = generate_sagemaker_response(
        prompt,
        sagemaker_endpoint,
        generation_config
    )
    
    return response, context


################################################################
#Notebook 2.3
###############################################################
import uuid
import boto3
from botocore.exceptions import ClientError, BotoCoreError

def save_variables(variables, file_path="../variables.json"):
    """
    Save configuration variables to a JSON file.
    
    Args:
        variables (dict): Variables to save
        file_path (str): Path to the JSON file
    """
    with open(file_path, "w") as f:
        json.dump(variables, f, indent=4, default=str)

def setup_bedrock_guardrails_client(region_name="us-west-2"):
    """
    Set up a Bedrock client for guardrails management.
    
    Args:
        region_name (str): AWS region name
        
    Returns:
        object: Bedrock client
    """
    return boto3.client('bedrock', region_name=region_name)

def create_guardrail_if_not_exists(
    bedrock_client=None, 
    guardrail_name="AdvancedRagWorkshopGuardrails", 
    region_name="us-west-2"
):
    """
    Create a guardrail only if it doesn't already exist by name.
    Uses list_guardrails to check for existing guardrails with the same name.
    
    Args:
        bedrock_client: Bedrock client (optional)
        guardrail_name (str): The name of the guardrail
        region_name (str): AWS region name
        
    Returns:
        str: The guardrail ID if successful, None otherwise
    """
    if bedrock_client is None:
        bedrock_client = setup_bedrock_guardrails_client(region_name)
        
    # First, check if a guardrail with this name already exists
    try:
        print(f"Checking if guardrail '{guardrail_name}' already exists...")
        
        # List all guardrails
        response = bedrock_client.list_guardrails()
        
        # The API might return guardrails under 'guardrailSummaries' or 'guardrails' key
        guardrails_key = None
        if 'guardrailSummaries' in response:
            guardrails_key = 'guardrailSummaries'
        elif 'guardrails' in response:
            guardrails_key = 'guardrails'
        
        if not guardrails_key:
            print(f"Unexpected API response format. Keys: {list(response.keys())}")
            # If we can't find any expected key, assume no guardrails exist
            guardrails = []
        else:
            print(f"Found guardrails under key: {guardrails_key}")
            guardrails = response[guardrails_key]
            
            # Handle pagination if needed
            while 'nextToken' in response:
                response = bedrock_client.list_guardrails(nextToken=response['nextToken'])
                if guardrails_key in response:
                    guardrails.extend(response[guardrails_key])
            
            # Debug information
            print(f"Found {len(guardrails)} existing guardrails")
            
            # Check if our guardrail name already exists
            for guardrail in guardrails:
                # Check different possible field names for the name
                guardrail_name_fields = ['name', 'Name']
                for field in guardrail_name_fields:
                    if field in guardrail and guardrail[field] == guardrail_name:
                        # Try different possible field names for the ID
                        for id_field in ['id', 'guardrailId', 'Id', 'GuardrailId']:
                            if id_field in guardrail:
                                guardrail_id = guardrail[id_field]
                                print(f"Guardrail '{guardrail_name}' already exists with ID: {guardrail_id}")
                                print(f"Full guardrail record: {guardrail}")
                                return guardrail_id
                        
                        # If we found the name but not the ID, print the entire record
                        print(f"Found guardrail with matching name but couldn't identify ID field")
                        print(f"Guardrail record: {guardrail}")
                        return None
            
        print(f"No existing guardrail found with name '{guardrail_name}'. Proceeding to create...")
    
    except Exception as e:
        print(f"Error checking existing guardrails: {e}")
        print("Will attempt to create a new guardrail anyway.")
    
    # If we get here, the guardrail doesn't exist or we couldn't check
    # Generate a unique client request token for each request
    client_request_token = str(uuid.uuid4())
    
    try:
        # Create the guardrail
        response = bedrock_client.create_guardrail(
            name=guardrail_name,
            description="Restrict responses to AWS services and Amazon 10K filings content only",
            blockedInputMessaging="I can only process questions related to AWS services and Amazon 10K filings.",
            blockedOutputsMessaging="I'm an AWS specialist focused on Amazon's 10K filings. I can only provide information related to AWS services and Amazon's financial reporting.",
            topicPolicyConfig={
                'topicsConfig': [
                    {
                        'name': 'non-aws-topics',
                        'definition': 'Any questions or topics not related to AWS services, Amazon cloud infrastructure, or Amazon 10K financial reporting.',
                        'type': 'DENY'
                    },
                    {
                        'name': 'general-trivia',
                        'definition': 'General knowledge questions about geography, capitals, populations, history, or other non-AWS related facts.',
                        'type': 'DENY'
                    },
                    {
                        'name': 'financial-advice',
                        'definition': 'Any recommendations about investments or financial decisions',
                        'type': 'DENY'
                    },
                    {
                        'name': 'legal-interpretation',
                        'definition': 'Interpretation of legal or regulatory requirements',
                        'type': 'DENY'
                    }
                ]
            },
            contentPolicyConfig={
                'filtersConfig': [
                    {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'INSULTS', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'VIOLENCE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'MISCONDUCT', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'PROMPT_ATTACK', 'inputStrength': 'HIGH', 'outputStrength': 'NONE'}
                ]
            },
            contextualGroundingPolicyConfig={
                'filtersConfig': [
                    {'type': 'GROUNDING', 'threshold': 0.1},
                    {'type': 'RELEVANCE', 'threshold': 0.1}
                ]
            },
            wordPolicyConfig={
                'wordsConfig': [
                    {'text': 'material weakness'},
                    {'text': 'undisclosed liabilities'},
                    {'text': 'shareholder lawsuit'},
                    {'text': 'SEC investigation'},
                    {'text': 'accounting irregularities'},
                    {'text': 'restate earnings'},
                    {'text': 'liquidity crisis'},
                    {'text': 'bankruptcy risk'},
                    {'text': 'fraudulent activity'},
                    {'text': 'insider trading'}
                ]
            },
            sensitiveInformationPolicyConfig={
                'piiEntitiesConfig': [
                    {'type': 'NAME', 'action': 'ANONYMIZE'},
                    {'type': 'EMAIL', 'action': 'ANONYMIZE'},
                    {'type': 'PHONE', 'action': 'ANONYMIZE'},
                    {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'ADDRESS', 'action': 'ANONYMIZE'},
                    {'type': 'AGE', 'action': 'ANONYMIZE'},
                    {'type': 'AWS_ACCESS_KEY', 'action': 'ANONYMIZE'},
                    {'type': 'AWS_SECRET_KEY', 'action': 'ANONYMIZE'},
                    {'type': 'CA_HEALTH_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'CREDIT_DEBIT_CARD_CVV', 'action': 'ANONYMIZE'},
                    {'type': 'CREDIT_DEBIT_CARD_EXPIRY', 'action': 'ANONYMIZE'},
                    {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'DRIVER_ID', 'action': 'ANONYMIZE'},
                    {'type': 'INTERNATIONAL_BANK_ACCOUNT_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'IP_ADDRESS', 'action': 'ANONYMIZE'},
                    {'type': 'LICENSE_PLATE', 'action': 'ANONYMIZE'},
                    {'type': 'MAC_ADDRESS', 'action': 'ANONYMIZE'},
                    {'type': 'PASSWORD', 'action': 'ANONYMIZE'},
                    {'type': 'PIN', 'action': 'ANONYMIZE'},
                    {'type': 'SWIFT_CODE', 'action': 'ANONYMIZE'},
                    {'type': 'UK_NATIONAL_HEALTH_SERVICE_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'URL', 'action': 'ANONYMIZE'},
                    {'type': 'USERNAME', 'action': 'ANONYMIZE'},
                    {'type': 'US_BANK_ACCOUNT_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'US_PASSPORT_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'VEHICLE_IDENTIFICATION_NUMBER', 'action': 'ANONYMIZE'},
                    {'type': 'US_BANK_ROUTING_NUMBER', 'action': 'ANONYMIZE'}
                ],
                'regexesConfig': [
                    {
                        'name': 'stock_ticker_with_price',
                        'description': 'Stock ticker with price pattern',
                        'pattern': '\\b[A-Z]{1,5}\\s*[@:]\\s*\\$?\\d+(\\.\\d{1,2})?\\b',
                        'action': 'ANONYMIZE'
                    },
                    {
                        'name': 'financial_figures',
                        'description': 'Large financial figures in billions/millions',
                        'pattern': '\\$\\s*\\d+(\\.\\d+)?\\s*(billion|million|B|M)\\b',
                        'action': 'ANONYMIZE'
                    },
                    {
                        'name': 'earnings_per_share',
                        'description': 'EPS figures',
                        'pattern': 'EPS\\s*(of)?\\s*\\$?\\d+\\.\\d{2}',
                        'action': 'ANONYMIZE'
                    },
                    {
                        'name': 'investor_relations_contact',
                        'description': 'Investor relations contact information',
                        'pattern': '(?i)investor\\s*relations\\s*[^\\n]+\\d{3}[\\.-]\\d{3}[\\.-]\\d{4}',
                        'action': 'ANONYMIZE'
                    }
                ]
            },
            tags=[
                {'key': 'Environment', 'value': 'Production'},
                {'key': 'Department', 'value': 'Finance'}
            ],
            clientRequestToken=client_request_token
        )
        
        # Try to get the guardrail ID from the response
        guardrail_id = None
        for field in ['guardrailId', 'id']:
            if field in response:
                guardrail_id = response[field]
                break
        
        print(f"Successfully created guardrail with ID: {guardrail_id}")
        print(f"Guardrail ARN: {response.get('guardrailArn')}")
        print(f"Version: {response.get('version')}")
        
        return guardrail_id
        
    except Exception as e:
        print(f"Error creating guardrail: {e}")
        # Check if it's because the guardrail already exists
        if 'ConflictException' in str(e) and 'already exists' in str(e):
            print("Guardrail with this name already exists. Please check all existing guardrails.")
            # Since we couldn't find it earlier but it exists, list all guardrails again
            try:
                response = bedrock_client.list_guardrails()
                if 'guardrailSummaries' in response:
                    print("Existing guardrails:")
                    for guardrail in response['guardrailSummaries']:
                        print(f"Name: {guardrail.get('name')}, ID: {guardrail.get('id')}")
            except:
                pass
        return None

def create_guardrail_version(
    guardrail_id, 
    description="Production version 1.0", 
    bedrock_client=None, 
    region_name="us-west-2"
):
    """
    Create a published version of a guardrail.
    
    Args:
        guardrail_id (str): The ID of the guardrail
        description (str): Description of the version
        bedrock_client: Bedrock client (optional)
        region_name (str): AWS region name
        
    Returns:
        str: The version number of the guardrail
    """
    if bedrock_client is None:
        bedrock_client = setup_bedrock_guardrails_client(region_name)
    
    try:
        # Create guardrail version
        response = bedrock_client.create_guardrail_version(
            guardrailIdentifier=guardrail_id,
            description=description
        )
        
        # Get the version from the response
        version = response.get('version')
        print(f"Successfully created guardrail version: {version}")
        
        return version
    
    except Exception as e:
        print(f"Error creating guardrail version: {e}")
        return None


#####################################################
#Notebook 2.4
#####################################################
def get_guardrail_arn(account_number, guardrail_id, region="us-west-2"):
    """
    Get the ARN for a Bedrock guardrail.
    
    Args:
        account_number (str): AWS account number
        guardrail_id (str): Guardrail ID
        region (str): AWS region
        
    Returns:
        str: Guardrail ARN
    """
    return f"arn:aws:bedrock:{region}:{account_number}:guardrail/{guardrail_id}"

def retrieve_and_generate_with_conditional_guardrails(
    query, 
    knowledge_base_id, 
    model_arn, 
    bedrock_client=None,
    metadata_filter=None,
    use_guardrails=False,
    guardrail_id=None,
    guardrail_version=None,
    num_results=5,
    region_name="us-west-2"
):
    """
    Retrieves and generates a response with optional Guardrails application.
    
    Args:
        query (str): The input query
        knowledge_base_id (str): The ID of the knowledge base
        model_arn (str): The ARN of the model
        bedrock_client: Bedrock agent runtime client (optional)
        metadata_filter (dict, optional): The filter for the vector search configuration
        use_guardrails (bool, optional): Whether to apply guardrails
        guardrail_id (str, optional): The ID of the guardrail to apply
        guardrail_version (str, optional): The version of the guardrail
        num_results (int): Number of results to retrieve
        region_name (str): AWS region name
        
    Returns:
        dict: The response from the retrieve_and_generate method
    """
    if bedrock_client is None:
        bedrock_client = boto3.client('bedrock-agent-runtime', region_name=region_name)
    
    # Start with base configuration
    kb_config = {
        'knowledgeBaseId': knowledge_base_id,
        "modelArn": model_arn,
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": num_results
            }
        }
    }
    
    # Add metadata filter if provided
    if metadata_filter:
        kb_config["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = metadata_filter
    
    # Add generation configuration with prompt template
    kb_config["generationConfiguration"] = {
        "promptTemplate": {
            "textPromptTemplate": "Answer the following question based on the context:\n$search_results$\n\nQuestion: {question}"
        }
    }
    
    # Add guardrail configuration only if requested
    if use_guardrails:
        # Validate required parameters
        if not guardrail_id:
            raise ValueError("guardrail_id is required when use_guardrails is True")
        
        guardrail_config = {
            "guardrailId": guardrail_id
        }
        
        # Add version if provided
        if guardrail_version:
            guardrail_config["guardrailVersion"] = guardrail_version
            
        # Add to generation configuration
        kb_config["generationConfiguration"]["guardrailConfiguration"] = guardrail_config
    
    # Make the API call
    response = bedrock_client.retrieve_and_generate(
        input={
            "text": query
        },
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": kb_config
        }
    )
    
    return response

def display_markdown_result(response):
    """
    Display the text response using Markdown formatting.
    
    Args:
        response (dict): Response from retrieve_and_generate
    """
    from IPython.display import display, Markdown
    answer = response['output']['text']
    display(Markdown(answer.replace("$", "\\$")))

def extract_text_response(response):
    """
    Extract the text response from a Bedrock response.
    
    Args:
        response (dict): Response from retrieve_and_generate
        
    Returns:
        str: Text response
    """
    return response['output']['text']