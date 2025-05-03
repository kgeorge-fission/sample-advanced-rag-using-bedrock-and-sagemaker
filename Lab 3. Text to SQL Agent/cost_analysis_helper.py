import boto3
from datetime import datetime

BEDROCK_MODEL_PRICING = {
    # Titan Embedding model
    'amazon.titan-embed-text-v2:0': {
        'input_per_million': 0.02,
        'output_per_million': 0.0
    },
    # Amazon Nova models - added based on CSV and search data
    'us.amazon.nova-micro-v1:0': {
        'input_per_million': 0.035,  # $0.000035 per 1,000 tokens
        'output_per_million': 0.12   # $0.00012 per 1,000 tokens
    },
    'us.amazon.nova-lite-v1:0': {
        'input_per_million': 0.12,   # $0.00012 per 1,000 tokens
        'output_per_million': 0.36   # $0.00036 per 1,000 tokens
    },
    'us.amazon.nova-pro-v1:0': {
        'input_per_million': 0.8,    # $0.0008 per 1,000 tokens
        'output_per_million': 3.2    # $0.0032 per 1,000 tokens
    },
    'anthropic.claude-3-5-haiku-20241022-v1:0':{
        'input_per_million': 0.8,    # $0.0008 per 1,000 tokens
        'output_per_million': 4.0    # $0.004 per 1,000 tokens        
    }
    
}

def get_bedrock_token_based_cost(model_id: str, start_time: datetime, end_time: datetime, granularity_seconds_period: int = 5, region="us-west-2"):
    """
    Calculate token counts and costs for Bedrock model usage
    
    Args:
        model_id: The Bedrock model ID
        start_time: Start time for querying CloudWatch metrics
        end_time: End time for querying CloudWatch metrics
        granularity_seconds_period: Period in seconds for CloudWatch metrics
        
    Returns:
        dict: Dictionary containing token counts and costs
    """

    cloudwatch = boto3.client('cloudwatch')

    namespace = 'AWS/Bedrock'
    metrics = ['InputTokenCount', 'OutputTokenCount', 'Invocations']

    results = {}

    for metric in metrics:
        response = cloudwatch.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric,
            Dimensions=[
                {
                    'Name': 'ModelId',
                    'Value': f"arn:aws:bedrock:{region}::foundation-model/{model_id}" if "titan" in model_id else model_id
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=granularity_seconds_period,
            Statistics=['Sum']
        )
        
        # Sum up all datapoints
        total_tokens = sum(dp['Sum'] for dp in response['Datapoints'])
        results[metric] = int(total_tokens)
    
    json_token = {
        'WARNING': "These costs are approximate and directional as of April 2025. They will vary as per region and may change in future. The costs are for Bedrock On-Demand model. If you see zero costs, chances are high that the LLM was not considered in cost calculations when the notebook was prepared. Please use AWS calculator for more accurate cost calculations.",
        'model_id': model_id,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration in minutes': (end_time - start_time).total_seconds() / 60,
        'input_tokens': results.get('InputTokenCount', 0),
        'output_tokens': results.get('OutputTokenCount', 0),
        'invocation_count': results.get('Invocations', 0)
    }
    
    # Get pricing for the model (default to zero if model not found)
    model_pricing = BEDROCK_MODEL_PRICING.get(model_id, {'input_per_million': 0.0, 'output_per_million': 0.0})
    
    # Add pricing info
    json_token['per million input token costs'] = model_pricing['input_per_million']
    json_token['per million output token costs'] = model_pricing['output_per_million']
        
    json_token['input token costs'] = json_token['per million input token costs'] * (float(json_token['input_tokens'])/1000000.0)
    json_token['output token costs'] = json_token['per million output token costs'] * (float(json_token['output_tokens'])/1000000.0)
    json_token['total token costs'] = json_token['input token costs'] + json_token['output token costs']
    
    if json_token['invocation_count'] != 0:
        json_token['average token costs per invocation'] = json_token['total token costs'] / float(json_token['invocation_count'])
    else:
        json_token['average token costs per invocation'] = 0

    json_token['token costs per MILLION such invocations'] = json_token['average token costs per invocation'] * 1000000.0
    
    return json_token