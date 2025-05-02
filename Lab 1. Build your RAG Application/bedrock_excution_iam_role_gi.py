"""
AWS IAM Role & OpenSearch Policy Management for Advanced RAG

This script automates the creation of IAM roles, policies, and OpenSearch Serverless configurations
required for an Advanced Retrieval-Augmented Generation (RAG) setup using Amazon Bedrock & Amazon SageMaker.

Key functionalities:
1. **IAM Role Creation**:
   - Creates an Amazon Bedrock execution role with permissions to:
     - Invoke Bedrock foundation models
     - Access S3 for data storage and retrieval
     - Invoke a custom chunking Lambda function
     - Full access to CloudWatch metrics and logs
     - Configure Bedrock model invocation logging

2. **IAM Policy Management**:
   - Defines and attaches policies for:
     - Bedrock foundation models (`bedrock:InvokeModel`)
     - S3 storage (`s3:GetObject`, `s3:PutObject`, etc.)
     - Lambda invocation (`lambda:InvokeFunction`)
     - OpenSearch Serverless API access (`aoss:APIAccessAll`)
     - CloudWatch full access (`cloudwatch:*`)
     - Bedrock logging configuration (`bedrock:PutModelInvocationLoggingConfiguration`, etc.)

3. **OpenSearch Serverless Policies**:
   - Creates security, network, and data access policies for OpenSearch Serverless.
   - Enables encryption, public access settings, and fine-grained permissions.

4. **Bedrock Logging Setup**:
   - Creates CloudWatch log groups for Bedrock model invocation logs
   - Configures Bedrock to use these log groups

This script is designed for use in an AWS environment with proper permissions.
"""

import boto3
import json
import time
from datetime import datetime, UTC
from botocore.exceptions import ClientError
from sagemaker import get_execution_role

# Initialize AWS clients
s3 = boto3.client("s3")
iam = boto3.client("iam")
aoss = boto3.client("opensearchserverless")

# Retrieve AWS credentials and session
credentials = boto3.Session().get_credentials()
boto3_session = boto3.session.Session()


class AdvancedRagIamRoles:
    def __init__(self, account_number, region_name):
        self.account_number = account_number
        self.region_name = region_name
        
        # Initialize additional clients
        self.logs = boto3.client('logs', region_name=region_name)
        self.bedrock = boto3.client('bedrock', region_name=region_name)
        
        # Define the default log group name for Bedrock
        self.bedrock_log_group_name = f"/aws/bedrock/{account_number}/model-invocations"

    # Function to create Amazon Bedrock Execution Role
    def create_bedrock_execution_role(self, bucket_name):
        """Creates an Amazon Bedrock execution role with permissions for Bedrock, S3, Lambda, and CloudWatch."""

        # Define Bedrock foundation model policy
        foundation_model_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeModel"],
                    "Resource": [f"arn:aws:bedrock:{self.region_name}::foundation-model/*"]
                }
            ]
        }

        # Define S3 policy with access restrictions
        s3_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                    "Resource": [
                        f'arn:aws:s3:::{bucket_name}',
                        f'arn:aws:s3:::{bucket_name}/*',
                        f'arn:aws:s3:::{bucket_name}-custom-chunk',
                        f'arn:aws:s3:::{bucket_name}-custom-chunk/*'
                    ],
                    "Condition": {
                        "StringEquals": {"aws:ResourceAccount": self.account_number}
                    }
                }
            ]
        }

        # Define Lambda policy for invoking a custom chunking function
        lambda_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["lambda:InvokeFunction"],
                    "Resource": [
                        f'arn:aws:lambda:{self.region_name}:{self.account_number}:function:advanced-rag-custom-chunk:*'
                    ],
                    "Condition": {
                        "StringEquals": {"aws:ResourceAccount": self.account_number}
                    }
                }
            ]
        }

        # Define CloudWatch full access policy
        cloudwatch_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["cloudwatch:*"],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogStreams",
                        "logs:GetLogEvents"
                    ],
                    "Resource": [
                        f"arn:aws:logs:{self.region_name}:{self.account_number}:log-group:*"
                    ]
                }
            ]
        }

        # Define Bedrock logging configuration policy
        bedrock_logging_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:PutModelInvocationLoggingConfiguration",
                        "bedrock:GetModelInvocationLoggingConfiguration",
                        "bedrock:DeleteModelInvocationLoggingConfiguration"
                    ],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "iam:PassRole"
                    ],
                    "Resource": [
                        f"arn:aws:iam::{self.account_number}:role/*"
                    ],
                    "Condition": {
                        "StringEquals": {
                            "iam:PassedToService": "bedrock.amazonaws.com"
                        }
                    }
                }
            ]
        }

        # Define trust policy for Bedrock execution role
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        # Create policies
        fm_policy = iam.create_policy(
            PolicyName=f"advanced-rag-fm-policy-{self.region_name}",
            PolicyDocument=json.dumps(foundation_model_policy_document),
            Description="Policy for accessing foundation models",
        )

        s3_policy = iam.create_policy(
            PolicyName=f"advanced-rag-s3-policy-{self.region_name}",
            PolicyDocument=json.dumps(s3_policy_document),
            Description="Policy for accessing S3 storage"
        )

        lambda_policy = iam.create_policy(
            PolicyName=f"advanced-rag-lambda-policy-{self.region_name}",
            PolicyDocument=json.dumps(lambda_policy_document),
            Description="Policy for invoking Lambda functions"
        )

        cloudwatch_policy = iam.create_policy(
            PolicyName=f"advanced-rag-cloudwatch-policy-{self.region_name}",
            PolicyDocument=json.dumps(cloudwatch_policy_document),
            Description="Policy for CloudWatch full access"
        )

        bedrock_logging_policy = iam.create_policy(
            PolicyName=f"advanced-rag-bedrock-logging-policy-{self.region_name}",
            PolicyDocument=json.dumps(bedrock_logging_policy_document),
            Description="Policy for Bedrock model invocation logging configuration"
        )

        # Create Bedrock execution role
        bedrock_kb_execution_role = iam.create_role(
            RoleName=f"advanced-rag-workshop-bedrock_execution_role-{self.region_name}",
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            Description="Amazon Bedrock Knowledge Base Execution Role",
            MaxSessionDuration=3600
        )

        # Attach policies to the Bedrock execution role
        iam.attach_role_policy(RoleName=bedrock_kb_execution_role["Role"]["RoleName"], PolicyArn=fm_policy["Policy"]["Arn"])
        iam.attach_role_policy(RoleName=bedrock_kb_execution_role["Role"]["RoleName"], PolicyArn=s3_policy["Policy"]["Arn"])
        iam.attach_role_policy(RoleName=bedrock_kb_execution_role["Role"]["RoleName"], PolicyArn=lambda_policy["Policy"]["Arn"])
        iam.attach_role_policy(RoleName=bedrock_kb_execution_role["Role"]["RoleName"], PolicyArn=cloudwatch_policy["Policy"]["Arn"])
        iam.attach_role_policy(RoleName=bedrock_kb_execution_role["Role"]["RoleName"], PolicyArn=bedrock_logging_policy["Policy"]["Arn"])

        print(f"CloudWatch full access policy attached to {bedrock_kb_execution_role['Role']['RoleName']}")
        print(f"Bedrock logging policy attached to {bedrock_kb_execution_role['Role']['RoleName']}")
        
        # Create and configure CloudWatch log group for Bedrock
        self.create_bedrock_log_group()
        
        # Wait for IAM changes to propagate
        print("Waiting for IAM changes to propagate...")
        time.sleep(10)
        
        # Configure Bedrock logging using the new role
        self.configure_bedrock_model_logging(bedrock_kb_execution_role["Role"]["Arn"])
        
        return bedrock_kb_execution_role

    def create_bedrock_log_group(self):
        """Creates a CloudWatch log group for Bedrock model invocation logs."""
        try:
            self.logs.create_log_group(
                logGroupName=self.bedrock_log_group_name
            )
            
            # Set retention policy (90 days)
            self.logs.put_retention_policy(
                logGroupName=self.bedrock_log_group_name,
                retentionInDays=90
            )
            
            # Add tags
            self.logs.tag_log_group(
                logGroupName=self.bedrock_log_group_name,
                tags={
                    'Service': 'Bedrock',
                    'Purpose': 'ModelInvocationLogging',
                    'Environment': 'Production',
                    'ManagedBy': 'AdvancedRagIamRoles'
                }
            )
            
            print(f"Created CloudWatch log group: {self.bedrock_log_group_name}")
            return True
        except self.logs.exceptions.ResourceAlreadyExistsException:
            print(f"Log group {self.bedrock_log_group_name} already exists")
            return True
        except Exception as e:
            print(f"Error creating log group: {str(e)}")
            return False

    def configure_bedrock_model_logging(self, role_arn):
        """Configures Bedrock to use CloudWatch for model invocation logging."""
        try:
            # Define the logging configuration
            logging_config = {
                'cloudWatchConfig': {
                    'logGroupName': self.bedrock_log_group_name,
                    'roleArn': role_arn
                },
                'textDataDeliveryEnabled': True,
                'embeddingDataDeliveryEnabled': True,
                'imageDataDeliveryEnabled': False,
                'videoDataDeliveryEnabled': False
            }
            
            # Apply the configuration
            response = self.bedrock.put_model_invocation_logging_configuration(
                loggingConfig=logging_config
            )
            
            print(f"Successfully configured Bedrock model invocation logging to {self.bedrock_log_group_name}")
            return response
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            error_message = e.response.get('Error', {}).get('Message', '')
            
            print(f"Error configuring Bedrock logging: {error_code} - {error_message}")
            
            if error_code == 'AccessDeniedException' and 'not authorized to perform: bedrock:PutModelInvocationLoggingConfiguration' in error_message:
                print("The current role doesn't have permission to configure Bedrock logging.")
                print("Make sure to attach the Bedrock logging policy to the SageMaker execution role as well.")
            
            return None

    def add_bedrock_logging_policy_to_sagemaker_role(self, sagemaker_role_name):
        """Adds Bedrock logging permissions to a SageMaker execution role."""
        try:
            # Create the Bedrock logging policy
            bedrock_logging_policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "bedrock:PutModelInvocationLoggingConfiguration",
                            "bedrock:GetModelInvocationLoggingConfiguration",
                            "bedrock:DeleteModelInvocationLoggingConfiguration"
                        ],
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                            "logs:DescribeLogGroups",
                            "logs:DescribeLogStreams"
                        ],
                        "Resource": [
                            f"arn:aws:logs:{self.region_name}:{self.account_number}:log-group:/aws/bedrock/*",
                            f"arn:aws:logs:{self.region_name}:{self.account_number}:log-group:/aws/bedrock*"
                        ]
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "iam:PassRole"
                        ],
                        "Resource": [
                            f"arn:aws:iam::{self.account_number}:role/*"
                        ],
                        "Condition": {
                            "StringEquals": {
                                "iam:PassedToService": "bedrock.amazonaws.com"
                            }
                        }
                    }
                ]
            }
            
            # Create the policy
            try:
                policy_response = iam.create_policy(
                    PolicyName=f"sagemaker-bedrock-logging-policy-{self.region_name}",
                    PolicyDocument=json.dumps(bedrock_logging_policy_document),
                    Description="Policy for SageMaker to configure Bedrock model invocation logging"
                )
                policy_arn = policy_response['Policy']['Arn']
            except iam.exceptions.EntityAlreadyExistsException:
                # Get ARN of existing policy
                policies = iam.list_policies(Scope='Local', PathPrefix='/')
                policy_arn = None
                for policy in policies['Policies']:
                    if policy['PolicyName'] == f"sagemaker-bedrock-logging-policy-{self.region_name}":
                        policy_arn = policy['Arn']
                        break
            
            # Attach the policy to the SageMaker role
            iam.attach_role_policy(
                RoleName=sagemaker_role_name,
                PolicyArn=policy_arn
            )
            
            print(f"Attached Bedrock logging policy to SageMaker role: {sagemaker_role_name}")
            return True
        except Exception as e:
            print(f"Error adding Bedrock logging policy to SageMaker role: {str(e)}")
            return False

    # Function to add OpenSearch Vector Collection access to Bedrock Execution Role
    def create_oss_policy_attach_bedrock_execution_role(self, collection_id, bedrock_kb_execution_role):
        """Creates and attaches an OpenSearch Serverless (OSS) policy to the Bedrock execution role."""
        
        oss_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": [
                        f"arn:aws:aoss:{self.region_name}:{self.account_number}:collection/{collection_id}"
                    ]
                }
            ]
        }

        oss_policy = iam.create_policy(
            PolicyName=f"advanced-rag-oss-policy-{self.region_name}",
            PolicyDocument=json.dumps(oss_policy_document),
            Description="Policy for accessing OpenSearch Serverless",
        )

        # Attach the policy to the Bedrock execution role
        iam.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=oss_policy["Policy"]["Arn"]
        )

        return None

    # Function to attach AWS managed CloudWatch full access policy (alternative approach)
    def attach_cloudwatch_managed_policy(self, role_name):
        """Attaches the AWS managed CloudWatch full access policy to the specified role."""
        
        try:
            # AWS managed policy ARN for CloudWatch full access
            cloudwatch_managed_policy_arn = "arn:aws:iam::aws:policy/CloudWatchFullAccess"
            
            # Attach the AWS managed policy to the role
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=cloudwatch_managed_policy_arn
            )
            
            print(f"AWS managed CloudWatch full access policy attached to {role_name}")
            return True
        except Exception as e:
            print(f"Error attaching CloudWatch managed policy: {str(e)}")
            return False

    # Function to create OpenSearch Serverless security, network, and data access policies
    def create_policies_in_oss(self, vector_store_name, aoss_client, bedrock_kb_execution_role_arn):
        try:
            try:
                # Check if the encryption policy exists
                encryption_policy = aoss_client.create_security_policy(
                    name="advanced-rag-enc-policy2",
                    policy=json.dumps(
                        {
                            'Rules': [{'Resource': ['collection/' + vector_store_name],
                                       'ResourceType': 'collection'}],
                            'AWSOwnedKey': True
                        }),
                    type='encryption'
                )
            except Exception as e:
                print(f"Encryption policy already exists or error: {str(e)}")
        
            try:
                # Check if the network policy exists
                network_policy = aoss_client.create_security_policy(
                    name="advanced-rag-network-policy2",
                    policy=json.dumps(
                        [
                            {'Rules': [{'Resource': ['collection/' + vector_store_name],
                                        'ResourceType': 'collection'}],
                             'AllowFromPublic': True}
                        ]),
                    type='network'
                )
            except Exception as e:
                print(f"Network policy already exists or error: {str(e)}")
        
            try:
                # Check if the access policy exists
                access_policy = aoss_client.create_access_policy(
                    name="advanced-rag-access-policy2",
                    policy=json.dumps(
                        [
                            {
                                'Rules': [
                                    {
                                        'Resource': ['collection/' + vector_store_name],
                                        'Permission': [
                                            'aoss:CreateCollectionItems',
                                            'aoss:DeleteCollectionItems',
                                            'aoss:UpdateCollectionItems',
                                            'aoss:DescribeCollectionItems'],
                                        'ResourceType': 'collection'
                                    },
                                    {
                                        'Resource': ['index/' + vector_store_name + '/*'],
                                        'Permission': [
                                            'aoss:CreateIndex',
                                            'aoss:DeleteIndex',
                                            'aoss:UpdateIndex',
                                            'aoss:DescribeIndex',
                                            'aoss:ReadDocument',
                                            'aoss:WriteDocument'],
                                        'ResourceType': 'index'
                                    }],
                                'Principal': [get_execution_role(), bedrock_kb_execution_role_arn],
                                'Description': 'Easy data policy'}
                        ]),
                    type='data'
                )
            except Exception as e:
                print(f"Access policy already exists or error: {str(e)}")
        
            return encryption_policy, network_policy, access_policy
        except Exception as e:
            print(f"Error: {str(e)}")


# Function to fix Bedrock logging permissions for a SageMaker role
def fix_sagemaker_bedrock_logging_permissions(sagemaker_role_name, account_number, region_name):
    """
    Creates and attaches a policy to enable Bedrock logging permissions for a SageMaker role.
    
    Args:
        sagemaker_role_name (str): Name of the SageMaker execution role
        account_number (str): AWS account number
        region_name (str): AWS region name
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Initialize IAM client
    iam_client = boto3.client('iam')
    
    # Define the policy for Bedrock logging
    bedrock_logging_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:PutModelInvocationLoggingConfiguration",
                    "bedrock:GetModelInvocationLoggingConfiguration",
                    "bedrock:DeleteModelInvocationLoggingConfiguration"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams"
                ],
                "Resource": [
                    f"arn:aws:logs:{region_name}:{account_number}:log-group:/aws/bedrock/*",
                    f"arn:aws:logs:{region_name}:{account_number}:log-group:/aws/bedrock*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "iam:PassRole"
                ],
                "Resource": [
                    f"arn:aws:iam::{account_number}:role/*"
                ],
                "Condition": {
                    "StringEquals": {
                        "iam:PassedToService": "bedrock.amazonaws.com"
                    }
                }
            }
        ]
    }
    
    policy_name = f"sagemaker-bedrock-logging-policy-{region_name}"
    
    try:
        # Create the policy if it doesn't exist
        try:
            policy_response = iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(bedrock_logging_policy_document),
                Description="Allows SageMaker to configure Bedrock model invocation logging"
            )
            policy_arn = policy_response['Policy']['Arn']
            print(f"Created policy: {policy_name}")
        except iam_client.exceptions.EntityAlreadyExistsException:
            # Get the ARN of the existing policy
            paginator = iam_client.get_paginator('list_policies')
            for page in paginator.paginate(Scope='Local'):
                for policy in page['Policies']:
                    if policy['PolicyName'] == policy_name:
                        policy_arn = policy['Arn']
                        print(f"Using existing policy: {policy_name}")
                        break
        
        # Attach the policy to the SageMaker role
        iam_client.attach_role_policy(
            RoleName=sagemaker_role_name,
            PolicyArn=policy_arn
        )
        
        print(f"Successfully attached Bedrock logging policy to SageMaker role: {sagemaker_role_name}")
        return True
    except Exception as e:
        print(f"Error fixing Bedrock logging permissions: {str(e)}")
        return False
