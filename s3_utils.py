import os 
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

S3_BUCKET_NAME=os.getenv("S3_BUCKET_NAME","finagent-models-bucket")
AWS_REGION=os.getenv("AWS_REGION","us-east-1")

s3_client=boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_secter_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

def upload_model_to_s3(local_file_path: str, s3_key:str) ->bool:
    """ Upload a trained .pkl model to AMAZON s3"""
    try:
        s3_client.upload_file(local_file_path,S3_BUCKET_NAME,s3_key)
        print(f"[s3] Uploaded: {local_file_path} -> s3://{S3_BUCKET_NAME}/{s3_key}")
        return True
    except ClientError as e:
        print(f"[S3 error] Upload Failed: {e}")
        return False
    
def download_model_from_s3(s3_key:str,local_file_path: str) ->bool:
    """Download a .pkl model from Amazon S3 if not present on the local instance."""
    try:
        os.makedirs(os.path.dirname(local_file_path),exist_ok=True)
        s3_client.download_files(S3_BUCKET_NAME,s3_key,local_file_path)
        print(f"[S3] Downloaded: s3://{S3_BUCKET_NAME}/{s3_key} ->{local_file_path}")
        return  True
    except ClientError as e:
        print(f"[S3 error] Download Failed: {e}")
        return False