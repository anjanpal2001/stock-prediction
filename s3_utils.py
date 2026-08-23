import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# S3 bucket names must be all lowercase and without spaces
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "financial-analysis-models-default").lower().strip()
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

def ensure_bucket_exists():
    """Checks if the S3 bucket exists, and creates it if it does not."""
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
    except ClientError:
        print(f"[S3] Bucket '{S3_BUCKET_NAME}' not found. Creating it now in region '{AWS_REGION}'...")
        try:
            if AWS_REGION == "us-east-1":
                # us-east-1 does not accept LocationConstraint
                s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
            else:
                s3_client.create_bucket(
                    Bucket=S3_BUCKET_NAME,
                    CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
                )
            print(f"[S3] Bucket '{S3_BUCKET_NAME}' created successfully.")
        except Exception as e:
            print(f"[S3 Error] Could not create bucket: {e}")
            raise e

def upload_model_to_s3(local_file_path: str, s3_key: str) -> bool:
    """Uploads a trained .pkl model to Amazon S3, auto-creating the bucket if needed."""
    s3_key = s3_key.replace("\\", "/")
    
    # 1. Ensure bucket exists before uploading
    ensure_bucket_exists()
    
    try:
        s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_key)
        print(f"[S3] Uploaded: {local_file_path} -> s3://{S3_BUCKET_NAME}/{s3_key}")
        return True
    except ClientError as e:
        print(f"[S3 Error] Upload Failed: {e}")
        raise e

def download_model_from_s3(s3_key: str, local_file_path: str) -> bool:
    """Downloads a .pkl model from Amazon S3 if not present locally."""
    s3_key = s3_key.replace("\\", "/")
    try:
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
        print(f"[S3] Downloaded: s3://{S3_BUCKET_NAME}/{s3_key} -> {local_file_path}")
        return True
    except ClientError as e:
        print(f"[S3 Error] Download Failed: {e}")
        return False