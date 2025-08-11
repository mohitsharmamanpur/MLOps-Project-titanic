import boto3
from botocore.exceptions import ClientError

def create_s3_bucket(bucket_name, region="ap-south-1"):
    """Create an S3 bucket in a specific region."""
    try:
        s3_client = boto3.client("s3", region_name=region)
        location = {"LocationConstraint": region}
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration=location
        )
        print(f" Bucket '{bucket_name}' created successfully in region {region}.")
    except ClientError as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    # Change this to your unique bucket name
    bucket_name = "mohit-titanic-mlops-lw-2025"
    create_s3_bucket(bucket_name)
