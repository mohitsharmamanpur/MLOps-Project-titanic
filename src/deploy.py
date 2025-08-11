import boto3

def upload_to_s3(file_name, bucket_name, object_name=None):
    s3 = boto3.client('s3')
    if object_name is None:
        object_name = file_name
    s3.upload_file(file_name, bucket_name, object_name)
    print(f"Uploaded {file_name} to s3://{bucket_name}/{object_name}")
