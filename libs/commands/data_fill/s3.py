import re
import boto3


def upload_to_s3(
    markdown_content: str, title: str, folder: str, bucket_name: str
) -> None:
    """
    Sanitizes a title, and uploads markdown content to an S3 bucket within a specified folder.

    Args:
        markdown_content (str): The markdown content to upload.
        title (str): The title of the markdown content. This will be sanitized and used as the filename.
        folder (str): The folder (prefix) within the S3 bucket where the file will be uploaded.
        bucket_name (str): The name of the S3 bucket.
    """
    # Handle None title
    if title is None:
        title = "untitled"

    # Sanitize title by replacing special characters and spaces with underscore
    sanitized_title = re.sub(r"[^a-zA-Z0-9]", "_", title)

    # Ensure the title is not empty after sanitization
    if not sanitized_title:
        raise ValueError("The sanitized title is empty. Please provide a valid title.")

    # lowercase title
    sanitized_title = sanitized_title.lower()

    filename = f"{sanitized_title}.md"
    # Handle empty folder case to avoid double slashes
    s3_key = filename if not folder else f"{folder}/{filename}"

    try:
        # Initialize S3 client. boto3 automatically handles the standard credential chain.
        s3 = boto3.client("s3")

        # Upload the file
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=markdown_content)

        print(f"Successfully uploaded '{filename}' to s3://{bucket_name}/{s3_key}")

    except Exception as e:
        print(f"Error uploading file to S3: {e}")
