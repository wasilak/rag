import requests
import logging
import argparse

logger = logging.getLogger("RAG.OpenWebUI")


class OpenWebUIUploader:
    def __init__(
        self,
        args: argparse.Namespace,
        timeout: int = 60,
    ):
        self.api_url = args.open_webui_url.rstrip("/")
        self.api_key = args.open_webui_api_key
        self.knowledge_id = args.open_webui_knowledge_id
        self.timeout = timeout

    def upload_file(self, file, filename: str) -> str:
        url = f"{self.api_url}/api/v1/files/"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        try:
            # Create a tuple of (filename, file object, content_type)
            files = {"file": (filename, file, "text/markdown")}
            logger.info(f"Uploading file to Open WebUI: {filename}")
            response = requests.post(url, headers=headers, files=files, timeout=self.timeout, verify=False)
            response.raise_for_status()
            data = response.json()
            file_id = data.get("id") or data.get("file_id")
            if not file_id:
                logger.error(f"0/2 File upload succeeded but no file ID returned: {data}")
                return ""
            logger.info(f"1/2 File uploaded to Open WebUI. File ID: {file_id}")
            return file_id
        except Exception as e:
            logger.error(f"0/2 Failed to upload file to Open WebUI: {e}, error: {response.text}")
            return ""

    def add_file_to_knowledge(self, file_id: str) -> bool:
        if not self.knowledge_id:
            logger.warning("1/2 No knowledge_id provided, skipping add_file_to_knowledge.")
            return False

        url = f"{self.api_url}/api/v1/knowledge/{self.knowledge_id}/file/add"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"file_id": file_id}
        try:
            logger.info(f"Adding file {file_id} to knowledge collection {self.knowledge_id}")
            response = requests.post(url, headers=headers, json=data, timeout=self.timeout, verify=False)

            # Check for duplicate content before raising other status errors
            if response.status_code == 400 and "Duplicate content detected" in response.text:
                logger.warning(f"2/2 File {file_id} already exists in knowledge collection (duplicate content)")
                return True  # Return True since this is not a failure case

            response.raise_for_status()
            logger.info(f"2/2 File {file_id} added to knowledge collection {self.knowledge_id}")
            return True
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError):
                logger.error(f"2/2 Failed to add file to knowledge collection: {e}, error: {e.response.text}")
            else:
                logger.error(f"2/2 Failed to add file to knowledge collection: {str(e)}")
            return False

    def upload_and_add(self, file, filename: str) -> str:
        file_id = self.upload_file(file, filename=filename)
        if file_id and self.knowledge_id:
            success = self.add_file_to_knowledge(file_id)
            if not success:
                logger.warning(f"1/2 File uploaded but failed to add to knowledge collection: {file_id}")
        return file_id
