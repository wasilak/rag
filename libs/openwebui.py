import requests
import logging
from typing import Optional

logger = logging.getLogger("RAG.OpenWebUI")

class OpenWebUIUploader:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        knowledge_id: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        :param api_url: Base URL for Open WebUI API (e.g., http://localhost:3000)
        :param api_key: Bearer token for authentication
        :param knowledge_id: Optional knowledge collection ID to add files to
        :param timeout: Timeout for HTTP requests (seconds)
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.knowledge_id = knowledge_id
        self.timeout = timeout

    def upload_file(self, file, filename: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to Open WebUI.

        :param file: File-like object (opened in binary mode) or file path (str)
        :param filename: Optional filename to use for upload (required if file is file-like)
        :return: File ID if upload is successful, None otherwise
        """
        url = f"{self.api_url}/api/v1/files/"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        try:
            files = {"file": (filename, file)}
            logger.info(f"Uploading file to Open WebUI: {filename}")
            response = requests.post(url, headers=headers, files=files, timeout=self.timeout, verify=False)
            response.raise_for_status()
            data = response.json()
            file_id = data.get("id") or data.get("file_id")
            if not file_id:
                logger.error(f"File upload succeeded but no file ID returned: {data}")
                return None
            logger.info(f"File uploaded to Open WebUI. File ID: {file_id}")
            return file_id
        except Exception as e:
            logger.error(f"Failed to upload file to Open WebUI: {e}")
            return None

    def add_file_to_knowledge(self, file_id: str, knowledge_id: Optional[str] = None) -> bool:
        """
        Add a file to a knowledge collection in Open WebUI.

        :param file_id: The ID of the uploaded file
        :param knowledge_id: The knowledge collection ID (if None, use self.knowledge_id)
        :return: True if successful, False otherwise
        """
        knowledge_id = knowledge_id or self.knowledge_id
        if not knowledge_id:
            logger.warning("No knowledge_id provided, skipping add_file_to_knowledge.")
            return False

        url = f"{self.api_url}/api/v1/knowledge/{knowledge_id}/file/add"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"file_id": file_id}
        try:
            logger.info(f"Adding file {file_id} to knowledge collection {knowledge_id}")
            response = requests.post(url, headers=headers, json=data, timeout=self.timeout, verify=False)
            response.raise_for_status()
            logger.info(f"File {file_id} added to knowledge collection {knowledge_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add file to knowledge collection: {e}")
            return False

    def upload_and_add(self, file, filename: Optional[str] = None) -> Optional[str]:
        """
        Upload a file (from path or file-like object) and add it to the knowledge collection (if knowledge_id is set).

        :param file: File-like object (opened in binary mode) or file path (str)
        :param filename: Optional filename to use for upload (required if file is file-like)
        :return: File ID if upload (and add, if applicable) is successful, None otherwise
        """
        file_id = self.upload_file(file, filename=filename)
        if file_id and self.knowledge_id:
            success = self.add_file_to_knowledge(file_id)
            if not success:
                logger.warning(f"File uploaded but failed to add to knowledge collection: {file_id}")
        return file_id
