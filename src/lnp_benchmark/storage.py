"""Load a bounded, checksum-pinned model from Blob Storage or local Azurite."""

import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient

from .artifact import MAX_MODEL_BYTES, load_verified


@dataclass(frozen=True)
class ModelSettings:
    account_url: str
    container: str
    blob: str
    sha256: str
    connection_string: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[a-f0-9]{64}", self.sha256):
            raise ValueError("MODEL_SHA256 must be a lowercase SHA-256 digest")
        if not self.container or not self.blob:
            raise ValueError("Model container and blob are required")
        if not self.connection_string:
            parsed = urlparse(self.account_url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.query:
                raise ValueError("MODEL_ACCOUNT_URL must be an HTTPS account URL")

    @classmethod
    def from_env(cls):
        return cls(
            account_url=os.environ.get("MODEL_ACCOUNT_URL", ""),
            container=os.environ.get("MODEL_CONTAINER", "models"),
            blob=os.environ["MODEL_BLOB"],
            sha256=os.environ["MODEL_SHA256"],
            connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
        )


def load_model(settings):
    credential = None
    if settings.connection_string:
        client = BlobClient.from_connection_string(
            settings.connection_string,
            settings.container,
            settings.blob,
            connection_timeout=5,
            read_timeout=30,
            retry_total=3,
        )
    else:
        credential = DefaultAzureCredential()
        client = BlobClient(
            settings.account_url,
            settings.container,
            settings.blob,
            credential=credential,
            connection_timeout=5,
            read_timeout=30,
            retry_total=3,
        )
    try:
        with client:
            # Use one ETag-pinned download, rather than a size check followed by a racy GET.
            stream = client.download_blob(max_concurrency=1)
            if stream.size > MAX_MODEL_BYTES:
                raise ValueError("Model exceeds size limit")
            return load_verified(stream.readall(), settings.sha256)
    finally:
        if credential:
            credential.close()
