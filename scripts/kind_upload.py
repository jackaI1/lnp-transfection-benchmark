"""Upload the CI model to the disposable in-cluster Azurite service."""

from azure.storage.blob import BlobServiceClient

# Azurite's published development account, for the ephemeral CI cluster only.
connection = (
    "DefaultEndpointsProtocol=http;AccountName=lnptest;"
    "AccountKey=bG9jYWwtZW11bGF0b3Itb25seS1rZXktMzItYnl0ZXMhIQ==;"
    "BlobEndpoint=http://127.0.0.1:10000/lnptest;"
)
with BlobServiceClient.from_connection_string(connection) as service:
    service.create_container("models")
    with open("results/deploy/model.joblib", "rb") as model:
        service.get_blob_client("models", "model.joblib").upload_blob(model)
