import base64
import hashlib
import os
import shutil
import socket
import subprocess
import time
import uuid
from unittest.mock import patch

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from fastapi.testclient import TestClient
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from lnp_benchmark.api import create_app
from lnp_benchmark.artifact import load_verified, predict, validate_bundle, versions
from lnp_benchmark.features import DESCRIPTORS, featurize
from lnp_benchmark.storage import ModelSettings, load_model


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def azurite(tmp_path_factory):
    executable = shutil.which("azurite-blob")
    if not executable:
        pytest.fail(
            "Install azurite@3.35.0 and add its bin directory to PATH; see docs/deployment.md"
        )
    root = tmp_path_factory.mktemp("azurite")
    port = free_port()
    # Public, disposable emulator account. Never a cloud credential.
    key = base64.b64encode(b"local-emulator-only-key-32-bytes!!").decode()
    connection = (
        f"DefaultEndpointsProtocol=http;AccountName=lnptest;AccountKey={key};"
        f"BlobEndpoint=http://127.0.0.1:{port}/lnptest;"
    )
    command = [
        executable,
        "--blobHost",
        "127.0.0.1",
        "--blobPort",
        str(port),
        "--location",
        str(root),
        "--silent",
        "--skipApiVersionCheck",
    ]
    with (root / "process.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=log,
            env={**os.environ, "AZURITE_ACCOUNTS": f"lnptest:{key}"},
        )
        try:
            (root / "ownership.txt").write_text(
                f"Owner: pytest test_serving.py\nPID: {process.pid}\nPort: {port}\n"
                + subprocess.check_output(
                    ["ps", "-p", str(process.pid), "-o", "lstart="], text=True
                )
                + f"Command: {command!r}\nCleanup: kill -TERM {process.pid}\n"
            )
            with BlobServiceClient.from_connection_string(connection) as service:
                for attempt in range(100):
                    if process.poll() is not None:
                        pytest.fail("Azurite exited before startup; inspect its process.log")
                    try:
                        service.create_container("models")
                        break
                    except Exception:
                        if attempt == 99:
                            raise
                        time.sleep(0.1)
            yield connection
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            with socket.socket() as sock:
                assert sock.connect_ex(("127.0.0.1", port)) != 0, "Azurite port still open"
            assert process.poll() is not None
            print(f"Azurite PID {process.pid}, port {port}: stopped and verified closed")


@pytest.fixture
def bundle():
    smiles = ["CCN(CC)CCCC", "CCCCN(C)CCO", "CCCCCC", "c1ccccc1", "CCO", "CC(=O)O"]
    X, _ = featurize(smiles, n_bits=64)
    model = RandomForestRegressor(n_estimators=8, random_state=42, n_jobs=1)
    model.fit(X, [1.1, 2.3, -0.7, 0.2, -1.0, 0.5])
    return {
        "schema": 1,
        "versions": versions(),
        "model": model,
        "model_id": "scaffold/random_forest",
        "quick": True,
        "dataset_sha256": "a" * 64,
        "feature_config": {
            "morgan": "hashed count",
            "radius": 2,
            "n_bits": 64,
            "chirality": True,
            "descriptor_names": list(DESCRIPTORS),
        },
    }


@pytest.fixture
def stored_model(azurite, bundle, tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    joblib.dump(bundle, artifact)
    payload = artifact.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    blob = f"{uuid.uuid4()}/model.joblib"
    with BlobServiceClient.from_connection_string(azurite) as service:
        client = service.get_blob_client("models", blob)
        client.upload_blob(payload)
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", azurite)
        monkeypatch.setenv("MODEL_BLOB", blob)
        monkeypatch.setenv("MODEL_SHA256", checksum)
        yield ModelSettings("", "models", blob, checksum, azurite), bundle
        client.delete_blob()


def test_blob_roundtrip_and_endpoint_match_offline_predictions(stored_model):
    settings, original = stored_model
    loaded = load_model(settings)
    smiles = ["CCN(CC)CCCC", "CCCCN(C)CCO"]
    expected = predict(original, smiles)
    np.testing.assert_allclose(predict(loaded, smiles), expected)
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json()["model_sha256"] == settings.sha256
        response = client.post("/predict", json={"smiles": smiles})
        assert response.status_code == 200
        body = response.json()
        np.testing.assert_allclose(body["predictions"], expected)
        assert body["model_sha256"] == settings.sha256
        assert "not a percentage" in body["target_units"]
        assert "unreliable" in body["warning"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"smiles": []},
        {"smiles": ["CC"] * 65},
        {"smiles": [""]},
        {"smiles": [" "]},
        {"smiles": ["X" * 2049]},
        {"smiles": [123]},
        {"smiles": [None]},
        {"smiles": "CC"},
        {"smiles": ["CC"], "dose": 5},
        {"smiles": ["invalid"]},
        {"smiles": ["CC", "C1CCC"]},
    ],
)
def test_request_validation(bundle, monkeypatch, payload):
    monkeypatch.setenv("MODEL_BLOB", "fixture")
    monkeypatch.setenv("MODEL_SHA256", "a" * 64)
    monkeypatch.setenv("MODEL_ACCOUNT_URL", "https://example.blob.core.windows.net")
    with TestClient(create_app(loader=lambda _: bundle)) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_unloaded_model_is_not_ready():
    client = TestClient(create_app())
    try:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"smiles": ["CC"]}).status_code == 503
    finally:
        client.close()


def test_corrupt_blob_fails_before_deserialization(stored_model, monkeypatch):
    settings, _ = stored_model
    monkeypatch.setenv("MODEL_SHA256", "0" * 64)
    with patch("lnp_benchmark.artifact.joblib.load") as deserialize:
        with pytest.raises(ValueError, match="checksum"):
            with TestClient(create_app()):
                pass
        deserialize.assert_not_called()
    assert settings.sha256 != "0" * 64


def test_missing_blob_has_no_fallback(stored_model, monkeypatch):
    monkeypatch.setenv("MODEL_BLOB", "missing-model")
    with pytest.raises(ResourceNotFoundError):
        with TestClient(create_app()):
            pass


@pytest.mark.parametrize(
    "change",
    [
        {"schema": 2},
        {"versions": {}},
        {"feature_config": {"n_bits": 8}},
    ],
)
def test_incompatible_artifact_rejected(bundle, change):
    bundle.update(change)
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_model_dimension_mismatch_rejected(bundle):
    bundle["feature_config"]["n_bits"] = 128
    with pytest.raises(ValueError, match="disagree"):
        validate_bundle(bundle)


def test_oversized_artifact_is_not_deserialized():
    with patch("lnp_benchmark.artifact.MAX_MODEL_BYTES", 4):
        with pytest.raises(ValueError, match="size"):
            load_verified(b"12345", "0" * 64)


def test_managed_identity_credential_is_used_and_closed(bundle):
    settings = ModelSettings("https://example.blob.core.windows.net", "models", "model", "a" * 64)
    with (
        patch("lnp_benchmark.storage.DefaultAzureCredential") as identity,
        patch("lnp_benchmark.storage.BlobClient") as blob,
        patch("lnp_benchmark.storage.load_verified", return_value=bundle),
    ):
        stream = blob.return_value.download_blob.return_value
        stream.size = 100
        assert load_model(settings) is bundle
        assert blob.call_args.kwargs["credential"] is identity.return_value
        identity.return_value.close.assert_called_once()


def test_managed_identity_credential_closes_on_download_failure():
    settings = ModelSettings("https://example.blob.core.windows.net", "models", "model", "a" * 64)
    with (
        patch("lnp_benchmark.storage.DefaultAzureCredential") as identity,
        patch("lnp_benchmark.storage.BlobClient") as blob,
    ):
        blob.return_value.download_blob.side_effect = RuntimeError("storage unavailable")
        with pytest.raises(RuntimeError, match="storage unavailable"):
            load_model(settings)
        identity.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    "changes",
    [
        {"sha256": "invalid"},
        {"account_url": "http://example.blob.core.windows.net"},
        {"container": ""},
        {"blob": ""},
    ],
)
def test_invalid_storage_settings_fail_before_network(changes):
    settings = {
        "account_url": "https://example.blob.core.windows.net",
        "container": "models",
        "blob": "model",
        "sha256": "a" * 64,
    }
    settings.update(changes)
    with pytest.raises(ValueError):
        ModelSettings(**settings)


@pytest.mark.skipif(
    not os.environ.get("LNP_TEST_IMAGE"), reason="Container exercised in CI build job"
)
def test_container_loads_blob_and_serves_predictions(stored_model, tmp_path):
    settings, bundle = stored_model
    name = "lnp-test-" + uuid.uuid4().hex[:12]
    port = free_port()
    command = ["docker", "run", "--rm", "--detach", "--network", "host", "--name", name]
    for key, value in {
        "AZURE_STORAGE_CONNECTION_STRING": settings.connection_string,
        "MODEL_BLOB": settings.blob,
        "MODEL_SHA256": settings.sha256,
    }.items():
        command.extend(["--env", f"{key}={value}"])
    command += [
        os.environ["LNP_TEST_IMAGE"],
        "uvicorn",
        "lnp_benchmark.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        metadata = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.State.Pid}} {{.State.StartedAt}}", name],
            text=True,
        )
        (tmp_path / "container-ownership.txt").write_text(
            f"Owner: pytest\nContainer: {name}\nPID/start: {metadata}\nPort: {port}\n"
            f"Cleanup: docker stop {name}\n"
        )
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/check_endpoint.py",
                f"http://127.0.0.1:{port}",
                "--sha256",
                settings.sha256,
            ],
            check=True,
        )
        import httpx

        result = httpx.post(f"http://127.0.0.1:{port}/predict", json={"smiles": ["CCN(CC)CCCC"]})
        np.testing.assert_allclose(result.json()["predictions"], predict(bundle, ["CCN(CC)CCCC"]))
    finally:
        subprocess.run(["docker", "stop", "--time", "10", name], capture_output=True)
        assert name not in subprocess.check_output(
            ["docker", "ps", "--format", "{{.Names}}"], text=True
        )
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0
