"""Serve an exported model through real Blob requests, with owned-process cleanup."""

import argparse
import base64
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from azure.storage.blob import BlobServiceClient

from check_endpoint import check


@contextmanager
def process(command, env, port, root, label):
    with (root / f"{label}.log").open("w") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            started = subprocess.check_output(
                ["ps", "-p", str(child.pid), "-o", "lstart="], text=True
            )
            (root / f"{label}.ownership").write_text(
                f"Owner: serve_local.py\nPID: {child.pid}\nStart: {started}Port: {port}\n"
                f"Command: {command!r}\nCleanup: kill -TERM {child.pid}\n"
            )
            yield child
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Owned {label} port {port} is still open")
            print(f"Stopped {label}: PID {child.pid}, port {port} closed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/deploy/model.joblib")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true", help="Check HTTP inference, then stop")
    args = parser.parse_args()
    executable = shutil.which("azurite-blob")
    if not executable:
        parser.error("Install azurite@3.35.0 and add its bin directory to PATH")
    payload = Path(args.model).read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))  # Refuse to interfere with another listener.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        blob_port = sock.getsockname()[1]
    key = base64.b64encode(b"local-emulator-only-key-32-bytes!!").decode()
    connection = (
        f"DefaultEndpointsProtocol=http;AccountName=lnptest;AccountKey={key};"
        f"BlobEndpoint=http://127.0.0.1:{blob_port}/lnptest;"
    )
    with tempfile.TemporaryDirectory(prefix="lnp-local-") as directory:
        root = Path(directory)
        env = {**os.environ, "AZURITE_ACCOUNTS": f"lnptest:{key}"}
        azurite = [
            executable,
            "--blobHost",
            "127.0.0.1",
            "--blobPort",
            str(blob_port),
            "--location",
            str(root),
            "--silent",
            "--skipApiVersionCheck",
        ]
        with process(azurite, env, blob_port, root, "azurite") as storage:
            with BlobServiceClient.from_connection_string(
                connection, retry_total=0, connection_timeout=1
            ) as client:
                for attempt in range(100):
                    if storage.poll() is not None:
                        raise RuntimeError("Azurite exited before startup")
                    try:
                        client.create_container("models")
                        break
                    except Exception:
                        if attempt == 99:
                            raise
                        time.sleep(0.1)
                client.get_blob_client("models", "model.joblib").upload_blob(payload)
            env.update(
                AZURE_STORAGE_CONNECTION_STRING=connection,
                MODEL_BLOB="model.joblib",
                MODEL_CONTAINER="models",
                MODEL_SHA256=checksum,
            )
            command = [
                sys.executable,
                "-m",
                "uvicorn",
                "lnp_benchmark.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ]
            with process(command, env, args.port, root, "api") as api:
                url = f"http://127.0.0.1:{args.port}"
                check(url, checksum)
                if not args.check:
                    print(
                        f"Open {url}/docs to try predictions. Ctrl+C stops both processes.",
                        flush=True,
                    )
                    api.wait()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
