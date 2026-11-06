# Serving the trained model

I wanted to compare proposed lipid structures without rerunning the report or copying a notebook. The prediction service accepts batches of SMILES, text descriptions of molecules, and returns one delivery prediction per input in the original dataset's `Target` units. It uses the same fingerprints and molecular properties as training. A prediction is a research estimate, not a percentage or evidence that a lipid will work in another experiment.

## Try the complete path locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Node.js 22 or later. From the repository root:

```bash
uv sync --locked --python 3.11
npm install --prefix /tmp/lnp-tools azurite@3.35.0 --no-audit --no-fund
export PATH="/tmp/lnp-tools/node_modules/.bin:$PATH"
uv run lnp-benchmark download
uv run lnp-benchmark featurize
uv run lnp-benchmark train
uv run python -m lnp_benchmark.artifact
uv run python scripts/serve_local.py
```

Open <http://127.0.0.1:8000/docs> to try a prediction in the browser, or use another terminal:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"smiles":["CCN(CC)CCCC","CCCCN(C)CCO"]}'
```

The example molecules show the request format; they are not proposed formulations. The response includes `predictions`, `model_id`, `model_sha256`, units and a warning about unfamiliar lipid families. `/health` checks the process; `/ready` checks that the model has loaded. Batches must contain 1 to 64 strings, each at most 2,048 characters. Bad chemistry, extra fields and wrong types return HTTP 422. A mixed valid/invalid batch is rejected together so row order stays unambiguous.

`serve_local.py` uploads the exported model to Azurite, Microsoft's local Blob Storage emulator, then starts the actual HTTP server. Ctrl+C stops both owned processes. It uses temporary data, records each PID, start time, port and cleanup command, and verifies that both ports close. `--check --port 18080` runs the same HTTP checks and exits automatically. `--quick` on training can shorten a development smoke run, but the Azure workflow always runs full training.

## What gets deployed

GitHub Actions runs the checks, trains on the checksum-verified public dataset, and exports the **scaffold split's cross-validation winner**. It keeps the original holdout out of training; this is the evaluated benchmark model, not a newly fitted model using all labels. The image contains the app and pinned Python libraries. The weights are a separate versioned blob at `<git commit>/model.joblib`.

The workflow builds the Docker image on GitHub's Linux runner and pushes it to Azure Container Registry, a private image store. Bicep, Azure's infrastructure template language, creates Blob Storage, a Container Apps environment, a logging workspace and access roles before creating the app. Splitting `main.bicep` and `app.bicep` avoids trying to start an app before its image and weights exist.

The app uses a managed identity, an Azure-issued identity, for `AcrPull` on the registry and `Storage Blob Data Reader` on just the model container. It uses `DefaultAzureCredential` with that identity's client ID. Storage keys and registry admin passwords are disabled. The workflow signs in using GitHub's short-lived OpenID Connect (OIDC) token; no Azure password is stored in GitHub. These choices follow Microsoft's [managed identity image-pull guidance](https://learn.microsoft.com/en-us/azure/container-apps/managed-identity-image-pull) and [Python Blob Storage guidance](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python).

## Connect an Azure subscription

I verified the storage/API path with Azurite and the Linux container in CI, and compiled the infrastructure templates. I have not provisioned a paid Azure subscription for this project, so there is no claimed hosted endpoint. The following setup is for an account that can create a resource group and assign roles. Azure resources, especially the registry and logging workspace, can incur charges even while the app scales to zero.

Install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli), sign in with `az login`, and select your subscription with `az account set --subscription <subscription-id>`. Run these commands in Bash or Zsh, replacing the region if needed:

```bash
GROUP=lnp-benchmark
LOCATION=canadacentral
SUBSCRIPTION=$(az account show --query id -o tsv)
TENANT=$(az account show --query tenantId -o tsv)
az group create --name "$GROUP" --location "$LOCATION" -o none
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.Storage --wait
az provider register --namespace Microsoft.ManagedIdentity --wait
CLIENT=$(az ad app create --display-name lnp-benchmark-github --query appId -o tsv)
PRINCIPAL=$(az ad sp create --id "$CLIENT" --query id -o tsv)
SCOPE="/subscriptions/$SUBSCRIPTION/resourceGroups/$GROUP"
az role assignment create --assignee-object-id "$PRINCIPAL" \
  --assignee-principal-type ServicePrincipal --role Contributor --scope "$SCOPE" -o none
az role assignment create --assignee-object-id "$PRINCIPAL" \
  --assignee-principal-type ServicePrincipal --role 'Role Based Access Control Administrator' \
  --scope "$SCOPE" -o none
```

The second role permits the template to assign the narrower image/storage roles within this group. For GitHub, create the `azure` deployment environment and federate only that environment:

```bash
gh api --method PUT repos/jackaI1/lnp-transfection-benchmark/environments/azure
cat > /tmp/lnp-federation.json <<'JSON'
{
  "name": "github-azure",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:jackaI1/lnp-transfection-benchmark:environment:azure",
  "audiences": ["api://AzureADTokenExchange"]
}
JSON
az ad app federated-credential create --id "$CLIENT" --parameters /tmp/lnp-federation.json -o none
gh variable set AZURE_CLIENT_ID --env azure --body "$CLIENT" --repo jackaI1/lnp-transfection-benchmark
gh variable set AZURE_PRINCIPAL_ID --env azure --body "$PRINCIPAL" --repo jackaI1/lnp-transfection-benchmark
gh variable set AZURE_TENANT_ID --env azure --body "$TENANT" --repo jackaI1/lnp-transfection-benchmark
gh variable set AZURE_SUBSCRIPTION_ID --env azure --body "$SUBSCRIPTION" --repo jackaI1/lnp-transfection-benchmark
gh variable set AZURE_RESOURCE_GROUP --env azure --body "$GROUP" --repo jackaI1/lnp-transfection-benchmark
```

In the repository's Settings → Environments → azure, restrict deployment branches to `main`. An optional required reviewer can control when paid infrastructure starts. Then choose Actions → deploy Azure inference → Run workflow → main, or run:

```bash
gh workflow run deploy.yml --ref main --repo jackaI1/lnp-transfection-benchmark
```

The workflow reuses all CI checks, uploads weights using its scoped data role, builds/pushes a commit-tagged image and applies `app.bicep`. It retries storage/registry authentication while new role assignments propagate. The last step waits for readiness with the **new model checksum**, requests two predictions and verifies invalid chemistry is rejected. The HTTPS URL appears in the workflow summary. Run `scripts/check_endpoint.py <url> --sha256 <digest>` to repeat those checks. The service is a public, bounded research demo; put authentication in front of it before using it as a shared private service.

## Model loading and revisions

At startup, `storage.py` downloads at most 128 MiB, verifies `MODEL_SHA256`, then deserializes the trusted artifact. Joblib uses Python pickle, which can execute code, so only upload artifacts built from this repository's trusted training output. A digest prevents accidental blob replacement; it does not make an untrusted model safe. Runtime versions, feature settings and model dimensions are checked. Missing blobs, incompatible versions and changed bytes fail startup rather than serving a fallback.

Azure configuration uses `MODEL_ACCOUNT_URL`, `MODEL_CONTAINER`, `MODEL_BLOB`, `MODEL_SHA256` and `AZURE_CLIENT_ID`. `AZURE_STORAGE_CONNECTION_STRING` is for the local emulator; the Azure template never sets it. The model loads once per process, not for every request. New weights require a new app revision. Startup, readiness and liveness probes check the loaded service, and a single active revision receives traffic. It can scale from zero to two replicas, each with 0.5 CPU and 1 GiB memory. A cold start includes downloading the model.

To roll back, select an earlier successful workflow run's commit and model checksum, then redeploy `app.bicep` with that run's image and blob path. Keep the matching runtime/image together. To remove the deployment, delete only the dedicated resource group and identity created above:

```bash
az group delete --name lnp-benchmark --yes
az ad app delete --id "$CLIENT"
```

Remove the five environment variables and the `azure` GitHub environment if no longer needed. Deleting the group removes its app, registry, stored models and logging workspace. It does not delete the Entra application, which is why that has a separate command.

## Tests and build checks

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
az bicep build --file infra/main.bicep --outfile /tmp/lnp-foundation.json
az bicep build --file infra/app.bicep --outfile /tmp/lnp-app.json
```

Pytest launches one owned Azurite process on an unused loopback port and cleans it up on failure too. Tests compare offline predictions with Blob-loaded API predictions, reject bad inputs, exercise missing/corrupt artifacts, and verify credential use and closure. The CLI test also exports its cross-validation-selected trained model. CI adds the full saved-data audit and a real-data training smoke run through the HTTP server. The separate Linux image job runs the storage/API tests inside a built non-root container and compares its HTTP predictions with the original fitted model. For that last test locally on Linux with Docker running, build `docker build -t lnp-predict:test .`, then run `LNP_TEST_IMAGE=lnp-predict:test uv run pytest tests/test_serving.py -q`. Host networking is used only by that Linux test.

The original report's dependency lock is preserved byte-for-byte as `results/reference/training.uv.lock`; its source manifest points there so the historical training audit remains valid as serving dependencies change.
