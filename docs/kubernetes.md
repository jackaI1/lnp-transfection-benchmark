# Kubernetes prediction and team lab

The existing prediction service loads the trained model from Blob Storage and checks its SHA-256 before accepting requests. Kubernetes runs the same container and checks `/health` for liveness and `/ready` for a loaded model. The Service gives it a stable internal address. Kustomize combines a shared base with either the kind or AKS overlay. The image, storage account, blob name and checksum must be set for each deployment; the checked-in values are placeholders.

## Team setup

Start the template API locally after `uv sync --locked --python 3.11`:

```sh
uv run uvicorn lnp_benchmark.platform:app --port 8001
curl -sS -X POST http://localhost:8001/teams -H 'Content-Type: application/json' \
  -d '{"team":"lipids","group":"researchers"}' > team-lipids.json
kubectl apply -f team-lipids.json
```

The response is a Kubernetes List: namespace, resource quota, a namespace-scoped edit role binding to an existing identity-provider group, and default-deny network rules with same-team traffic and DNS allowed. Put the JSON in a Git repository and have an administrator review it before Argo CD syncs it. This API generates configuration; it does not authenticate callers or directly change the cluster, so keep it local or put authentication in front of it. NetworkPolicy enforcement requires a compatible cluster network plugin; kind's default network does not enforce it. Add explicit egress rules for storage or other external services before using these policies for workloads. Namespace edit permission is powerful within that namespace; do not give it to untrusted groups without further restrictions.

## Local cluster

On a Linux machine with Docker, kind and kubectl, follow the executable workflow in `.github/workflows/kubernetes.yml`: train the quick model, build and load the image, run Azurite, upload the artifact, apply `kubernetes/overlays/kind`, set its storage connection and checksum, then port-forward the Service. The CI workflow exercises `/health`, `/ready`, valid and invalid predictions, team resources and Argo CD reconciliation. Its emulator credentials are public test-only values. Delete the cluster with `kind delete cluster --name lnp-ci` when finished. The quick model is for testing only.

## Azure AKS

Azure resources cost money. The existing [deployment guide](deployment.md) explains training, creating a registry and uploading a checksum-pinned model. To provision AKS using an existing ACR in the same resource group, run:

```sh
az deployment group create -g "$GROUP" --template-file infra/aks.bicep \
  --parameters registryName="$ACR_NAME" clusterName=lnp-aks
az aks get-credentials -g "$GROUP" -n lnp-aks
```

The Bicep template enables workload identity, Azure network policy and ACR pull access for the kubelet. Create a user-assigned managed identity with Blob Data Reader on the model container, and a federated credential for subject `system:serviceaccount:lnp:lnp-predict` using the AKS OIDC issuer. Annotate the service account with `azure.workload.identity/client-id: <client-id>`. In a deployment repository, configure the AKS overlay's image to `<registry>.azurecr.io/lnp-predict:<immutable-tag>`, set `MODEL_ACCOUNT_URL`, `MODEL_BLOB` and `MODEL_SHA256` in the ConfigMap generator, and commit these non-secret values. Do not commit connection strings. Apply with `kubectl apply -k kubernetes/overlays/aks`, or install Argo CD and point the provided Application at the repository and overlay containing your configuration. Argo CD watches Git and restores changes made outside Git. The checked-in Application points at placeholder configuration, so customize it before syncing. For a public endpoint, add an authenticated ingress and TLS; the included Service is internal only.

No live Azure deployment has been verified. Remove the resource group when finished after checking that it contains only disposable resources.
