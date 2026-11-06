param location string = resourceGroup().location
param registryServer string
param image string
param accountUrl string
param identityId string
param environmentId string
param modelBlob string
@minLength(64)
@maxLength(64)
param modelSha256 string

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'lnp-predict'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityId}': {} }
  }
  properties: {
    managedEnvironmentId: environmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: { external: true, targetPort: 8000, transport: 'auto', allowInsecure: false }
      registries: [{ server: registryServer, identity: identityId }]
    }
    template: {
      containers: [{
        name: 'predict'
        image: image
        resources: { cpu: json('0.5'), memory: '1Gi' }
        env: [
          { name: 'MODEL_ACCOUNT_URL', value: accountUrl }
          { name: 'MODEL_CONTAINER', value: 'models' }
          { name: 'MODEL_BLOB', value: modelBlob }
          { name: 'MODEL_SHA256', value: modelSha256 }
          { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
        ]
        probes: [
          { type: 'Startup', httpGet: { path: '/ready', port: 8000 }, periodSeconds: 5, failureThreshold: 60 }
          { type: 'Readiness', httpGet: { path: '/ready', port: 8000 }, periodSeconds: 10, failureThreshold: 3 }
          { type: 'Liveness', httpGet: { path: '/health', port: 8000 }, periodSeconds: 30, failureThreshold: 3 }
        ]
      }]
      scale: {
        minReplicas: 0
        maxReplicas: 2
        rules: [{ name: 'http', http: { metadata: { concurrentRequests: '8' } } }]
      }
    }
  }
}
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: last(split(identityId, '/'))
}
output url string = 'https://${app.properties.configuration.ingress.fqdn}'
