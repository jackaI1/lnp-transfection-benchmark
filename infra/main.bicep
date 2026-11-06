@description('Globally unique names are derived from this resource group.')
param location string = resourceGroup().location
@description('Object ID of the GitHub federated identity, not its client ID.')
param deployerPrincipalId string

var suffix = uniqueString(resourceGroup().id)

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'lnp${suffix}'
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'lnp${suffix}'
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
  }
}
resource blobs 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}
resource models 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobs
  name: 'models'
  properties: { publicAccess: 'None' }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'lnp-inference'
  location: location
}
var pullRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var readerRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1')
var pushRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8311e382-0749-4cb8-b61a-304f252e45ec')
var writerRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')

resource pull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, pullRole)
  scope: registry
  properties: { roleDefinitionId: pullRole, principalId: identity.properties.principalId, principalType: 'ServicePrincipal' }
}
resource read 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(models.id, identity.id, readerRole)
  scope: models
  properties: { roleDefinitionId: readerRole, principalId: identity.properties.principalId, principalType: 'ServicePrincipal' }
}
resource push 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, deployerPrincipalId, pushRole)
  scope: registry
  properties: { roleDefinitionId: pushRole, principalId: deployerPrincipalId, principalType: 'ServicePrincipal' }
}
resource write 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(models.id, deployerPrincipalId, writerRole)
  scope: models
  properties: { roleDefinitionId: writerRole, principalId: deployerPrincipalId, principalType: 'ServicePrincipal' }
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'lnp-logs'
  location: location
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}
resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'lnp-environment'
  location: location
  properties: {
    workloadProfiles: [{ name: 'Consumption', workloadProfileType: 'Consumption' }]
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}
output registryName string = registry.name
output registryServer string = registry.properties.loginServer
output storageName string = storage.name
output accountUrl string = storage.properties.primaryEndpoints.blob
output identityId string = identity.id
output environmentId string = environment.id
