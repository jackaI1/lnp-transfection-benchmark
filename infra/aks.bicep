@description('AKS cluster name')
param clusterName string = 'lnp-aks'
param location string = resourceGroup().location
@description('Existing Azure Container Registry name in this resource group')
param registryName string

resource aks 'Microsoft.ContainerService/managedClusters@2024-09-01' = {
  name: clusterName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    dnsPrefix: clusterName
    enableRBAC: true
    oidcIssuerProfile: { enabled: true }
    securityProfile: { workloadIdentity: { enabled: true } }
    agentPoolProfiles: [{
      name: 'system'
      count: 2
      vmSize: 'Standard_D2s_v3'
      mode: 'System'
      type: 'VirtualMachineScaleSets'
    }]
    networkProfile: { networkPlugin: 'azure', networkPolicy: 'azure' }
  }
}
resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: registryName
}
resource pull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, aks.id, 'acrpull')
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: aks.properties.identityProfile.kubeletidentity.objectId
    principalType: 'ServicePrincipal'
  }
}
output clusterName string = aks.name
output issuer string = aks.properties.oidcIssuerProfile.issuerURL
