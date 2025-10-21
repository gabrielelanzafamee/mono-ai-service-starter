provider "azurerm" {
  features {}
  
  # Explicitly use Azure CLI authentication
  use_cli = true
  
  # Set subscription ID explicitly
  subscription_id = "4ed3492b-2990-4db9-943c-a28a0cefbe4c"
  
  # Set tenant ID explicitly
  tenant_id = "7eaa2e74-6523-4fc6-808b-123fb57c115a"
}

variable "resource_group_name" {
  description = "Name of the existing resource group to use"
  type = string
  default = "Faith"
}

variable "resource_group_id" {
  description = "ID of the resource group"
  type = string
  default = "/subscriptions/4ed3492b-2990-4db9-943c-a28a0cefbe4c/resourceGroups/Faith"
}

variable "location" {
  description = "Azure region where resources will be deployed"
  type = string
  default = "West Europe"
}

variable "acr_name" {
  description = "Name of the Azure Container Registry (must be globally unique)"
  type = string
  default = "monoaiinstance"
}

data "azurerm_resource_group" "rg" {
  name = var.resource_group_name
}

resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = var.resource_group_name
  location            = data.azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
}