# Terraform Configuration for Genie ML Infrastructure

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  default = "eu-west-1"
}

variable "environment" {
  default = "production"
}

# S3 Bucket for reference imagery and scan archives
resource "aws_s3_bucket" "genie_images" {
  bucket = "genie-product-intelligence-${var.environment}"
}

resource "aws_s3_bucket_lifecycle_configuration" "scans_lifecycle" {
  bucket = aws_s3_bucket.genie_images.id

  rule {
    id     = "delete-unconsented-scans-30-days"
    status = "Enabled"

    filter {
      prefix = "scans/ephemeral/"
    }

    expiration {
      days = 30
    }
  }
}
