#!/bin/bash

# Configuration
AWS_REGION="us-east-1"
ECR_REPO_BACKEND="rayban-backend"
ECR_REPO_FRONTEND="rayban-frontend"
ECS_CLUSTER="rayban-cluster"
ECS_SERVICE="rayban-service"

# Build and push Docker images
echo "Building and pushing Docker images..."

# Get ECR login token
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push backend
docker build -f backend.Dockerfile -t $ECR_REPO_BACKEND .
docker tag $ECR_REPO_BACKEND:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_BACKEND:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_BACKEND:latest

# Build and push frontend
docker build -f frontend.Dockerfile -t $ECR_REPO_FRONTEND .
docker tag $ECR_REPO_FRONTEND:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_FRONTEND:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_FRONTEND:latest

# Update ECS service
echo "Updating ECS service..."
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --force-new-deployment

echo "Deployment complete!"
