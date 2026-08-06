#!/usr/bin/env bash
# ============================================================================
# AWS ECS Setup Script for RAG Research Assistant
# Run this once to create the infrastructure, then CI/CD handles deploys.
# ============================================================================
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
AWS_REGION="us-east-1"
ECR_REPO="rag-research-assistant"
ECS_CLUSTER="rag-cluster"
ECS_SERVICE="rag-service"
TASK_FAMILY="rag-research-assistant"
VPC_CIDR="10.0.0.0/16"
SUBNET1_CIDR="10.0.1.0/24"
SUBNET2_CIDR="10.0.2.0/24"

echo "=== Step 1: Create ECR Repository ==="
aws ecr create-repository \
    --repository-name "$ECR_REPO" \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || echo "ECR repo already exists"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"

echo "=== Step 2: Create ECS Cluster ==="
aws ecs create-cluster \
    --cluster-name "$ECS_CLUSTER" \
    --capacity-providers FARGATE FARGATE_SPOT \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
    --region "$AWS_REGION" \
    2>/dev/null || echo "Cluster already exists"

echo "=== Step 3: Create VPC and Subnets ==="
VPC_ID=$(aws ec2 create-vpc --cidr-block "$VPC_CIDR" --query 'Vpc.VpcId' --output text 2>/dev/null || echo "")
if [ -n "$VPC_ID" ]; then
    aws ec2 modify-vpc-attribute --vpc-id "$VPC_ID" --enable-dns-hostnames

    SUBNET1_ID=$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block "$SUBNET1_CIDR" \
        --availability-zone "${AWS_REGION}a" --query 'Subnet.SubnetId' --output text)
    SUBNET2_ID=$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block "$SUBNET2_CIDR" \
        --availability-zone "${AWS_REGION}b" --query 'Subnet.SubnetId' --output text)

    IGW_ID=$(aws ec2 create-internet-gateway --query 'InternetGateway.InternetGatewayId' --output text)
    aws ec2 attach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID"

    RTB_ID=$(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID" \
        --query 'RouteTables[0].RouteTableId' --output text)
    aws ec2 create-route --route-table-id "$RTB_ID" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW_ID"

    echo "VPC: $VPC_ID  Subnets: $SUBNET1_ID, $SUBNET2_ID"
fi

echo "=== Step 4: Create Security Group ==="
SG_ID=$(aws ec2 create-security-group \
    --group-name rag-sg \
    --description "RAG Research Assistant SG" \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' --output text 2>/dev/null || echo "")

if [ -n "$SG_ID" ]; then
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 8000 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 443 --cidr 0.0.0.0/0
    echo "Security Group: $SG_ID"
fi

echo "=== Step 5: Create CloudWatch Log Group ==="
aws logs create-log-group \
    --log-group-name /ecs/rag-research-assistant \
    --region "$AWS_REGION" \
    2>/dev/null || echo "Log group exists"

echo "=== Step 6: Store JWT Secret ==="
aws secretsmanager create-secret \
    --name rag/jwt-secret \
    --secret-string "$(openssl rand -base64 32)" \
    --region "$AWS_REGION" \
    2>/dev/null || echo "Secret already exists"

echo "=== Step 7: Build and Push Docker Image ==="
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker build -t "$ECR_REPO" .
docker tag "$ECR_REPO:latest" "$ECR_URI:latest"
docker push "$ECR_URI:latest"

echo "=== Step 8: Register Task Definition ==="
# Replace placeholder ACCOUNT_ID in task def
sed "s/ACCOUNT_ID/$ACCOUNT_ID/g" aws/ecs-task-definition.json > /tmp/task-def.json
aws ecs register-task-definition --cli-input-json file:///tmp/task-def.json --region "$AWS_REGION"

echo "=== Step 9: Create ECS Service ==="
aws ecs create-service \
    --cluster "$ECS_CLUSTER" \
    --service-name "$ECS_SERVICE" \
    --task-definition "$TASK_FAMILY" \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET1_ID,$SUBNET2_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region "$AWS_REGION"

echo ""
echo "=== Deployment Complete ==="
echo "ECR:     $ECR_URI"
echo "Cluster: $ECS_CLUSTER"
echo "Service: $ECS_SERVICE"
echo ""
echo "Next: Set these GitHub Secrets for CI/CD:"
echo "  AWS_ACCESS_KEY_ID"
echo "  AWS_SECRET_ACCESS_KEY"
echo "  AWS_ACCOUNT_ID"
