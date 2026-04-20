#!/bin/bash
#
# GitHub Issue Solver - V3.1 Deployment Script
# Deploys the new secure trial tracking system
#

set -e  # Exit on error

echo "=================================================="
echo "  GitHub Issue Solver V3.1 Deployment"
echo "  Secure Trial Tracking System"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="github-issue-solver"
IMAGE_TAG="v3.1"
VOLUME_NAME="github-issue-solver-data"
CONTAINER_NAME="github-issue-solver"

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/6]${NC} Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR:${NC} Docker is not installed"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo -e "${YELLOW}WARNING:${NC} docker-compose not found, will use docker run instead"
    USE_COMPOSE=false
else
    USE_COMPOSE=true
fi

echo -e "${GREEN}✓${NC} Docker is installed"

# Step 2: Backup existing data (if exists)
echo -e "${YELLOW}[2/6]${NC} Checking for existing deployment..."

if docker ps -a | grep -q $CONTAINER_NAME; then
    echo -e "${YELLOW}WARNING:${NC} Existing container found: $CONTAINER_NAME"
    read -p "Do you want to stop and remove it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping and removing existing container..."
        docker stop $CONTAINER_NAME || true
        docker rm $CONTAINER_NAME || true
        echo -e "${GREEN}✓${NC} Existing container removed"
    else
        echo -e "${RED}ERROR:${NC} Cannot proceed with existing container running"
        exit 1
    fi
fi

# Step 3: Build new Docker image
echo -e "${YELLOW}[3/6]${NC} Building Docker image..."

docker build -t $IMAGE_NAME:$IMAGE_TAG .

# Tag as latest
docker tag $IMAGE_NAME:$IMAGE_TAG $IMAGE_NAME:latest

echo -e "${GREEN}✓${NC} Docker image built successfully"

# Step 4: Create persistent volume
echo -e "${YELLOW}[4/6]${NC} Setting up persistent volume..."

if ! docker volume ls | grep -q $VOLUME_NAME; then
    echo "Creating new volume: $VOLUME_NAME"
    docker volume create $VOLUME_NAME
    echo -e "${GREEN}✓${NC} Volume created"
else
    echo -e "${GREEN}✓${NC} Volume already exists: $VOLUME_NAME"
fi

# Step 5: Verify environment variables
echo -e "${YELLOW}[5/6]${NC} Verifying environment variables..."

if [ -f .env ]; then
    echo -e "${GREEN}✓${NC} .env file found"

    # Check required variables
    if ! grep -q "GITHUB_TOKEN=" .env; then
        echo -e "${RED}ERROR:${NC} GITHUB_TOKEN not found in .env"
        exit 1
    fi

    if ! grep -q "GOOGLE_API_KEY=" .env; then
        echo -e "${RED}ERROR:${NC} GOOGLE_API_KEY not found in .env"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Required environment variables present"
else
    echo -e "${RED}ERROR:${NC} .env file not found"
    echo "Please create .env file with:"
    echo "  GITHUB_TOKEN=your_token_here"
    echo "  GOOGLE_API_KEY=your_key_here"
    exit 1
fi

# Step 6: Deploy
echo -e "${YELLOW}[6/6]${NC} Deploying container..."

if [ "$USE_COMPOSE" = true ] && [ -f docker-compose.yml ]; then
    echo "Using docker-compose..."
    docker compose up -d
else
    echo "Using docker run..."

    # Load environment variables
    source .env

    docker run -d \
        --name $CONTAINER_NAME \
        -v $VOLUME_NAME:/data \
        -e GITHUB_TOKEN=$GITHUB_TOKEN \
        -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
        --restart unless-stopped \
        $IMAGE_NAME:latest
fi

echo -e "${GREEN}✓${NC} Container started successfully"

# Step 7: Verify deployment
echo ""
echo "=================================================="
echo "  Deployment Verification"
echo "=================================================="
echo ""

sleep 3  # Wait for container to start

if docker ps | grep -q $CONTAINER_NAME; then
    echo -e "${GREEN}✓${NC} Container is running"

    echo ""
    echo "Container logs (last 20 lines):"
    echo "------------------------------------------------"
    docker logs --tail 20 $CONTAINER_NAME
    echo "------------------------------------------------"

    echo ""
    echo -e "${GREEN}SUCCESS!${NC} Deployment completed"
    echo ""
    echo "Next steps:"
    echo "  1. Check persistent machine ID in logs above"
    echo "  2. Update your MCP client config (Claude Desktop, etc.)"
    echo "  3. Restart your MCP client"
    echo "  4. Verify trial tracking in Supabase"
    echo ""
    echo "Useful commands:"
    echo "  View logs:        docker logs -f $CONTAINER_NAME"
    echo "  Stop container:   docker stop $CONTAINER_NAME"
    echo "  Start container:  docker start $CONTAINER_NAME"
    echo "  Remove container: docker rm -f $CONTAINER_NAME"
    echo "  Remove volume:    docker volume rm $VOLUME_NAME"
    echo ""

else
    echo -e "${RED}ERROR:${NC} Container failed to start"
    echo ""
    echo "Container logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

echo "=================================================="
echo "  Deployment Complete!"
echo "=================================================="
