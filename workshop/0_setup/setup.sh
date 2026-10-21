function start_mem0() {
    local force_build="$1"
    local target_tag="v2.2.1"
    local base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local repo_path="$base_path/mem0"

    mkdir -p "$base_path"
    if [ ! -d "$repo_path" ]; then
        git clone https://github.com/mem0ai/mem0.git "$repo_path"
    fi

    cd "$repo_path" || return
    git checkout "$target_tag"
    cd server || return

    # Preemptive API Key Validation
    if [[ -z "${OPENAI_API_KEY}" ]]; then
        echo "CRITICAL FAILURE: OPENAI_API_KEY is not exported in the shell environment."
        return 1
    fi

    echo -n "Validating OpenAI API Key with gpt-5.4-mini"
    
    # Capture both the response body and the HTTP status code
    local response
    response=$(curl -s -w "\n%{http_code}" https://api.openai.com/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${OPENAI_API_KEY}" \
        -d '{
            "model": "gpt-5.4-mini",
            "messages": [{"role": "user", "content": "Ping"}],
            "max_completion_tokens": 5
        }')

    # Extract status code from the last line, and the JSON body from the rest
    local api_status
    api_status=$(echo "$response" | tail -n1)
    local api_body
    api_body=$(echo "$response" | sed '$d')

    if [[ "$api_status" -ne 200 ]]; then
        echo " [FAILED]"
        echo "API Key validation failed! OpenAI returned HTTP status: ${api_status}"
        echo "OpenAI Error Details: ${api_body}"
        return 1
    fi
    echo " [SUCCESS]"

    # 0. Configure Environment Variables
    echo "Configuring .env file..."
    cp -f .env.example .env
    
    update_env() {
        local key=$1
        local val=$2
        if grep -q "^${key}=" .env; then
            sed -i.bak "s|^${key}=.*|${key}=${val}|" .env
            rm -f .env.bak
        else
            echo "${key}=${val}" >> .env
        fi
    }

    update_env "POSTGRES_PASSWORD" "postgrespostgres"
    update_env "MEM0_DEFAULT_LLM_MODEL" "gpt-5.4-mini"
    update_env "POSTGRES_HOST" "postgres"
    
    local jwt_secret
    jwt_secret=$(openssl rand -hex 32)
    update_env "JWT_SECRET" "$jwt_secret"

    # 1. Setup Network and Volumes
    podman network create mem0_network 2>/dev/null || true
    podman volume create postgres_db 2>/dev/null || true

    # 2. Build Images
    if [[ "$force_build" == "--force" ]] || ! podman image exists mem0-api 2>/dev/null; then
        echo "Building mem0-api image..."
        podman build -t mem0-api -f dev.Dockerfile ..
    else
        echo "mem0-api image already exists. Skipping build."
    fi

    if [[ "$force_build" == "--force" ]] || ! podman image exists mem0-dashboard 2>/dev/null; then
        echo "Building mem0-dashboard image..."
        podman build -t mem0-dashboard ./dashboard
    else
        echo "mem0-dashboard image already exists. Skipping build."
    fi

    # 3. Start Postgres 
    podman run -d --name postgres \
        --network mem0_network \
        --shm-size 128mb \
        --env-file .env \
        -e POSTGRES_USER=postgres \
        -v postgres_db:/var/lib/postgresql/data \
        -v "$PWD/init-db.sh":/docker-entrypoint-initdb.d/init-db.sh \
        -p 8432:5432 \
        pgvector/pgvector:pg17

    # 3.5. Wait for Postgres Database Initialization
    echo -n "Waiting for Postgres database to initialize"
    until podman exec postgres pg_isready -U postgres -q >/dev/null 2>&1; do
        echo -n "."
        sleep 2
    done
    echo " Ready!"

    # 4. Start Mem0 API
    mkdir -p "$PWD/history"

    podman run -d --name mem0 \
        --network mem0_network \
        -w /app \
        --env-file .env \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -e PYTHONUNBUFFERED=1 \
        -e PYTHONPATH= \
        -e DASHBOARD_URL=http://localhost:3000 \
        -e APP_DB_NAME=mem0_app \
        -e AUTH_DISABLED="${AUTH_DISABLED:-true}" \
        -e MEM0_TELEMETRY="${MEM0_TELEMETRY:-true}" \
        -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
        -v "$PWD/history":/app/history \
        -v "$PWD":/app \
        -p 8888:8000 \
        mem0-api \
        sh -c "rm -rf /app/packages && pip install -q --force-reinstall --no-deps mem0ai && alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

    # 5. Start Mem0 Dashboard
    podman run -d --name mem0-dashboard \
        --network mem0_network \
        -e NEXT_PUBLIC_API_URL=http://localhost:8888 \
        -e API_INTERNAL_URL=http://mem0:8000 \
        -e NEXT_PUBLIC_INSTANCE_NAME=Mem0 \
        -p 3000:3000 \
        mem0-dashboard

    # 6. Wait for Services
    echo -n "Waiting for API"
    until curl -fsS "http://localhost:8888/auth/setup-status" >/dev/null 2>&1; do 
        echo -n "."
        sleep 2
    done
    echo " Ready!"

    echo -n "Waiting for dashboard"
    until curl -fsS "http://localhost:3000/api/health" >/dev/null 2>&1; do 
        echo -n "."
        sleep 2
    done
    echo " Ready!"

    # 7. Seed
    echo "Stack is ready. Bootstrapping..."
    API_URL="http://localhost:8888" DASHBOARD_URL="http://localhost:3000" OUTPUT="text" ./scripts/seed.sh
}

function opensearch-create {
    local NETWORK_NAME="opensearch-net"
    local API_CONTAINER="opensearch-single"
    local DASH_CONTAINER="opensearch-single-dashboards"
    local DATA_DIR="$HOME/opensearch/data"
    local SNAPSHOT_DIR="$HOME/opensearch/snapshots"

    mkdir -p "${DATA_DIR}" "${SNAPSHOT_DIR}"

    if ! podman network exists "${NETWORK_NAME}"; then
        podman network create "${NETWORK_NAME}" >/dev/null
    fi

    podman rm -f "${DASH_CONTAINER}" "${API_CONTAINER}" 2>/dev/null || true

    podman run -d \
        --name "${API_CONTAINER}" \
        --network "${NETWORK_NAME}" \
        -p 9200:9200 -p 9600:9600 \
        -e "discovery.type=single-node" \
        -e "DISABLE_SECURITY_PLUGIN=true" \
        -e "cluster.routing.allocation.disk.threshold_enabled=false" \
        -v "${DATA_DIR}:/usr/share/opensearch/data" \
        -v "${SNAPSHOT_DIR}:/mnt/snapshots" \
        opensearchproject/opensearch:3.5.0  >/dev/null
        # opensearchproject/opensearch:3.2.0  >/dev/null

    podman run -d \
        --name "${DASH_CONTAINER}" \
        --network "${NETWORK_NAME}" \
        -p 5601:5601 \
        -e 'OPENSEARCH_HOSTS=["http://opensearch-single:9200"]' \
        -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
        opensearchproject/opensearch-dashboards:3.5.0
        # opensearchproject/opensearch-dashboards:3.2.0

    echo "API: http://localhost:9200"
    echo "Admin Panel: http://localhost:5601"
}

start_mem0
opensearch-create
