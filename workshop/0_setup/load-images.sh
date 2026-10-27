function load_mem0_images() {
    # Hardcoded array of the exact container images required for the stack
    local images=(
        "localhost/mem0-dashboard:latest"
        "localhost/mem0-api:latest"
        "docker.io/pgvector/pgvector:pg17"
        "docker.io/opensearchproject/opensearch-dashboards:3.5.0"
        "docker.io/opensearchproject/opensearch:3.5.0"
    )

    echo "Initiating Podman image restoration sequence..."

    for img in "${images[@]}"; do
        # Dynamically derive the archive filename by replacing slashes and colons with underscores
        local archive_name="exported_containers/${img//[\/:]/_}.tar"
        
        echo "Locating container archive: ${archive_name}"
        
        if [[ -f "${archive_name}" ]]; then
            echo "Restoring ${img} into local Podman registry..."
            podman load -i "${archive_name}"
        else
            echo "CRITICAL FAILURE: ${archive_name} is missing from the current working directory."
        fi
    done
    
    echo "Image restoration sequence complete."
}

load_mem0_images
