function delete_mem0_images() {
    local images=(
        "localhost/mem0-dashboard:latest"
        "localhost/mem0-api:latest"
        "docker.io/pgvector/pgvector:pg17"
        "docker.io/opensearchproject/opensearch-dashboards:3.5.0"
        "docker.io/opensearchproject/opensearch:3.5.0"
    )

    echo "Initiating Podman image purge sequence..."

    for img in "${images[@]}"; do
        echo "Attempting to remove container image: ${img}"
        podman rmi "${img}" || echo "Warning: Failed to remove ${img}. It might be actively utilized by a running container or already deleted."
    done

    echo "Image purge sequence complete."
}

delete_mem0_images