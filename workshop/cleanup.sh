
function cleanup_mem0() {
    podman stop mem0-dashboard postgres 2>/dev/null || true
    podman stop mem0 2>/dev/null || true
    podman stop postgres 2>/dev/null || true

    podman rm mem0-dashboard mem0 postgres 2>/dev/null || true
    podman network rm mem0_network 2>/dev/null || true

    echo "Executing targeted image deletion..."
    # Extract image IDs dynamically based on repository names and forcefully remove them
    IMAGES_TO_REMOVE=$(podman images --filter reference="*mem0-dashboard*" \
                                     --filter reference="*mem0-api*" \
                                     --filter reference="docker.io/pgvector/pgvector" -q)
    
    if [ -n "$IMAGES_TO_REMOVE" ]; then
        podman rmi -f $IMAGES_TO_REMOVE 2>/dev/null || true
    fi

    # echo "Incinerating dangling <none> images..."
    # # The prune command targets and removes all dangling (<none>) images natively
    # podman image prune -f 2>/dev/null || true

    # delete the local copy of mem0 github repo
    rm -rf ./mem0
}

function cleanup_opensearch {
    podman stop opensearch-single opensearch-single-dashboards 2>/dev/null || true
    podman rm opensearch-single opensearch-single-dashboards 2>/dev/null || true

    # clean up network
    podman network rm opensearch-net 2>/dev/null || true

    echo "Executing targeted image deletion..."
    # Extract image IDs dynamically based on repository names and forcefully remove them
    IMAGES_TO_REMOVE=$(podman images --filter reference="*opensearch-single-dashboards*" \
                                     --filter reference="*opensearch-single*" -q)
    
    if [ -n "$IMAGES_TO_REMOVE" ]; then
        podman rmi -f $IMAGES_TO_REMOVE 2>/dev/null || true
    fi

    # echo "Incinerating dangling <none> images..."
    # # The prune command targets and removes all dangling (<none>) images natively
    # podman image prune -f 2>/dev/null || true

    rm -rf $HOME/opensearch
    echo "opensearch reset to factory."
}

cleanup_mem0
cleanup_opensearch
