
function cleanup_mem0() {
    podman stop mem0-dashboard postgres 2>/dev/null || true
    podman stop mem0 2>/dev/null || true
    podman stop postgres 2>/dev/null || true

    podman rm mem0-dashboard mem0 postgres 2>/dev/null || true
    podman network rm mem0_network 2>/dev/null || true
}

function cleanup_opensearch {
    podman stop opensearch-single opensearch-single-dashboards 2>/dev/null || true
    podman rm opensearch-single opensearch-single-dashboards 2>/dev/null || true

    podman network rm opensearch-net 2>/dev/null || true
}

cleanup_mem0
cleanup_opensearch
