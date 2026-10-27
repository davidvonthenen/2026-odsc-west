# Load workshop container images from USB

Use this option when internet image downloads or image builds are too slow. The supplied `load-images.sh` imports the workshop's container images from `.tar` archives using `podman load`.

## 1. Start Podman

Verify that the Podman engine is accessible in the terminal where you will load images:

```bash
podman info
```

On macOS or Windows, start the Podman machine if it is stopped. On Windows, run the following workshop commands in your configured WSL2 Bash terminal.

## 2. Copy the USB archives to the workshop directory

Copy the `exported_containers` folder from the instructor-provided USB drive into the same workshop directory as `load-images.sh`.

The resulting structure must be:

```text
scripts/0_setup/
├── load-images.sh
├── setup.sh
└── exported_containers/
    ├── localhost_mem0-dashboard_latest.tar
    ├── localhost_mem0-api_latest.tar
    ├── docker.io_pgvector_pgvector_pg17.tar
    ├── docker.io_opensearchproject_opensearch-dashboards_3.5.0.tar
    └── docker.io_opensearchproject_opensearch_3.5.0.tar
```

Keep the filenames and the `exported_containers` directory name unchanged. If the USB drive contains a compressed bundle, extract it first so these five `.tar` files are available.

`load-images.sh` resolves `exported_containers/` **relative to your current working directory**, so run it from the directory shown above.

## 3. Import the images

From the workshop project directory:

```bash
bash load-images.sh
```

The script loads these five image tags:

| Container image | Tag |
| --- | --- |
| `localhost/mem0-api` | `latest` |
| `localhost/mem0-dashboard` | `latest` |
| `docker.io/pgvector/pgvector` | `pg17` |
| `docker.io/opensearchproject/opensearch` | `3.5.0` |
| `docker.io/opensearchproject/opensearch-dashboards` | `3.5.0` |

Check the output for any `CRITICAL FAILURE` or `podman load` error. **The supplied loader continues past missing archives**, so its completion message alone does not confirm that all images were restored.

## 4. Verify all images

```bash
podman image exists localhost/mem0-api:latest && echo "mem0 API: OK"
podman image exists localhost/mem0-dashboard:latest && echo "mem0 dashboard: OK"
podman image exists docker.io/pgvector/pgvector:pg17 && echo "pgvector: OK"
podman image exists docker.io/opensearchproject/opensearch:3.5.0 && echo "OpenSearch: OK"
podman image exists docker.io/opensearchproject/opensearch-dashboards:3.5.0 && echo "OpenSearch Dashboards: OK"
```

All five lines must print `OK`. If one is missing, confirm that its corresponding `.tar` file is present and repeat the load. You can also inspect tags with `podman images`.

## 5. Continue with setup

Return to [Attendee environment setup](0-attendee-setup.md#3-start-the-workshop-containers) and run:

```bash
bash setup.sh
```

**USB images do not make the entire setup offline.** The supplied `setup.sh` can still access GitHub to obtain the Mem0 source tree, install the Mem0 Python package during startup, and make a live OpenAI API request. Internet connectivity and API keys remain necessary. Importing the images avoids rebuilding or downloading the five named container images when their tags are available locally.

Reference: [Podman image loading](https://docs.podman.io/en/stable/markdown/podman-load.1.html).
