# syntax=docker/dockerfile:1.7

ARG ISAAC_SIM_VERSION=6.0.1
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

ARG ISAAC_SIM_VERSION

LABEL org.opencontainers.image.title="Isaac Sim Renderer Tutorial" \
      org.opencontainers.image.description="Isaac Sim ${ISAAC_SIM_VERSION} with video export support" \
      org.opencontainers.image.version="${ISAAC_SIM_VERSION}"

USER root

# The NGC image contains Isaac Sim, its matching CUDA runtime, and the Kit Python
# environment.  ffmpeg is the only system package added by this tutorial image.
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -m 0775 -o 1234 -g 1234 \
        /workspace \
        /outputs \
        /isaac-sim/.cache \
        /isaac-sim/.nv/ComputeCache \
        /isaac-sim/.nvidia-omniverse/logs \
        /isaac-sim/.nvidia-omniverse/config \
        /isaac-sim/.local/share/ov/data \
        /isaac-sim/.local/share/ov/pkg \
        /var/cache/hub

ENV HOME=/isaac-sim \
    ISAAC_OUTPUT_DIR=/outputs \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace

# NVIDIA's Isaac Sim container user is UID/GID 1234.  Keeping the application
# rootless also makes every persistent cache volume usable without sudo in the
# running container.
USER 1234:1234

# Always invoke examples through Kit's bundled Python environment.
ENTRYPOINT ["/isaac-sim/python.sh"]

# `docker compose up -d` starts a reusable worker.  Use scripts/exec-example.sh
# against it, or scripts/run-example.sh for an ephemeral one-shot container.
CMD ["-c", "import signal; signal.pause()"]
