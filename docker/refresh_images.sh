#!/usr/bin/env bash
# Rebuild, witness and re-capture every image a release describes -- A16-03, A16-11.
#
#   EFORECAST_WORKSPACE=... bash docker/refresh_images.sh [release]
#
# Any edit to a file an image carries -- the runtime contract, the witness chain, anything
# under eforecast/ or eforecast_online/ -- makes that image stale against the release, and
# A16-11's gate then refuses to route to it. That is correct: a stale image produces a wrong
# number rather than an error. It also means the steps below always go together, which is why
# they are one command rather than six remembered ones.
#
# Which tag gets built:
#
#   * the **model runtimes** (foundation, lag_llama) keep the tags their runtime contract
#     declares in `eforecast.image_release.RUNTIMES`. Their reference is part of the contract,
#     not of the release.
#   * the **host images** (worker, online) are built as `<repo>:<release>`, and the release
#     records those dated references. The floating tags production may still reference --
#     `eforecast-worker:latest`, `eforecast-online:step02` -- are never moved by this script.
#   * the build each dated tag replaces stays reachable as `<repo>:<release>-previous`.
#
# The witness must be recorded where the *service* reads it, so EFORECAST_WORKSPACE matters:
# without it the recording lands in ~/.cache and the service never sees it (the A13-02 trap).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release="${1:-${EFORECAST_RELEASE:-a16-02}}"
python_bin="${EFORECAST_PYTHON:-/home/smartrue/pipe_env/bin/python}"

if [ -z "${EFORECAST_WORKSPACE:-}" ]; then
  echo "refresh_images: set EFORECAST_WORKSPACE so the witness lands where the service reads it" >&2
  echo "  e.g. EFORECAST_WORKSPACE=\$(sed -n 's/^EFORECAST_WORKSPACE=\"\\(.*\\)\"$/\\1/p' /etc/eforecast/webui.env)" >&2
  exit 2
fi

cd "$here"

# 1. every image the release describes. Each recipe is named here exactly as
#    `eforecast.image_release.RUNTIMES` names it, so that a runtime added there and forgotten
#    here is a test failure rather than an image nobody rebuilt
#    (tests/test_image_release.py::test_the_refresh_script_covers_every_runtime_the_release_requires).
for build in \
  "docker/Dockerfile.foundation|eforecast-foundation:a16-02|" \
  "docker/Dockerfile.lagllama|eforecast-lagllama:a16-02|" \
  "docker/Dockerfile.worker|eforecast-worker:$release|dated" \
  "eforecast_online/Dockerfile|eforecast-online:$release|dated"
do
  IFS='|' read -r recipe tag dated <<< "$build"
  before="$(docker image inspect --format '{{.Id}}' "$tag" 2> /dev/null || true)"
  # Hold the current build under a tag of its own first: with the containerd image store an
  # image is deleted the moment the build moves its last tag, and "$before" then names nothing.
  holding="${tag%:*}:$release-holding"
  [ -n "$dated" ] && [ -n "$before" ] && docker tag "$tag" "$holding"
  printf 'building %-34s ' "$tag"
  DOCKER_BUILDKIT=1 docker build -f "$recipe" -t "$tag" . > /dev/null
  after="$(docker image inspect --format '{{.Id}}' "$tag")"
  if [ -n "$dated" ] && [ -n "$before" ]; then
    if [ "$before" != "$after" ]; then
      # Only a build that actually produced something else moves the pointer, so running this
      # twice does not throw away the build you were keeping.
      docker tag "$holding" "${tag%:*}:$release-previous"
    fi
    docker rmi "$holding" > /dev/null
  fi
  echo ok
done

# 3. the witnesses, in the file the service reads
PYTHONPATH="$here" EFORECAST_RELEASE="$release" "$python_bin" - <<'PY'
import os
from eforecast import image_release, image_runtime, runtime_launcher
release = os.environ['EFORECAST_RELEASE']
for reference in sorted(runtime_launcher.IMAGES.values()):
    witness = image_runtime.record(reference, entrypoint=runtime_launcher.ENTRYPOINT_WITNESS)
    print(f'witnessed {reference:34s} {witness["image_id"][:22]}')
for name, repository in (('worker', 'eforecast-worker'), ('online', 'eforecast-online')):
    reference = f'{repository}:{release}'
    witness = image_runtime.record(reference, entrypoint=image_release.RUNTIMES[name]['entrypoint'])
    print(f'witnessed {reference:34s} {witness["image_id"][:22]}')
print(f'witness file: {image_runtime.witness_file()}')
PY

# 4. the release, pinned to the dated host images, and then checked against what is here
references=(--reference "worker=eforecast-worker:$release" --reference "online=eforecast-online:$release")
PYTHONPATH="$here" "$python_bin" -m eforecast.image_release capture --release "$release" \
  "${references[@]}" --require foundation --require lag_llama --require worker --require online
PYTHONPATH="$here" "$python_bin" -m eforecast.image_release check --release "$release"
