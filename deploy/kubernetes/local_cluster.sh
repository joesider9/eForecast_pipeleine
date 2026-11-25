#!/usr/bin/env bash
# A Kubernetes cluster of one node on this machine, in containers: the `broker` venue from
# start to finish, to see a case's runs execute as Jobs without touching the installation.
#
#   export EFORECAST_CLUSTER_HOME=/data/eforecast-local-cluster
#   deploy/kubernetes/local_cluster.sh up --shared /data          # once: the cluster and its services
#   deploy/kubernetes/local_cluster.sh image                      # after every change of the code
#   deploy/kubernetes/local_cluster.sh drill                      # the project's own drill, on a synthetic case
#   deploy/kubernetes/local_cluster.sh run <case> datasets --options '{"dry_run": true}'
#   deploy/kubernetes/local_cluster.sh status | stop | start | kubectl ... | token | down <name>
#
# deploy/kubernetes/README.md, "Try it on one workstation", explains each step and its limits.
#
# What it creates is its own and nothing else's: three containers (a k3s node, PostgreSQL,
# RabbitMQ) on a Docker network of their own, and one folder. The installation's service, its
# run database and its routes are neither read nor changed. Needs Docker and no sudo.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
here="$repo/deploy/kubernetes"
me="deploy/kubernetes/local_cluster.sh"

DEFAULT_NAME=eforecast-local-cluster
DEFAULT_PORT=56444                      # the cluster's API, on 127.0.0.1 only
DEFAULT_SUBNET=172.30.78                # the first three numbers of the cluster's own Docker network
DEFAULT_K3S=rancher/k3s:v1.34.1-k3s1    # the version the drill passed on, 17 November 2026
DEFAULT_PG=postgres:16
DEFAULT_RABBIT=rabbitmq:4.1-management-alpine
# Pure-Python packages the API and the broker client need and the worker image does not carry.
# They are lent to the control plane from the interpreter the installation runs.
DEFAULT_PACKAGES="fastapi starlette anyio pika annotated_doc itsdangerous"

say()  { printf '%s\n' "$*"; }
fail() { printf 'local_cluster: %s\n' "$*" >&2; exit 2; }

usage() {
  cat <<EOF
$me <command>

  up --shared FOLDER [--home FOLDER] [--workspace FOLDER] [--name NAME] [--port N]
     [--subnet A.B.C] [--cpus N] [--memory 24g] [--job-cpus N] [--job-memory-mb N]
     [--job-hours N] [--token-hours N] [--python INTERPRETER]
                    create the cluster: a network, PostgreSQL, RabbitMQ and one k3s node, the
                    namespace, the two accounts, the connection secret and a token.
                    --shared is the one folder of this machine the Pods will see, under the
                    same path: it must hold the workspace, the data, the weather and the models.
  image             build the worker image from this checkout, check that it carries exactly
                    this code, copy it into the node and write the two target files.
                    Again after every change under eforecast/: a Job refuses any other code.
  status            what is there and whether it is ready.
  drill             the project's drill on a synthetic case: submit and redeliver, cancel, a
                    real dataset stage.
  run CASE datasets|training|combine [--options JSON] [--seconds N]
                    one step of a case of the workspace, as a Job. It prints what it replaces
                    and goes on, like a run typed in a terminal; {"dry_run": true} only plans.
  kubectl ...       kubectl, inside the node:  $me kubectl -n eforecast get jobs,pods
  token             a new token for the coordinator (they expire; "start" issues one too).
  stop [--force]    stop the three containers; everything is kept.   start   start them again.
  down NAME         remove the cluster: containers, network, databases, its folder.

Every command but "up" finds the cluster through EFORECAST_CLUSTER_HOME, the folder given to
"up --home" (default: <shared>/$DEFAULT_NAME).
EOF
}

# ---------------------------------------------------------------- settings
installation() {     # what the installation's settings file says, where this shell says nothing
  local file="${EFORECAST_ENV_FILE:-/etc/eforecast/webui.env}" name value
  [ -r "$file" ] || return 0
  for name in EFORECAST_PYTHON EFORECAST_WORKSPACE; do
    [ -n "$(eval "printf '%s' \"\${$name:-}\"")" ] && continue
    value=$(sed -n "s/^$name=\"\(.*\)\"\$/\1/p; s/^$name=\([^\"].*\)\$/\1/p" "$file" | tail -1)
    [ -n "$value" ] && export "$name=$value"
  done
  return 0
}

plain() {            # <what> <path>: a path that can be written into JSON and a settings file as it is
  case "$2" in /*) ;; *) fail "$1 must be an absolute path: $2" ;; esac
  case "$2" in *[!A-Za-z0-9_./-]*) fail "$1 may hold letters, digits, '_', '-', '.' and '/' only: $2" ;; esac
}

number() {           # <what> <value>
  case "$2" in ''|*[!0-9]*) fail "$1 must be a whole number: $2" ;; esac
  [ "$2" -gt 0 ] || fail "$1 must be positive: $2"
}

load() {             # the cluster every command but "up" is about
  home="${EFORECAST_CLUSTER_HOME:-}"
  [ -n "$home" ] || fail "which cluster? export EFORECAST_CLUSTER_HOME=<the folder given to 'up --home'>"
  [ -f "$home/cluster.env" ] || fail "$home holds no cluster (no cluster.env). Create one: $me up --home $home --shared <folder>"
  # shellcheck disable=SC1090,SC1091
  . "$home/cluster.env"
  node="$NAME"; database="$NAME-pg"; broker="$NAME-rabbit"; image="eforecast-worker:$NAME"
}

is_running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

need_running() {
  if is_running "$node" && is_running "$database" && is_running "$broker"; then return 0; fi
  fail "the cluster $NAME is not running: $me start"
}

need_image() {
  docker image inspect "$image" >/dev/null 2>&1 || fail "no worker image yet: $me image"
  [ -f "$home/private/target-drill.json" ] || fail "no target file yet: $me image"
}

kube() { docker exec -i "$node" kubectl "$@"; }

wait_services() {
  local n
  for n in $(seq 1 60); do
    docker exec "$database" pg_isready -U postgres -d eforecast >/dev/null 2>&1 && break
    sleep 2
  done
  docker exec "$database" pg_isready -U postgres -d eforecast >/dev/null 2>&1 \
    || fail "PostgreSQL does not answer; look at: docker logs $database"
  for n in $(seq 1 60); do
    docker exec "$broker" rabbitmq-diagnostics -q ping >/dev/null 2>&1 && return 0
    sleep 2
  done
  fail "RabbitMQ does not answer; look at: docker logs $broker"
}

wait_node() {
  local n nodes
  for n in $(seq 1 100); do
    nodes=$(kube get nodes --no-headers 2>/dev/null || true)
    case "$nodes" in *" Ready"*) return 0 ;; esac
    sleep 3
  done
  fail "the node did not become Ready in five minutes; look at: docker logs $node"
}

issue_token() {      # the coordinator's way into the cluster: the executor account, for a while
  ( umask 077
    kube -n eforecast create token eforecast-executor --duration="${TOKEN_HOURS}h" > "$home/private/executor.token.new"
    mv "$home/private/executor.token.new" "$home/private/executor.token"
    date -d "+${TOKEN_HOURS} hours" '+%Y-%m-%d %H:%M' > "$home/private/executor.token.until" )
}

site_packages() {    # where the lent packages are; fails when one is missing
  local site package
  site=$("$PYTHON" -c 'import os, fastapi; print(os.path.dirname(os.path.dirname(fastapi.__file__)))' 2>/dev/null) \
    || fail "$PYTHON has no fastapi. The control plane borrows $PACKAGES from an interpreter that runs the web service: name it with 'up --python' or EFORECAST_PYTHON (they are in requirements.txt)"
  for package in $PACKAGES; do
    [ -e "$site/$package" ] || fail "$PYTHON has no $package (looked in $site); install it there: it is in requirements.txt"
  done
  printf '%s\n' "$site"
}

control() {          # <target file> <workspace> <arguments of python...>: one program of the control plane
  local target="$1" workspace="$2" site package; shift 2
  site=$(site_packages)
  local mounts=()
  for package in $PACKAGES; do mounts+=(-v "$site/$package:/extra/$package:ro"); done
  # Inside the worker image, on the host's network, with the settings of this cluster only:
  # the image has the PostgreSQL driver and the pipeline's own packages, and nothing of the
  # installation's service is read -- not its settings file, not its run database.
  docker run --rm --network host --user "$(id -u):$(id -g)" \
    --env-file "$home/private/control.env" \
    -e EFORECAST_KUBE_RUNS_FILE="$home/private/$target" -e EFORECAST_WORKSPACE="$workspace" \
    -e EFORECAST_USER="$(id -un)" \
    -v "$SHARED:$SHARED" "${mounts[@]}" \
    -v "$repo/docker:/opt/eforecast/docker:ro" -v "$repo/tests:/opt/eforecast/tests:ro" -v "$here:/driver:ro" \
    -w /opt/eforecast -e PYTHONPATH=/opt/eforecast:/extra:/driver \
    --entrypoint /opt/conda/bin/python "$image" "$@"
}

code_digests() {     # "<code in the image> <code of this checkout>"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$repo:/checkout:ro" \
    --entrypoint /opt/conda/bin/python "$image" -c "
from eforecast.config.kubernetes_runs import code_digest
print('code', code_digest('/opt/eforecast'), code_digest('/checkout'))" 2>/dev/null | sed -n 's/^code //p' | tail -1
}

target() {           # <file> <shared root> <workspace> <state root> <cpus> <ram mb> <seconds> <jobs> <image id>
  ( umask 077
    cat > "$home/private/$1" <<EOF
{
  "version": 1,
  "image": "docker.io/library/eforecast-worker@$9",
  "server": "https://127.0.0.1:$PORT",
  "namespace": "eforecast",
  "shared_root": "$2",
  "workspace": "$3",
  "state_root": "$4",
  "source_root": "/opt/eforecast",
  "python": "/opt/conda/bin/python",
  "service_account": "eforecast-worker",
  "connection_secret": "managed-connections",
  "resources": {"cpu_threads": $5, "ram_mb": $6, "gpu_slots": 0},
  "node_selector": {"kubernetes.io/hostname": "$NAME"},
  "pending_deadline": 180,
  "active_deadline": $7,
  "max_jobs": $8,
  "volume": {"hostPath": {"path": "$2", "type": "Directory"}}
}
EOF
  )
}

# ---------------------------------------------------------------- up
cmd_up() {
  installation
  local home="${EFORECAST_CLUSTER_HOME:-}" shared="" workspace="${EFORECAST_WORKSPACE:-}" name=$DEFAULT_NAME
  local port=$DEFAULT_PORT subnet=$DEFAULT_SUBNET cpus="" memory="" job_cpus="" job_ram="" job_hours=6 token_hours=72
  local python="${EFORECAST_PYTHON:-$(command -v python3 || true)}"
  while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || fail "$1 needs a value"
    case "$1" in
      --home) home="$2" ;;           --shared) shared="$2" ;;        --workspace) workspace="$2" ;;
      --name) name="$2" ;;           --port) port="$2" ;;            --subnet) subnet="$2" ;;
      --cpus) cpus="$2" ;;           --memory) memory="$2" ;;        --job-cpus) job_cpus="$2" ;;
      --job-memory-mb) job_ram="$2" ;; --job-hours) job_hours="$2" ;; --token-hours) token_hours="$2" ;;
      --python) python="$2" ;;
      *) fail "up does not take $1; see: $me help" ;;
    esac
    shift 2
  done
  [ -n "$shared" ] || fail "up needs --shared: the folder of this machine the Pods will see (workspace, data, weather and models lie under it)"
  [ -d "$shared" ] || fail "--shared is not a folder: $shared"
  shared=$(realpath "$shared"); [ "$shared" != / ] || fail "--shared cannot be /"
  [ -n "$home" ] || home="$shared/$name"
  home=$(realpath -m "$home")
  plain --shared "$shared"; plain --home "$home"
  [ "$home" != "$shared" ] || fail "--home is a folder of the cluster's own, under --shared; it cannot be --shared itself"
  case "$home/" in "$shared"/*) ;; *) fail "--home must lie under --shared ($shared): the Pods reach the cluster's own files there" ;; esac
  if [ -n "$workspace" ]; then
    workspace=$(realpath -m "$workspace"); plain --workspace "$workspace"
    [ -d "$workspace" ] || fail "--workspace is not a folder: $workspace"
    case "$workspace/" in "$shared"/*) ;; *) fail "the workspace $workspace is not under --shared ($shared): a Pod could not see it" ;; esac
    case "$home/" in "$workspace"/*) fail "--home may not lie inside the workspace" ;; esac
  fi
  case "$name" in ''|*[!a-z0-9-]*|-*) fail "--name may hold small letters, digits and '-' only: $name" ;; esac
  case "$subnet" in *[!0-9.]*|'') fail "--subnet is three numbers, like $DEFAULT_SUBNET: $subnet" ;; esac
  number --port "$port"; number --job-hours "$job_hours"; number --token-hours "$token_hours"
  [ -x "$python" ] || fail "no interpreter to borrow the API's packages from: --python <the one the web service runs>"
  local all half ram_gb
  all=$(nproc); half=$(( all > 3 ? all / 2 : 2 ))
  ram_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1024 / 1024 / 2}' /proc/meminfo); [ "$ram_gb" -ge 4 ] || ram_gb=4
  [ -n "$cpus" ] || cpus=$half
  [ -n "$memory" ] || memory="${ram_gb}g"
  number --cpus "$cpus"
  case "$memory" in *g) ;; *) fail "--memory is a number of gigabytes, like 24g: $memory" ;; esac
  number --memory "${memory%g}"
  [ -n "$job_cpus" ] || job_cpus=$(( cpus > 3 ? cpus - 2 : 1 ))
  [ -n "$job_ram" ] || job_ram=$(( ${memory%g} * 1024 * 2 / 3 ))
  number --job-cpus "$job_cpus"; number --job-memory-mb "$job_ram"
  command -v docker >/dev/null || fail "Docker is needed, and the right to use it without sudo"
  docker info >/dev/null 2>&1 || fail "Docker does not answer for this user (docker info)"
  [ ! -e "$home/cluster.env" ] || fail "$home already holds a cluster: EFORECAST_CLUSTER_HOME=$home $me start"
  if [ -d "$home" ] && [ -n "$(ls -A "$home")" ]; then fail "$home exists and is not empty; --home is a folder of the cluster's own"; fi
  if docker network inspect "$name" >/dev/null 2>&1; then fail "a Docker network named $name exists already; choose another --name"; fi
  local taken part
  taken=$(docker ps -a --format '{{.Names}}')
  for part in "$name" "$name-pg" "$name-rabbit"; do
    case $'\n'"$taken"$'\n' in *$'\n'"$part"$'\n'*) fail "a container named $part exists already; choose another --name" ;; esac
  done

  install -d -m 700 "$home" "$home/private"
  install -d "$home/integration/workspace" "$home/integration/execution-state" "$home/state" \
             "$home/pgdata" "$home/rabbitdata" "$home/k3s" "$home/kubelet"
  cat > "$home/cluster.env" <<EOF
# The local Kubernetes cluster of $me.
# Written by "up"; every other command reads it. Nothing secret is in this file.
NAME=$name
SHARED=$shared
WORKSPACE=$workspace
SUBNET=$subnet
PORT=$port
CPUS=$cpus
MEMORY=$memory
JOB_CPUS=$job_cpus
JOB_RAM_MB=$job_ram
JOB_SECONDS=$(( job_hours * 3600 ))
TOKEN_HOURS=$token_hours
PYTHON=$python
PACKAGES="${EFORECAST_CONTROL_PACKAGES:-$DEFAULT_PACKAGES}"
K3S_IMAGE=${EFORECAST_K3S_IMAGE:-$DEFAULT_K3S}
PG_IMAGE=${EFORECAST_PG_IMAGE:-$DEFAULT_PG}
RABBIT_IMAGE=${EFORECAST_RABBIT_IMAGE:-$DEFAULT_RABBIT}
EOF
  export EFORECAST_CLUSTER_HOME="$home"
  load
  site_packages >/dev/null

  # Passwords of this cluster's own two services, in files only their owner can read.
  local pg_password rabbit_password dsn
  pg_password=$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9'); pg_password=${pg_password:0:32}
  rabbit_password=$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9'); rabbit_password=${rabbit_password:0:32}
  dsn="postgresql://postgres:$pg_password@$SUBNET.10:5432/eforecast"
  ( umask 077
    printf 'POSTGRES_PASSWORD=%s\nPOSTGRES_DB=eforecast\n' "$pg_password" > "$home/private/postgres.env"
    printf 'RABBITMQ_DEFAULT_USER=eforecast\nRABBITMQ_DEFAULT_PASS=%s\n' "$rabbit_password" > "$home/private/rabbit.env"
    # what the Pods are given, as a Secret: where the run database is, and nothing else
    printf 'EFORECAST_RUN_STORE_DSN=%s\nEFORECAST_ALLOCATOR_DSN=%s\nEFORECAST_ALLOCATOR_NAMESPACE=local-cluster\n' \
      "$dsn" "$dsn" > "$home/private/managed-connections.env"
    # what the control plane (the API and the coordinator, one process) is given
    cat > "$home/private/control.env" <<EOF
EFORECAST_RUN_STORE=postgres
EFORECAST_RUN_STORE_DSN=$dsn
EFORECAST_RUN_STORE_NAMESPACE=local-cluster
EFORECAST_ALLOCATOR_BACKEND=postgres
EFORECAST_ALLOCATOR_DSN=$dsn
EFORECAST_ALLOCATOR_NAMESPACE=local-cluster
EFORECAST_BROKER_URL=amqp://eforecast:$rabbit_password@$SUBNET.11:5672/
EFORECAST_KUBE_SERVER=https://127.0.0.1:$PORT
EFORECAST_KUBE_NAMESPACE=eforecast
EFORECAST_KUBE_CA=$home/private/ca.crt
EFORECAST_KUBE_TOKEN_FILE=$home/private/executor.token
EFORECAST_AUTH=off
EFORECAST_RUN_COORDINATOR=off
HOME=/tmp
EOF
  )

  say "creating $NAME in $home (if this stops half way: $me down $NAME)"
  docker network create --subnet "$SUBNET.0/24" "$NAME" >/dev/null \
    || fail "the network $SUBNET.0/24 could not be created; choose another --subnet"
  docker run -d --name "$database" --network "$NAME" --ip "$SUBNET.10" \
    -v "$home/pgdata:/var/lib/postgresql/data" --env-file "$home/private/postgres.env" "$PG_IMAGE" >/dev/null
  docker run -d --name "$broker" --hostname "$NAME-rabbit" --network "$NAME" --ip "$SUBNET.11" \
    -v "$home/rabbitdata:/var/lib/rabbitmq" --env-file "$home/private/rabbit.env" "$RABBIT_IMAGE" >/dev/null
  # The node: k3s in one privileged container. The shared folder is mounted under its own
  # path, so that a path means the same file on the host, in the node and in a Pod.
  docker run -d --name "$node" --hostname "$NAME" --network "$NAME" --ip "$SUBNET.2" \
    --privileged --cpus="$CPUS" --memory="$MEMORY" -p "127.0.0.1:$PORT:6443" \
    -v "$SHARED:$SHARED" -v "$home/k3s:/var/lib/rancher/k3s" -v "$home/kubelet:/var/lib/kubelet" \
    "$K3S_IMAGE" server --disable=traefik --disable=servicelb --disable=metrics-server \
    --tls-san=127.0.0.1 --node-name="$NAME" >/dev/null
  say "  the database, the broker and the node are started; waiting for them"
  wait_services
  wait_node
  kube apply -f - < "$here/00-namespace.yaml" >/dev/null
  kube apply -f - < "$here/10-rbac.yaml" >/dev/null
  ( umask 077; docker exec "$node" cat /var/lib/rancher/k3s/server/tls/server-ca.crt > "$home/private/ca.crt" )
  issue_token
  kube -n eforecast create secret generic managed-connections --from-env-file=/dev/stdin \
    --dry-run=client -o yaml < "$home/private/managed-connections.env" | kube apply -f - >/dev/null
  say "  namespace eforecast, the executor and worker accounts and the connection secret are in place"
  say "the cluster $NAME is up: $CPUS CPUs, $MEMORY, API on 127.0.0.1:$PORT, no GPU"
  say "next:"
  say "  export EFORECAST_CLUSTER_HOME=$home"
  say "  $me image"
}

# ---------------------------------------------------------------- image
cmd_image() {
  load; need_running
  say "building the worker image of $repo (docker/Dockerfile.worker); the first build takes a while"
  # Without a provenance record the same code gives the same image, and an image the node
  # already has is not copied again. (A record of the build is dated, so every build with one
  # is a new image.) An older Docker does not know the flag, and needs none.
  if ! DOCKER_BUILDKIT=1 docker build -q --provenance=false -f "$repo/docker/Dockerfile.worker" -t "$image" "$repo" >/dev/null 2>&1; then
    DOCKER_BUILDKIT=1 docker build -q -f "$repo/docker/Dockerfile.worker" -t "$image" "$repo" >/dev/null
  fi
  local id reference in_image in_checkout old
  id=$(docker image inspect "$image" --format '{{.Id}}')
  read -r in_image in_checkout <<< "$(code_digests || true)"
  if [ -z "${in_image:-}" ] || [ "$in_image" != "${in_checkout:-}" ]; then
    fail "the image does not carry the code of this checkout (was a file changed while it was built?); run it again"
  fi
  reference="docker.io/library/eforecast-worker@$id"
  if ! docker exec "$node" crictl inspecti "$reference" >/dev/null 2>&1; then
    say "copying it into the node -- $(docker image inspect "$image" --format '{{.Size}}' | awk '{printf "%.1f GB", $1 / 1e9}'), a minute or several"
    docker save "$image" | docker exec -i "$node" ctr -n k8s.io images import - >/dev/null
    # A Job names its image by digest and may not pull: the copy gets the name the Job asks for.
    docker exec "$node" ctr -n k8s.io images tag --force "docker.io/library/$image" "$reference" >/dev/null
  fi
  if [ -z "$(kube -n eforecast get jobs --no-headers 2>/dev/null || echo unknown)" ]; then
    # nothing runs: the images of earlier code go, so that the node does not fill up with them
    for old in $(docker exec "$node" ctr -n k8s.io images ls -q | grep '^docker.io/library/eforecast-worker@' | grep -v -F "$reference" || true); do
      docker exec "$node" ctr -n k8s.io images rm "$old" >/dev/null 2>&1 || true
    done
  fi
  # The drill refuses a folder that is not named as what it is: it writes synthetic cases there.
  target target-drill.json "$home/integration" "$home/integration/workspace" "$home/integration/execution-state" 2 4096 1800 2 "$id"
  if [ -n "$WORKSPACE" ]; then
    target target-workspace.json "$SHARED" "$WORKSPACE" "$home/state" "$JOB_CPUS" "$JOB_RAM_MB" "$JOB_SECONDS" 1 "$id"
  fi
  for old in target-drill.json target-workspace.json; do
    [ -f "$home/private/$old" ] || continue
    control "$old" "$home/integration/workspace" -c "
from eforecast.config.kubernetes_runs import configured_target
configured_target()" || fail "the target file $old is not valid"
  done
  say "the node has image ${id:0:19}… with the code of this checkout (${in_image:0:12})"
  say "next:  $me drill"
}

# ---------------------------------------------------------------- status
cmd_status() {
  load
  local part state ready until in_image in_checkout jobs
  say "cluster    $NAME in $home"
  say "shared     $SHARED   (what a Pod sees of this machine)"
  say "workspace  ${WORKSPACE:-none: 'run' is not available, only the drill}"
  for part in "node $node" "database $database" "broker $broker"; do
    state=$(docker inspect -f '{{.State.Status}}' "${part#* }" 2>/dev/null || echo missing)
    printf '%-10s %s   (container %s)\n' "${part%% *}" "$state" "${part#* }"
  done
  if is_running "$node"; then
    ready=$(kube get nodes --no-headers 2>/dev/null | awk '{print $2 ", " $5}' || true)
    say "kubernetes ${ready:-does not answer yet}; $CPUS CPUs, $MEMORY, no GPU; API https://127.0.0.1:$PORT"
    jobs=$(kube -n eforecast get jobs --no-headers 2>/dev/null | wc -l || true)
    say "jobs       ${jobs:-0} in the namespace eforecast"
  fi
  until=$(cat "$home/private/executor.token.until" 2>/dev/null || echo unknown)
  say "token      valid until $until   ('$me token' issues another)"
  if docker image inspect "$image" >/dev/null 2>&1; then
    read -r in_image in_checkout <<< "$(code_digests || true)"
    if [ -n "${in_image:-}" ] && [ "$in_image" = "${in_checkout:-}" ]; then
      say "image      $image carries the code of this checkout"
    else
      say "image      $image carries OTHER code than this checkout: a Job would refuse it. Run: $me image"
    fi
  else
    say "image      none yet. Run: $me image"
  fi
}

# ---------------------------------------------------------------- the rest
cmd_drill() {
  load; need_running; need_image
  control target-drill.json "$home/integration/workspace" docker/smoke_managed_kubernetes.py \
    --scratch "$home/integration" --datasets "$@"
}

cmd_run() {
  load; need_running; need_image
  [ -n "$WORKSPACE" ] || fail "this cluster was created without --workspace, so it knows no cases; only the drill runs on it"
  [ -f "$home/private/target-workspace.json" ] || fail "no target file for the workspace yet: $me image"
  [ $# -ge 2 ] || fail "run CASE datasets|training|combine [--options JSON] [--seconds N]"
  control target-workspace.json "$WORKSPACE" /driver/run_case_step.py "$@"
}

cmd_kubectl() {
  load
  is_running "$node" || fail "the cluster $NAME is not running: $me start"
  if [ -t 0 ] && [ -t 1 ]; then docker exec -it "$node" kubectl "$@"; else docker exec -i "$node" kubectl "$@"; fi
}

cmd_token() {
  load; need_running
  issue_token
  say "a token for the executor account, valid until $(cat "$home/private/executor.token.until")"
}

cmd_stop() {
  load
  local jobs=0
  if is_running "$node"; then jobs=$(kube -n eforecast get jobs --no-headers 2>/dev/null | wc -l || true); fi
  if [ "$jobs" -gt 0 ] && [ "${1:-}" != --force ]; then
    fail "$jobs Job(s) are still on the cluster ($me kubectl -n eforecast get jobs,pods). Let them end, or: $me stop --force"
  fi
  docker stop "$node" "$broker" "$database" >/dev/null 2>&1 || true
  say "$NAME is stopped; its folder and databases are kept. Again: $me start"
}

cmd_start() {
  load
  docker start "$database" "$broker" "$node" >/dev/null || fail "the containers of $NAME could not be started (were they removed? $me down $NAME, then up)"
  wait_services
  wait_node
  issue_token
  say "$NAME is running; a token valid until $(cat "$home/private/executor.token.until")"
}

cmd_down() {
  load
  [ "${1:-}" = "$NAME" ] || fail "this removes the cluster $NAME for good -- its containers, its network, its databases and $home. To do it: $me down $NAME"
  docker rm -f "$node" "$broker" "$database" >/dev/null 2>&1 || true
  docker network rm "$NAME" >/dev/null 2>&1 || true
  # The node, the database and the broker wrote as users of their own: their folders are
  # removed from inside a container. Only what "up" created is touched.
  docker run --rm -v "$home:/cluster" --entrypoint /bin/sh "$PG_IMAGE" -c \
    'rm -rf /cluster/k3s /cluster/kubelet /cluster/pgdata /cluster/rabbitdata /cluster/integration /cluster/state' \
    || say "  some folders under $home could not be removed; they belong to root"
  rm -rf "$home/private" "$home/cluster.env"
  rmdir "$home" 2>/dev/null || say "  $home holds something this script did not put there, and is left"
  docker image rm "$image" >/dev/null 2>&1 || true
  say "$NAME is removed"
}

command="${1:-help}"
[ $# -gt 0 ] && shift
case "$command" in
  up) cmd_up "$@" ;;
  image) cmd_image "$@" ;;
  status) cmd_status "$@" ;;
  drill) cmd_drill "$@" ;;
  run) cmd_run "$@" ;;
  kubectl) cmd_kubectl "$@" ;;
  token) cmd_token "$@" ;;
  stop) cmd_stop "$@" ;;
  start) cmd_start "$@" ;;
  down) cmd_down "$@" ;;
  help|-h|--help) usage ;;
  *) usage >&2; fail "no such command: $command" ;;
esac
