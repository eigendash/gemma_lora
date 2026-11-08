#!/usr/bin/env bash
# Create the RTX PRO 6000 spot VM (same settings as tp-rtx6000 on 2026-11-07), ship the repo (git HEAD + train/eval
# data only; holdout files never leave the Mac), and start a chain script on it detached.
# usage: CHAIN_SCRIPT=path/to/chain_cmd.sh [VM=tp-rtx6000] [PROJECT=countdown-rl-16875] [SA=<project number>-compute@...]
#        [MAX_RUN=21h] [ZONES="us-central1-c us-central1-b"] bash scripts/gcp/create_gpu_vm.sh
# A VM in another project needs its service account granted roles/storage.objectAdmin on the checkpoint bucket.
# CHAIN_SCRIPT is a bash script run on the VM from ~/run (it should end with POWEROFF=1 / sudo poweroff).
# BUCKET is its GCS bucket: the prebuilt venv is restored from it BEFORE the chain starts (cloud_chain.sh reads configs
# with the venv python, so a missing venv silently skips the model fetch).
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CHAIN_SCRIPT:?}"; BUCKET="${BUCKET:?BUCKET=gs://... (the chain bucket, for the venv restore)}"
VM="${VM:-tp-rtx6000}"; ZONES="${ZONES:-us-central1-c us-central1-b}"
MAX_RUN="${MAX_RUN:-21h}"; PROJECT="${PROJECT:-countdown-rl-16875}"; SA="${SA:-516504598961-compute@developer.gserviceaccount.com}"
export CLOUDSDK_CORE_PROJECT="$PROJECT"
git diff --quiet HEAD -- || { echo "commit first: the VM gets git HEAD"; exit 1; }
STAGE="$(mktemp -d)"
git archive --format=tar HEAD | tar xf - -C "$STAGE"
mkdir -p "$STAGE/data" && cp data/train.jsonl data/eval.jsonl "$STAGE/data/"
ls "$STAGE/data" | grep -qi holdout && { echo "holdout file in the VM tarball: refusing"; exit 1; }
COPYFILE_DISABLE=1 tar czf "$STAGE.tgz" -C "$STAGE" .
ZONE=""
for z in $ZONES; do
  if gcloud compute instances create "$VM" --zone "$z" --machine-type g4-standard-48 \
      --provisioning-model SPOT --instance-termination-action DELETE --max-run-duration "${MAX_RUN:-21h}" \
      --maintenance-policy TERMINATE --no-restart-on-failure \
      --image-project deeplearning-platform-release --image common-cu129-ubuntu-2404-nvidia-580-v20261029 \
      --boot-disk-size 200GB --boot-disk-type hyperdisk-balanced \
      --metadata install-nvidia-driver=True \
      --service-account "$SA" \
      --scopes storage-rw,logging-write \
      --shielded-vtpm --shielded-integrity-monitoring; then ZONE="$z"; break; fi
  echo "create failed in $z; trying the next zone"
done
[ -n "$ZONE" ] || { echo "VM creation failed in every zone (capacity or error above)"; exit 2; }
echo "created $VM in $ZONE; waiting for ssh and the GPU driver"
for i in $(seq 1 40); do
  gcloud compute ssh "$VM" --zone "$ZONE" --command 'nvidia-smi -L' >/dev/null 2>&1 && break; sleep 30
done
gcloud compute scp "$STAGE.tgz" "$VM:code.tgz" --zone "$ZONE"
gcloud compute scp "$CHAIN_SCRIPT" "$VM:chain_cmd.sh" --zone "$ZONE"
gcloud compute ssh "$VM" --zone "$ZONE" --command \
  "mkdir -p ~/run && tar xzf ~/code.tgz -C ~/run && chmod +x ~/chain_cmd.sh && cd ~/run && GCS_BUCKET=$BUCKET bash scripts/cloud_restore_env.sh && .venv-vllm/bin/python -c 'import vllm'"
gcloud compute ssh "$VM" --zone "$ZONE" --command \
  '(setsid nohup bash ~/chain_cmd.sh > ~/chain.log 2>&1 < /dev/null &) && sleep 5 && echo started && nvidia-smi -L'
rm -rf "$STAGE" "$STAGE.tgz"
echo "ZONE=$ZONE"
