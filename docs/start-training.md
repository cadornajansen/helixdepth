# Start both training runs

Only the user executes remote commands. The assistant has not started the Pod.
The user replaced the unavailable old Pod on October 2. Use the new Pod's direct
TCP SSH address/port below (from the supplied screenshot); recheck if restarted.
Use fresh setup_runpod.sh, not setup_main.sh, because the old environment is gone.
The existing verified v2 archive remains usable; no rebuild is required.

## 1. Windows PowerShell: upload

```powershell
$podHost = '213.173.99.46'
$podPort = 20000
$bundle = 'C:\Development\personal\helixdepth\artifacts\runpod_main_v2\helixdepth-runpod.tar.gz'
scp -P $podPort -o PasswordAuthentication=no -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\id_ed25519" $bundle "$bundle.sha256" "root@${podHost}:/workspace/"
ssh -p $podPort -o PasswordAuthentication=no -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\id_ed25519" "root@$podHost"
```

## 2. Pod terminal: verify and prepare

```bash
cd /workspace
sha256sum -c helixdepth-runpod.tar.gz.sha256 && mkdir helixdepth-main-v2 && tar -xzf helixdepth-runpod.tar.gz -C helixdepth-main-v2 && cd helixdepth-main-v2 && bash scripts/setup_runpod.sh
```

Stop if this prints any error. setup_runpod.sh creates a fresh Linux environment,
installs the pinned dependencies and verifies Python, PyTorch, CUDA, package
checksums and tests. Its final message says Setup complete and mentions profiling;
for this main experiment continue with the launcher below instead of profiling.
It does not install evaluation extras or start training.

## 3. Pod terminal: launch and watch

```bash
nohup bash scripts/train_both.sh > artifacts/main.log 2>&1 < /dev/null &
tail -f artifacts/main.log
```

The launcher runs full-model GPU recovery checks before both main runs. It stops
on failure. A baseline step100/12206 line means main training is advancing;
recovery's step4/4 messages are not the main training run. Ctrl+C exits tail only;
the background job continues. SSH disconnection or laptop sleep does not stop
the remote job. The assistant does not need to remain connected.

There is a three-hour time cap per main run. It saves a recovery checkpoint and
exits on reaching that cap. It does NOT stop GPU billing. GPU runs can take longer
than the previous nondeterministic profiling estimates. Keep the existing Pod;
do not terminate it before downloading backups. Stop it when done to end GPU
charges (storage charges continue). Training only; evaluation is separate.

## Check later

```bash
cd /workspace/helixdepth-main-v2
tail -n 20 artifacts/main.log
cat artifacts/main_exit_code
```

No exit-code file means still running or a process/system failure; inspect the
log/processes. Exit code0 plus BOTH MODELS FINISHED means both runs succeeded.
Any other code means inspect the log before restarting. Never rerun the launcher
over existing outputs. Resume individual variants as in docs/main-experiment.md.

## Download both completed runs (Windows PowerShell)

Reuse podHost/podPort from step1. This only copies the two named main runs.

```powershell
$backup = 'C:\Development\personal\helixdepth\artifacts\main_backups'
New-Item -ItemType Directory -Force $backup | Out-Null
scp -r -P $podPort -o PasswordAuthentication=no -o IdentitiesOnly=yes -i "$env:USERPROFILE\.ssh\id_ed25519" "root@${podHost}:/workspace/helixdepth-main-v2/artifacts/main_baseline" "root@${podHost}:/workspace/helixdepth-main-v2/artifacts/main_helixdepth" $backup
.venv/Scripts/python.exe scripts/verify_transfer.py --root "$backup/main_baseline" --manifest "$backup/main_baseline/metrics.json"
.venv/Scripts/python.exe scripts/verify_transfer.py --root "$backup/main_helixdepth" --manifest "$backup/main_helixdepth/metrics.json"
```
