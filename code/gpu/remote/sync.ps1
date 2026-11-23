param(
  [Parameter(Mandatory = $true)][string]$HostName,
  [int]$Port = 22,
  [string]$User = 'root',
  [string]$Remote = '/root/autodl-tmp/mixcnn',
  [ValidateSet('up', 'down', 'check')][string]$Direction = 'up',
  [switch]$WithPreds
)

$ErrorActionPreference = 'Stop'
$Pkg = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
$Key = Join-Path $Pkg '..\.ssh\autodl_ed25519'
if (-not (Test-Path $Key)) { throw "SSH key not found: $Key" }

$target = "$User@$HostName"
$sshArgs = @('-p', $Port, '-i', $Key, '-o', 'StrictHostKeyChecking=accept-new')

if ($Direction -eq 'up') {
  $tmp = Join-Path $env:TEMP 'mixcnn_upload.tgz'
  if (Test-Path $tmp) { Remove-Item $tmp -Force }
  Push-Location $Pkg
  try {
    tar -czf $tmp `
      --exclude 'code/journal/.venv' `
      --exclude 'code/gpu/__pycache__' `
      --exclude 'code/__pycache__' `
      --exclude '*/__pycache__' `
      --exclude '*.pyc' `
      --exclude '*.pt' `
      --exclude 'results' `
      code data
  } finally { Pop-Location }
  & ssh @sshArgs $target "mkdir -p $Remote"
  & scp -P $Port -i $Key -o StrictHostKeyChecking=accept-new $tmp "${target}:$Remote/upload.tgz"
  & ssh @sshArgs $target "cd $Remote && tar -xzf upload.tgz && rm upload.tgz && mkdir -p results/gpu/{main,baselines,m1c,jnu,paderborn,seu,logs}"
  $mb = [math]::Round((Get-Item $tmp).Length / 1MB, 1)
  Write-Output "Uploaded code/ and data/ ($mb MB) to ${target}:$Remote"
}
elseif ($Direction -eq 'check') {
  & ssh @sshArgs $target @'
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
python - <<PY
import torch, platform
print('host', platform.node())
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('available', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('gpu', p.name, round(p.total_memory/1e9,1), 'GB', torch.cuda.get_device_capability(0))
PY
df -h /root/autodl-tmp | tail -1
'@
}
else {
  $ex = if ($WithPreds) { '' } else { "--exclude '*/preds'" }
  & ssh @sshArgs $target "cd $Remote && tar -czf results.tgz $ex results/gpu"
  $tmp = Join-Path $env:TEMP 'mixcnn_results.tgz'
  & scp -P $Port -i $Key -o StrictHostKeyChecking=accept-new "${target}:$Remote/results.tgz" $tmp
  $dest = $Pkg
  & tar -xzf $tmp -C $dest
  & ssh @sshArgs $target "rm -f $Remote/results.tgz"
  $mb = [math]::Round((Get-Item $tmp).Length / 1MB, 1)
  Write-Output "Downloaded results/gpu ($mb MB) to $dest"
}
