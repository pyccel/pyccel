#!bash
# https://github.com/chocolatey/choco/issues/1851
# https://gist.github.com/jayvdb/1daf8c60e20d64024f51ec333f5ce806

function refreshenv
{
  powershell -NonInteractive - <<\EOF
Import-Module "$env:ChocolateyInstall\helpers\chocolateyProfile.psm1"

Update-SessionEnvironment

# Round brackets in variable names cause problems with bash
Get-ChildItem env:* | %{
  if (!($_.Name.Contains('('))) {
    $value = $_.Value
    if ($_.Name -eq 'PATH') {
      $value = $value -replace ';',':'
    }
    Write-Output ("export " + $_.Name + "='" + $value + "'")
  }
} | Out-File -Encoding ascii $env:TEMP\refreshenv.sh

EOF

  source "$TEMP/refreshenv.sh"
}

alias RefreshEnv=refreshenv
