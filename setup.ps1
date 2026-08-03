
New-Variable -Name "PROJECT_NAME" -Value "ai-factory"

Write-Output "================================="
Write-Output "Beginning setup of $PROJECT_NAME"
Write-Output "================================="

$expectedDirectory = Get-Item -Path "$PSScriptRoot"

try {

    if (-Not $expectedDirectory.Name -eq "$PROJECT_NAME") {
        throw "The required root directory $PROJECT_NAME was not found!"
    }

    Write-Output "Found correct directory. Continuing..."

    $envFile = if (Test-Path -Path "$expectedDirectory\environment.yml") {
        "$expectedDirectory\environment.yml"
    } elseif (Test-Path -Path "$expectedDirectory\environment.yaml") {
        "$expectedDirectory\environment.yaml"
    } else {
        $null
    }
    if (-Not $envFile) {
        throw "environment.yml or environment.yaml not found in $expectedDirectory. Please create or install it."
    }

    if (Get-Command "conda" -ErrorAction SilentlyContinue) {
        $condaVersion = conda --version
        Write-Output "Found conda installation with version $condaVersion. Continuing..."
    } else {
        throw "Conda was not found. Please install anaconda or miniconda and try again."
    }

}
catch {
    Write-Warning "Script Stopped: $_"
}

# In case python local venv is active
deactivate

# Create conda environment (--file works with newer Miniconda when -f does not)
$envFileName = Split-Path -Leaf $envFile
Push-Location $expectedDirectory.FullName
try {
    conda env create --file $envFileName
    if ($LASTEXITCODE -ne 0) {
        conda env create -f $envFileName
    }
    if ($LASTEXITCODE -ne 0) {
        conda env create $envFileName
    }
} finally {
    Pop-Location
}

if ($LASTEXITCODE -eq 0) {
    conda activate ai-factory
} else {
    throw "Conda environment creation resulted in an error."
}
