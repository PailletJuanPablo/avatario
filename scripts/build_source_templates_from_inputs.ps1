[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8010",
    [string]$ApiToken = "dev-token",
    [string]$InputRoot = "E:\animation\inputs",
    [string]$TemplateOutputRoot = "E:\animation\output_fasterliveportrait\source_template_packs",
    [string]$SummaryOutputRoot = "E:\animation\output_fasterliveportrait\source_template_packs\_batch_logs",
    [bool]$Recurse = $true,
    [int]$MaxCount = 0,
    [switch]$Force,
    [switch]$StopOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$supportedVideoExtensions = @(".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".wmv", ".flv")

function Resolve-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    return [System.IO.Path]::GetFullPath($PathValue)
}

function New-AuthHeaders {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Token
    )

    return @{
        Authorization = "Bearer $Token"
    }
}

function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $resolvedBasePath = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\', '/')
    $resolvedTargetPath = [System.IO.Path]::GetFullPath($TargetPath)
    $basePrefix = "$resolvedBasePath\"

    if ($resolvedTargetPath.StartsWith($basePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $resolvedTargetPath.Substring($basePrefix.Length)
    }

    if ($resolvedTargetPath.Equals($resolvedBasePath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return "."
    }

    throw "Target path is not inside base path. base=$resolvedBasePath target=$resolvedTargetPath"
}

function Resolve-TemplateName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InputRootPath,
        [Parameter(Mandatory = $true)]
        [string]$VideoPath
    )

    $relativePath = Get-RelativePathCompat -BasePath $InputRootPath -TargetPath $VideoPath
    $relativeWithoutExtension = [System.IO.Path]::ChangeExtension($relativePath, $null)
    $normalizedValue = $relativeWithoutExtension -replace "[\\/]+", "__"
    $normalizedValue = $normalizedValue -replace "[^A-Za-z0-9._-]", "_"
    $normalizedValue = $normalizedValue.Trim("_", ".", " ")
    if ([string]::IsNullOrWhiteSpace($normalizedValue)) {
        throw "Unable to derive a template name from source path: $VideoPath"
    }
    return $normalizedValue
}

function Get-VideoInputs {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootPath,
        [switch]$IncludeRecursion
    )

    $searchOptions = @{
        File = $true
        Path = $RootPath
    }
    if ($IncludeRecursion) {
        $searchOptions["Recurse"] = $true
    }

    return Get-ChildItem @searchOptions |
        Where-Object { $supportedVideoExtensions -contains $_.Extension.ToLowerInvariant() } |
        Sort-Object FullName
}

function Invoke-TemplateBuild {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Base,
        [Parameter(Mandatory = $true)]
        [string]$Token,
        [Parameter(Mandatory = $true)]
        [string]$VideoPath,
        [Parameter(Mandatory = $true)]
        [string]$TemplateName
    )

    $responseFile = [System.IO.Path]::GetTempFileName()
    try {
        $curlArguments = @(
            "-sS",
            "-X", "POST",
            "-H", "Authorization: Bearer $Token",
            "-F", "template_name=$TemplateName",
            "-F", "source_video=@$VideoPath",
            "-o", $responseFile,
            "-w", "%{http_code}",
            "$($Base.TrimEnd('/'))/api/source-templates"
        )

        $httpStatus = (& curl.exe @curlArguments).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe failed with exit code $LASTEXITCODE for video $VideoPath"
        }
        if ($httpStatus -match "(\d{3})$") {
            $httpStatus = $Matches[1]
        }

        $responseText = Get-Content -Path $responseFile -Raw
        if ([string]::IsNullOrWhiteSpace($httpStatus)) {
            throw "Missing HTTP status while building template for $VideoPath"
        }

        if ([string]::IsNullOrWhiteSpace($responseText)) {
            throw "Empty response while building template for $VideoPath"
        }

        $responsePayload = $null
        try {
            $responsePayload = $responseText | ConvertFrom-Json
        }
        catch {
            $responsePayload = $null
        }
        if (($null -ne $responsePayload) -and ($null -ne $responsePayload.item)) {
            return $responsePayload
        }

        if ($httpStatus -notmatch "^2\\d\\d$") {
            $errorDetail = $responseText
            if (($null -ne $responsePayload) -and $null -ne $responsePayload.detail -and -not [string]::IsNullOrWhiteSpace([string]$responsePayload.detail)) {
                $errorDetail = [string]$responsePayload.detail
            }
            throw "HTTP $httpStatus while building template for $VideoPath. $errorDetail"
        }

        if ($null -eq $responsePayload -or $null -eq $responsePayload.item) {
            throw "Template build for $VideoPath succeeded but response did not contain item. body=$responseText"
        }
        return $responsePayload
    }
    finally {
        Remove-Item -LiteralPath $responseFile -Force -ErrorAction SilentlyContinue
    }
}

function Test-TemplateIsReadyVideoBacked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TemplatePath
    )

    if (-not (Test-Path $TemplatePath -PathType Leaf)) {
        return $false
    }

    $metaPath = "$TemplatePath.json"
    if (-not (Test-Path $metaPath -PathType Leaf)) {
        return $false
    }

    try {
        $meta = Get-Content -Path $metaPath -Raw | ConvertFrom-Json
    }
    catch {
        return $false
    }

    return (($null -ne $meta) -and [bool]$meta.is_source_video)
}

function Remove-TemplateArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TemplatePath
    )

    $candidatePaths = @(
        $TemplatePath,
        "$TemplatePath.json",
        ([System.IO.Path]::ChangeExtension($TemplatePath, ".preview.png"))
    )
    foreach ($candidatePath in $candidatePaths) {
        if (Test-Path $candidatePath -PathType Leaf) {
            Remove-Item -LiteralPath $candidatePath -Force
        }
    }
}

$resolvedInputRoot = Resolve-AbsolutePath -PathValue $InputRoot
$resolvedTemplateOutputRoot = Resolve-AbsolutePath -PathValue $TemplateOutputRoot
$resolvedSummaryOutputRoot = Resolve-AbsolutePath -PathValue $SummaryOutputRoot

if (-not (Test-Path $resolvedInputRoot -PathType Container)) {
    throw "InputRoot not found: $resolvedInputRoot"
}

New-Item -ItemType Directory -Path $resolvedTemplateOutputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $resolvedSummaryOutputRoot -Force | Out-Null

$healthPayload = Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/api/health" -Headers (New-AuthHeaders -Token $ApiToken)
$videoInputs = Get-VideoInputs -RootPath $resolvedInputRoot -IncludeRecursion:$Recurse
if ($MaxCount -gt 0) {
    $videoInputs = @($videoInputs | Select-Object -First $MaxCount)
}

if (-not $videoInputs -or $videoInputs.Count -eq 0) {
    throw "No supported video files found under: $resolvedInputRoot"
}

$batchStartedAt = Get-Date
$results = New-Object System.Collections.Generic.List[object]

Write-Host "[template-batch] api: $($BaseUrl.TrimEnd('/'))"
Write-Host "[template-batch] input root: $resolvedInputRoot"
Write-Host "[template-batch] recurse: $Recurse"
Write-Host "[template-batch] max count: $MaxCount"
Write-Host "[template-batch] videos found: $($videoInputs.Count)"
Write-Host "[template-batch] runner python: $($healthPayload.runnerPython)"

foreach ($videoFile in $videoInputs) {
    $templateName = Resolve-TemplateName -InputRootPath $resolvedInputRoot -VideoPath $videoFile.FullName
    $templateFileName = "$templateName.pkl"
    $templatePath = Join-Path $resolvedTemplateOutputRoot $templateFileName
    $relativePath = Get-RelativePathCompat -BasePath $resolvedInputRoot -TargetPath $videoFile.FullName

    if ((-not $Force) -and (Test-TemplateIsReadyVideoBacked -TemplatePath $templatePath)) {
        Write-Host "[template-batch] skip existing template=$templateFileName source=$relativePath"
        $results.Add([pscustomobject]@{
            templateId = $templateFileName
            source = $videoFile.FullName
            relativeSource = $relativePath
            status = "skipped"
            detail = "already exists"
            templatePath = $templatePath
        }) | Out-Null
        continue
    }

    if ((-not $Force) -and (Test-Path $templatePath -PathType Leaf)) {
        Write-Host "[template-batch] rebuild legacy template=$templateFileName source=$relativePath"
        Remove-TemplateArtifacts -TemplatePath $templatePath
    }

    Write-Host "[template-batch] build template=$templateFileName source=$relativePath"
    try {
        $response = Invoke-TemplateBuild -Base $BaseUrl -Token $ApiToken -VideoPath $videoFile.FullName -TemplateName $templateName
        $item = $response.item
        $results.Add([pscustomobject]@{
            templateId = [string]($item.id)
            source = $videoFile.FullName
            relativeSource = $relativePath
            status = "created"
            detail = [string]($item.templatePackPath)
            templatePath = [string]($item.templatePackPath)
            frameTotal = $item.frameTotal
            sourceType = [string]($item.sourceType)
            sourceFps = $item.sourceFps
        }) | Out-Null
    }
    catch {
        $message = $_.Exception.Message
        Write-Warning "[template-batch] failed template=$templateFileName source=$relativePath error=$message"
        $results.Add([pscustomobject]@{
            templateId = $templateFileName
            source = $videoFile.FullName
            relativeSource = $relativePath
            status = "failed"
            detail = $message
            templatePath = $templatePath
        }) | Out-Null
        if ($StopOnError) {
            break
        }
    }
}

$summaryTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$summaryPath = Join-Path $resolvedSummaryOutputRoot "template_batch_$summaryTimestamp.json"
$createdItems = @($results | Where-Object { $_.status -eq "created" })
$skippedItems = @($results | Where-Object { $_.status -eq "skipped" })
$failedItems = @($results | Where-Object { $_.status -eq "failed" })
$summaryPayload = [ordered]@{}
$summaryPayload["startedAt"] = $batchStartedAt.ToString("o")
$summaryPayload["finishedAt"] = (Get-Date).ToString("o")
$summaryPayload["baseUrl"] = $BaseUrl.TrimEnd('/')
$summaryPayload["inputRoot"] = $resolvedInputRoot
$summaryPayload["recurse"] = $Recurse
$summaryPayload["maxCount"] = $MaxCount
$summaryPayload["force"] = $Force.IsPresent
$summaryPayload["total"] = $results.Count
$summaryPayload["created"] = $createdItems.Count
$summaryPayload["skipped"] = $skippedItems.Count
$summaryPayload["failed"] = $failedItems.Count
$summaryPayload["items"] = $results.ToArray()

$summaryPayload | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host "[template-batch] summary: $summaryPath"
Write-Host "[template-batch] created: $($summaryPayload.created) skipped: $($summaryPayload.skipped) failed: $($summaryPayload.failed)"

$summaryPayload | ConvertTo-Json -Depth 8
