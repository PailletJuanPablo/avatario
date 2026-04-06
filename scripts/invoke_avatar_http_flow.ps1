[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8010",
    [string]$ApiToken = "dev-token",
    [ValidateSet("health", "build-template", "enqueue-audio", "job-status", "avatar-status")]
    [string]$Action,
    [string]$SessionId = "",
    [string]$TemplateName = "demo-local",
    [string]$SourceImage = "",
    [string]$SourceVideo = "",
    [string]$SourceFrame = "",
    [string]$Audio = "",
    [string]$SourceTemplatePack = "",
    [string]$JobId = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-AuthHeaders {
    param(
        [string]$Token,
        [string]$AvatarSessionId
    )

    $headers = @{
        Authorization = "Bearer $Token"
    }
    if (-not [string]::IsNullOrWhiteSpace($AvatarSessionId)) {
        $headers["X-Avatar-Session-Id"] = $AvatarSessionId
    }
    return $headers
}

function Resolve-SessionId {
    param(
        [string]$Token,
        [string]$Base
    )

    $health = Invoke-RestMethod -Uri "$Base/api/health" -Headers (New-AuthHeaders -Token $Token -AvatarSessionId "")
    return [string]$health.avatarSessionId
}

function Invoke-CurlJson {
    param(
        [string[]]$Arguments
    )

    $responseText = & curl.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "curl.exe failed with exit code $LASTEXITCODE"
    }
    if ([string]::IsNullOrWhiteSpace($responseText)) {
        return $null
    }
    return $responseText | ConvertFrom-Json
}

$normalizedBaseUrl = $BaseUrl.TrimEnd("/")
$resolvedSessionId = $SessionId
if ([string]::IsNullOrWhiteSpace($resolvedSessionId) -and $Action -ne "health") {
    $resolvedSessionId = Resolve-SessionId -Token $ApiToken -Base $normalizedBaseUrl
}

switch ($Action) {
    "health" {
        Invoke-RestMethod -Uri "$normalizedBaseUrl/api/health" -Headers (New-AuthHeaders -Token $ApiToken -AvatarSessionId "") | ConvertTo-Json -Depth 8
        break
    }

    "avatar-status" {
        Invoke-RestMethod -Uri "$normalizedBaseUrl/api/avatar/status" -Headers (New-AuthHeaders -Token $ApiToken -AvatarSessionId $resolvedSessionId) | ConvertTo-Json -Depth 8
        break
    }

    "job-status" {
        if ([string]::IsNullOrWhiteSpace($JobId)) {
            throw "JobId is required for -Action job-status"
        }
        Invoke-RestMethod -Uri "$normalizedBaseUrl/api/jobs/$JobId/status" -Headers (New-AuthHeaders -Token $ApiToken -AvatarSessionId $resolvedSessionId) | ConvertTo-Json -Depth 8
        break
    }

    "build-template" {
        if (-not [string]::IsNullOrWhiteSpace($SourceImage) -and -not (Test-Path $SourceImage -PathType Leaf)) {
            throw "SourceImage not found: $SourceImage"
        }
        if (-not [string]::IsNullOrWhiteSpace($SourceVideo) -and -not (Test-Path $SourceVideo -PathType Leaf)) {
            throw "SourceVideo not found: $SourceVideo"
        }
        if ([string]::IsNullOrWhiteSpace($SourceImage) -and [string]::IsNullOrWhiteSpace($SourceVideo) -and [string]::IsNullOrWhiteSpace($SourceFrame)) {
            throw "Provide SourceImage, SourceVideo or SourceFrame for -Action build-template"
        }

        $curlArgs = @(
            "-sS",
            "-X", "POST",
            "-H", "Authorization: Bearer $ApiToken",
            "-H", "X-Avatar-Session-Id: $resolvedSessionId",
            "-F", "template_name=$TemplateName"
        )

        if (-not [string]::IsNullOrWhiteSpace($SourceImage)) {
            $curlArgs += @("-F", "source_image=@$SourceImage")
        }
        elseif (-not [string]::IsNullOrWhiteSpace($SourceVideo)) {
            $curlArgs += @("-F", "source_video=@$SourceVideo")
        }
        else {
            $curlArgs += @("-F", "source_frame=$SourceFrame")
        }

        $curlArgs += "$normalizedBaseUrl/api/source-templates"
        Invoke-CurlJson -Arguments $curlArgs | ConvertTo-Json -Depth 8
        break
    }

    "enqueue-audio" {
        if ([string]::IsNullOrWhiteSpace($Audio)) {
            throw "Audio is required for -Action enqueue-audio"
        }
        if (-not (Test-Path $Audio -PathType Leaf)) {
            throw "Audio not found: $Audio"
        }
        if ([string]::IsNullOrWhiteSpace($SourceTemplatePack)) {
            throw "SourceTemplatePack is required for -Action enqueue-audio"
        }

        $curlArgs = @(
            "-sS",
            "-X", "POST",
            "-H", "Authorization: Bearer $ApiToken",
            "-H", "X-Avatar-Session-Id: $resolvedSessionId",
            "-F", "audio=@$Audio",
            "-F", "source_template_pack=$SourceTemplatePack",
            "$normalizedBaseUrl/api/avatar/enqueue"
        )
        Invoke-CurlJson -Arguments $curlArgs | ConvertTo-Json -Depth 8
        break
    }
}
