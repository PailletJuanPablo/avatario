const AVATAR_CLIENT_GLOBAL_NAME = "AvatarClientSDK";
const AVATAR_CLIENT_SESSION_PREFIX = "avatar_";
const AVATAR_CLIENT_BOOLEAN_TRUE = "true";
const AVATAR_CLIENT_BOOLEAN_FALSE = "false";
const AVATAR_CLIENT_TRAILING_SLASH_PATTERN = /\/+$/;
const AVATAR_CLIENT_VIDEO_SOURCE_NAME_PATTERN = /\.(mp4|avi|mov|mkv|webm|flv|m4v|wmv)$/i;

const AVATAR_CLIENT_HTTP_METHODS = Object.freeze({
  GET: "GET",
  POST: "POST",
});

const AVATAR_CLIENT_TRANSPORTS = Object.freeze({
  HTTP: "http",
});

const AVATAR_CLIENT_MEDIA_KINDS = Object.freeze({
  AUDIO: "audio",
  VIDEO: "video",
});

const AVATAR_CLIENT_MEDIA_DIRECTIONS = Object.freeze({
  RECEIVE_ONLY: "recvonly",
});

const AVATAR_CLIENT_HEADERS = Object.freeze({
  AUTHORIZATION: "Authorization",
  CONTENT_TYPE: "Content-Type",
  SESSION_ID: "X-Avatar-Session-Id",
});

const AVATAR_CLIENT_CONTENT_TYPES = Object.freeze({
  JSON: "application/json",
});

const AVATAR_CLIENT_AUTH = Object.freeze({
  TOKEN_QUERY_KEY: "token",
  BEARER_PREFIX: "Bearer ",
});

const AVATAR_CLIENT_QUERY_KEYS = Object.freeze({
  SESSION_ID: "sessionId",
  TOKEN: "token",
});

const AVATAR_CLIENT_ENDPOINTS = Object.freeze({
  HEALTH: "/api/health",
  AVATAR_STATUS: "/api/avatar/status",
  AVATAR_ENQUEUE: "/api/avatar/enqueue",
  AVATAR_VIDEO_HTTP: "/api/avatar/video.mp4",
  JOBS_PREFIX: "/api/jobs/",
  SOURCE_TEMPLATES: "/api/source-templates",
});

const AVATAR_CLIENT_SOURCE_FIELDS = Object.freeze({
  AUDIO: "audio",
  SOURCE_IMAGE: "source_image",
  SOURCE_VIDEO: "source_video",
  SOURCE_FRAME: "source_frame",
  SOURCE_TEMPLATE_PACK: "source_template_pack",
  TEMPLATE_NAME: "template_name",
  AUDIO_TUNING_PRESET: "audio_tuning_preset",
});

const AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES = Object.freeze({
  BOOLEAN: "boolean",
  INTEGER: "integer",
  NULLABLE_INTEGER: "nullable-integer",
  NUMBER: "number",
  STRING: "string",
});

const AVATAR_CLIENT_DEFAULTS = Object.freeze({
  API_BASE_URL: "http://127.0.0.1:8010",
  MODE: "preview",
  MOTION_STRIDE: 2,
  GENERATION_FRAME_COUNT: null,
  ANIMATION_REGION: "lip",
  STITCHING_ENABLED: true,
  RELATIVE_MOTION_ENABLED: true,
  PASTE_BACK_ENABLED: true,
  DRIVING_MULTIPLIER: 1,
  CFG_SCALE: 1.2,
  JOYVASA_INFERENCE_STEPS: 15,
  AUDIO_EYE_TAMED_PRESET: false,
  AUDIO_EYE_SOFT_FACTOR: 0.45,
  AUDIO_EYE_HARD_FACTOR: 0.18,
  AUDIO_EYE_HARD_DY_MIN: -0.0045,
  AUDIO_EYE_HARD_DY_MAX: 0.0035,
  AUDIO_MOTION_TUNING_ENABLED: false,
  AUDIO_REANCHOR_FIRST_N: 5,
  AUDIO_MOUTH_OPEN_FACTOR: 1.18,
  AUDIO_POSE_SMOOTH_WINDOW: 5,
  AUDIO_EXP_SMOOTH_WINDOW: 3,
  AUDIO_POSE_JUMP_THRESHOLD: 8,
  AUDIO_TRANSLATION_JUMP_THRESHOLD: 0.03,
  AUDIO_LIP_SYNC_ASSIST: false,
  AUDIO_LIP_SYNC_MIN_RATIO: 0.03,
  AUDIO_LIP_SYNC_MAX_RATIO: 0.32,
  AUDIO_LIP_SYNC_SMOOTH_WINDOW: 5,
  AUDIO_LIP_SYNC_STRENGTH: 1.15,
  AUDIO_LIP_SYNC_POWER: 0.85,
  AUDIO_LIP_SYNC_ATTACK: 1,
  AUDIO_LIP_SYNC_RELEASE: 1,
  AUDIO_LIP_SYNC_OFFSET_MS: 0,
  AUDIO_MOUTH_FLOOR_STRENGTH: 0.26,
  AUDIO_MOUTH_PEAK_CLAMP: 0,
  HTTP_STREAM_AUTOPLAY: true,
  HTTP_STREAM_MUTED: true,
  HTTP_STREAM_PLAYS_INLINE: true,
  STATUS_POLL_INTERVAL_MS: 400,
});

const AVATAR_CLIENT_UPLOAD_PARAMETER_SPECS = Object.freeze({
  mode: Object.freeze({
    fieldName: "mode",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.STRING,
    defaultValue: AVATAR_CLIENT_DEFAULTS.MODE,
  }),
  motionStride: Object.freeze({
    fieldName: "motion_stride",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.INTEGER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.MOTION_STRIDE,
  }),
  generationFrameCount: Object.freeze({
    fieldName: "generation_frame_count",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NULLABLE_INTEGER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.GENERATION_FRAME_COUNT,
  }),
  animationRegion: Object.freeze({
    fieldName: "animation_region",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.STRING,
    defaultValue: AVATAR_CLIENT_DEFAULTS.ANIMATION_REGION,
  }),
  stitching: Object.freeze({
    fieldName: "stitching",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.BOOLEAN,
    defaultValue: AVATAR_CLIENT_DEFAULTS.STITCHING_ENABLED,
  }),
  relativeMotion: Object.freeze({
    fieldName: "relative_motion",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.BOOLEAN,
    defaultValue: AVATAR_CLIENT_DEFAULTS.RELATIVE_MOTION_ENABLED,
  }),
  pasteBack: Object.freeze({
    fieldName: "paste_back",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.BOOLEAN,
    defaultValue: AVATAR_CLIENT_DEFAULTS.PASTE_BACK_ENABLED,
  }),
  drivingMultiplier: Object.freeze({
    fieldName: "driving_multiplier",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.DRIVING_MULTIPLIER,
  }),
  cfgScale: Object.freeze({
    fieldName: "cfg_scale",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.CFG_SCALE,
  }),
  joyvasaInferenceSteps: Object.freeze({
    fieldName: "joyvasa_inference_steps",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.INTEGER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.JOYVASA_INFERENCE_STEPS,
  }),
  audioEyeTamedPreset: Object.freeze({
    fieldName: "audio_eye_tamed_preset",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.BOOLEAN,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_EYE_TAMED_PRESET,
  }),
  audioEyeSoftFactor: Object.freeze({
    fieldName: "audio_eye_soft_factor",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_EYE_SOFT_FACTOR,
  }),
  audioEyeHardFactor: Object.freeze({
    fieldName: "audio_eye_hard_factor",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_EYE_HARD_FACTOR,
  }),
  audioEyeHardDyMin: Object.freeze({
    fieldName: "audio_eye_hard_dy_min",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_EYE_HARD_DY_MIN,
  }),
  audioEyeHardDyMax: Object.freeze({
    fieldName: "audio_eye_hard_dy_max",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_EYE_HARD_DY_MAX,
  }),
  audioMotionTuningEnabled: Object.freeze({
    fieldName: "audio_motion_tuning_enabled",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.BOOLEAN,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_MOTION_TUNING_ENABLED,
  }),
  audioReanchorFirstN: Object.freeze({
    fieldName: "audio_reanchor_first_n",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.INTEGER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_REANCHOR_FIRST_N,
  }),
  audioMouthOpenFactor: Object.freeze({
    fieldName: "audio_mouth_open_factor",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_MOUTH_OPEN_FACTOR,
  }),
  audioPoseSmoothWindow: Object.freeze({
    fieldName: "audio_pose_smooth_window",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.INTEGER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_POSE_SMOOTH_WINDOW,
  }),
  audioExpSmoothWindow: Object.freeze({
    fieldName: "audio_exp_smooth_window",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.INTEGER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_EXP_SMOOTH_WINDOW,
  }),
  audioPoseJumpThreshold: Object.freeze({
    fieldName: "audio_pose_jump_threshold",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_POSE_JUMP_THRESHOLD,
  }),
  audioTranslationJumpThreshold: Object.freeze({
    fieldName: "audio_translation_jump_threshold",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_TRANSLATION_JUMP_THRESHOLD,
  }),
  audioLipSyncAssist: Object.freeze({
    fieldName: "audio_lip_sync_assist",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.BOOLEAN,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_LIP_SYNC_ASSIST,
  }),
  audioLipSyncMinRatio: Object.freeze({
    fieldName: "audio_lip_sync_min_ratio",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_LIP_SYNC_MIN_RATIO,
  }),
  audioLipSyncMaxRatio: Object.freeze({
    fieldName: "audio_lip_sync_max_ratio",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_LIP_SYNC_MAX_RATIO,
  }),
  audioLipSyncSmoothWindow: Object.freeze({
    fieldName: "audio_lip_sync_smooth_window",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.INTEGER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_LIP_SYNC_SMOOTH_WINDOW,
  }),
  audioLipSyncStrength: Object.freeze({
    fieldName: "audio_lip_sync_strength",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_LIP_SYNC_STRENGTH,
  }),
  audioLipSyncPower: Object.freeze({
    fieldName: "audio_lip_sync_power",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_LIP_SYNC_POWER,
  }),
  audioLipSyncAttack: Object.freeze({
    fieldName: "audio_lip_sync_attack",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_LIP_SYNC_ATTACK,
  }),
  audioLipSyncRelease: Object.freeze({
    fieldName: "audio_lip_sync_release",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_LIP_SYNC_RELEASE,
  }),
  audioLipSyncOffsetMs: Object.freeze({
    fieldName: "audio_lip_sync_offset_ms",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.INTEGER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_LIP_SYNC_OFFSET_MS,
  }),
  audioMouthFloorStrength: Object.freeze({
    fieldName: "audio_mouth_floor_strength",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_MOUTH_FLOOR_STRENGTH,
  }),
  audioMouthPeakClamp: Object.freeze({
    fieldName: "audio_mouth_peak_clamp",
    type: AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER,
    defaultValue: AVATAR_CLIENT_DEFAULTS.AUDIO_MOUTH_PEAK_CLAMP,
  }),
});

const AVATAR_CLIENT_HEALTH_DEFAULT_MAPPINGS = Object.freeze({
  mode: "defaultMode",
  motionStride: "defaultAudioMotionStride",
  animationRegion: "defaultAnimationRegion",
  stitching: "defaultStitchingEnabled",
  relativeMotion: "defaultRelativeMotionEnabled",
  pasteBack: "defaultPasteBackEnabled",
  drivingMultiplier: "defaultDrivingMultiplier",
  cfgScale: "defaultCfgScale",
  joyvasaInferenceSteps: "defaultJoyvasaInferenceSteps",
  audioEyeTamedPreset: "defaultAudioEyeTamedPreset",
  audioEyeSoftFactor: "defaultAudioEyeSoftFactor",
  audioEyeHardFactor: "defaultAudioEyeHardFactor",
  audioEyeHardDyMin: "defaultAudioEyeHardDyMin",
  audioEyeHardDyMax: "defaultAudioEyeHardDyMax",
  audioMotionTuningEnabled: "defaultAudioMotionTuningEnabled",
  audioReanchorFirstN: "defaultAudioReanchorFirstN",
  audioMouthOpenFactor: "defaultAudioMouthOpenFactor",
  audioPoseSmoothWindow: "defaultAudioPoseSmoothWindow",
  audioExpSmoothWindow: "defaultAudioExpSmoothWindow",
  audioPoseJumpThreshold: "defaultAudioPoseJumpThreshold",
  audioTranslationJumpThreshold: "defaultAudioTranslationJumpThreshold",
  audioLipSyncAssist: "defaultAudioLipSyncAssist",
  audioLipSyncMinRatio: "defaultAudioLipSyncMinRatio",
  audioLipSyncMaxRatio: "defaultAudioLipSyncMaxRatio",
  audioLipSyncSmoothWindow: "defaultAudioLipSyncSmoothWindow",
  audioLipSyncStrength: "defaultAudioLipSyncStrength",
  audioLipSyncPower: "defaultAudioLipSyncPower",
  audioLipSyncAttack: "defaultAudioLipSyncAttack",
  audioLipSyncRelease: "defaultAudioLipSyncRelease",
  audioLipSyncOffsetMs: "defaultAudioLipSyncOffsetMs",
  audioMouthFloorStrength: "defaultAudioMouthFloorStrength",
  audioMouthPeakClamp: "defaultAudioMouthPeakClamp",
});

const AVATAR_CLIENT_ERROR_CODES = Object.freeze({
  INVALID_CONFIGURATION: "INVALID_CONFIGURATION",
  MISSING_FETCH: "MISSING_FETCH",
  MISSING_FORM_DATA: "MISSING_FORM_DATA",
  MISSING_AUDIO_INPUT: "MISSING_AUDIO_INPUT",
  MISSING_SOURCE_INPUT: "MISSING_SOURCE_INPUT",
  MISSING_SOURCE_TEMPLATE_PACK: "MISSING_SOURCE_TEMPLATE_PACK",
  MISSING_SESSION: "MISSING_SESSION",
  INVALID_MEDIA_TARGET: "INVALID_MEDIA_TARGET",
  INVALID_TRANSPORT: "INVALID_TRANSPORT",
  HTTP_STREAM_UNSUPPORTED: "HTTP_STREAM_UNSUPPORTED",
  HTTP_REQUEST_FAILED: "HTTP_REQUEST_FAILED",
  LEGACY_API_DISABLED: "LEGACY_API_DISABLED",
});

const AVATAR_CLIENT_ERROR_MESSAGES = Object.freeze({
  INVALID_CONFIGURATION: "Avatar client requires a valid apiBaseUrl or avatarUrl.",
  MISSING_FETCH: "A fetch implementation is required in this environment.",
  MISSING_FORM_DATA: "A FormData implementation is required in this environment.",
  MISSING_AUDIO_INPUT: "Audio input is required to enqueue avatar speech.",
  MISSING_SOURCE_INPUT: "Provide sourceImage, sourceVideo, or sourceFrame.",
  MISSING_SOURCE_TEMPLATE_PACK: "sourceTemplatePack is required to enqueue avatar speech.",
  MISSING_SESSION: "Create or restore a session before streaming or sending audio.",
  INVALID_MEDIA_TARGET: "The provided media target does not look like an HTMLMediaElement.",
  INVALID_TRANSPORT: "Unsupported avatar transport.",
  HTTP_STREAM_UNSUPPORTED: "This browser does not support the HTTP MP4 avatar stream.",
  HTTP_REQUEST_FAILED: "Avatar API request failed.",
  LEGACY_API_DISABLED: "Legacy realtime APIs are disabled. Use template packs plus the HTTP MP4 stream.",
});

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function normalizeOptionalString(value) {
  return isNonEmptyString(value) ? value.trim() : "";
}

function normalizeStreamTransport(transport) {
  const normalizedTransport = normalizeOptionalString(transport).toLowerCase();
  if (!normalizedTransport) {
    return "";
  }
  return normalizedTransport;
}

function resolveFirstDefinedValue(values) {
  for (const value of values) {
    if (value !== undefined && value !== null) {
      return value;
    }
  }
  return undefined;
}

function isObjectLike(value) {
  return typeof value === "object" && value !== null;
}

class AvatarClientError extends Error {
  /**
   * Create a typed avatar client error.
   *
   * @param {string} code - Stable error code.
   * @param {object} [details] - Structured error details.
   * @param {unknown} [cause] - Original error cause.
   */
  constructor(code, details = {}, cause) {
    super(AVATAR_CLIENT_ERROR_MESSAGES[code] || AVATAR_CLIENT_ERROR_MESSAGES.HTTP_REQUEST_FAILED);
    this.name = "AvatarClientError";
    this.code = code;
    this.details = details;
    if (cause !== undefined) {
      this.cause = cause;
    }
  }
}

function createAvatarClientError(code, details = {}, cause) {
  return new AvatarClientError(code, details, cause);
}

function ensureFetchImplementation(fetchImplementation) {
  const resolvedFetch = fetchImplementation || globalThis.fetch;
  if (typeof resolvedFetch !== "function") {
    throw createAvatarClientError(AVATAR_CLIENT_ERROR_CODES.MISSING_FETCH);
  }
  return resolvedFetch.bind(globalThis);
}

function ensureFormDataImplementation(formDataConstructor) {
  const resolvedFormData = formDataConstructor || globalThis.FormData;
  if (typeof resolvedFormData !== "function") {
    throw createAvatarClientError(AVATAR_CLIENT_ERROR_CODES.MISSING_FORM_DATA);
  }
  return resolvedFormData;
}

function resolveApiBaseUrl(apiBaseUrl) {
  const normalizedApiBaseUrl = normalizeOptionalString(apiBaseUrl);
  if (normalizedApiBaseUrl) {
    return normalizedApiBaseUrl.replace(AVATAR_CLIENT_TRAILING_SLASH_PATTERN, "");
  }
  if (isObjectLike(globalThis.location) && isNonEmptyString(globalThis.location.origin)) {
    return String(globalThis.location.origin).replace(AVATAR_CLIENT_TRAILING_SLASH_PATTERN, "");
  }
  throw createAvatarClientError(AVATAR_CLIENT_ERROR_CODES.INVALID_CONFIGURATION);
}

function buildAbsoluteUrl(apiBaseUrl, pathOrUrl) {
  if (!isNonEmptyString(pathOrUrl)) {
    return apiBaseUrl;
  }
  if (/^https?:\/\//i.test(pathOrUrl)) {
    return pathOrUrl;
  }
  return new URL(pathOrUrl, `${apiBaseUrl}/`).toString();
}

function buildJobStatusPath(jobId) {
  return `${AVATAR_CLIENT_ENDPOINTS.JOBS_PREFIX}${encodeURIComponent(jobId)}/status`;
}

function createSessionIdentifier() {
  if (isObjectLike(globalThis.crypto) && typeof globalThis.crypto.randomUUID === "function") {
    return `${AVATAR_CLIENT_SESSION_PREFIX}${globalThis.crypto.randomUUID().replace(/-/g, "")}`;
  }
  return `${AVATAR_CLIENT_SESSION_PREFIX}${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 12)}`;
}

function buildAuthorizationValue(token) {
  const normalizedToken = normalizeOptionalString(token);
  return normalizedToken ? `${AVATAR_CLIENT_AUTH.BEARER_PREFIX}${normalizedToken}` : "";
}

function coerceBooleanValue(value, fallbackValue) {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalizedValue = value.trim().toLowerCase();
    if (normalizedValue === AVATAR_CLIENT_BOOLEAN_TRUE) {
      return true;
    }
    if (normalizedValue === AVATAR_CLIENT_BOOLEAN_FALSE) {
      return false;
    }
  }
  return fallbackValue;
}

function coerceNumberValue(value, fallbackValue) {
  const parsedValue = Number(value);
  return Number.isFinite(parsedValue) ? parsedValue : fallbackValue;
}

function coerceIntegerValue(value, fallbackValue) {
  const parsedValue = Number(value);
  return Number.isFinite(parsedValue) ? Math.round(parsedValue) : fallbackValue;
}

function normalizeUploadParameterValue(specification, value, fallbackValue) {
  if (!isObjectLike(specification)) {
    return value;
  }
  if (value === undefined) {
    return fallbackValue;
  }
  switch (specification.type) {
    case AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.BOOLEAN:
      return coerceBooleanValue(value, fallbackValue);
    case AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.INTEGER:
      return coerceIntegerValue(value, fallbackValue);
    case AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NULLABLE_INTEGER:
      return value === null || value === "" ? null : coerceIntegerValue(value, fallbackValue);
    case AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.NUMBER:
      return coerceNumberValue(value, fallbackValue);
    case AVATAR_CLIENT_UPLOAD_PARAMETER_TYPES.STRING:
      return normalizeOptionalString(value) || fallbackValue;
    default:
      return value;
  }
}

function readServerDefaultUploadParameters(healthPayload) {
  const defaults = {};
  const normalizedHealthPayload = isObjectLike(healthPayload) ? healthPayload : {};
  for (const [parameterName, healthKey] of Object.entries(AVATAR_CLIENT_HEALTH_DEFAULT_MAPPINGS)) {
    const specification = AVATAR_CLIENT_UPLOAD_PARAMETER_SPECS[parameterName];
    if (!specification) {
      continue;
    }
    defaults[parameterName] = normalizeUploadParameterValue(
      specification,
      normalizedHealthPayload[healthKey],
      specification.defaultValue
    );
  }
  return defaults;
}

function normalizeUploadParameters(healthPayload, inputParameters = {}) {
  const normalizedInputParameters = isObjectLike(inputParameters) ? inputParameters : {};
  const serverDefaults = readServerDefaultUploadParameters(healthPayload);
  const resolvedParameters = {};
  for (const [parameterName, specification] of Object.entries(AVATAR_CLIENT_UPLOAD_PARAMETER_SPECS)) {
    const fallbackValue = resolveFirstDefinedValue([
      serverDefaults[parameterName],
      specification.defaultValue,
    ]);
    resolvedParameters[parameterName] = normalizeUploadParameterValue(
      specification,
      normalizedInputParameters[parameterName],
      fallbackValue
    );
  }
  return resolvedParameters;
}

function normalizeBinaryName(binaryValue, fallbackName) {
  if (isObjectLike(binaryValue) && isNonEmptyString(binaryValue.name)) {
    return String(binaryValue.name);
  }
  return fallbackName;
}

function isProbablyVideoBinary(binaryValue) {
  if (!isObjectLike(binaryValue)) {
    return false;
  }
  const normalizedType = normalizeOptionalString(binaryValue.type).toLowerCase();
  if (normalizedType.startsWith(`${AVATAR_CLIENT_MEDIA_KINDS.VIDEO}/`)) {
    return true;
  }
  const normalizedName = normalizeBinaryName(binaryValue, "");
  return AVATAR_CLIENT_VIDEO_SOURCE_NAME_PATTERN.test(normalizedName);
}

function appendBinaryToFormData(formData, fieldName, binaryValue, fallbackName) {
  if (!binaryValue) {
    return;
  }
  const resolvedName = normalizeBinaryName(binaryValue, fallbackName);
  formData.append(fieldName, binaryValue, resolvedName);
}

function appendScalarToFormData(formData, fieldName, value) {
  if (value === undefined || value === null || value === "") {
    return;
  }
  if (typeof value === "boolean") {
    formData.append(fieldName, value ? AVATAR_CLIENT_BOOLEAN_TRUE : AVATAR_CLIENT_BOOLEAN_FALSE);
    return;
  }
  formData.append(fieldName, String(value));
}

function isMediaElementLike(target) {
  return (
    isObjectLike(target)
    && typeof target.load === "function"
    && typeof target.pause === "function"
    && "src" in target
  );
}

/**
 * Lightweight frontend client for the realtime avatar API.
 */
class AvatarClient {
  /**
   * Create a new avatar client instance.
   *
   * @param {object} options - Client configuration.
   * @param {string} [options.apiBaseUrl] - Avatar API base URL.
   * @param {string} [options.avatarUrl] - Alias for apiBaseUrl.
   * @param {string} [options.token] - Optional API bearer token.
   * @param {Function} [options.fetchImplementation] - Custom fetch implementation.
   * @param {Function} [options.formDataConstructor] - Custom FormData constructor.
   */
  constructor(options = {}) {
    this.apiBaseUrl = resolveApiBaseUrl(resolveFirstDefinedValue([options.apiBaseUrl, options.avatarUrl]));
    this.token = normalizeOptionalString(options.token);
    this.fetchImplementation = ensureFetchImplementation(options.fetchImplementation);
    this.formDataConstructor = ensureFormDataImplementation(options.formDataConstructor);
    this.sessionId = normalizeOptionalString(options.sessionId);
    this.health = null;
    this.lastAvatarStatus = null;
    this.lastJobStatus = null;
    this.activePeerConnection = null;
    this.activeHttpStreamController = null;
    this.activeVideoElement = null;
  }

  /**
   * Replace the API token used by subsequent requests.
   *
   * @param {string} token - New bearer token.
   */
  setToken(token) {
    this.token = normalizeOptionalString(token);
  }

  /**
   * Replace the current avatar session identifier.
   *
   * @param {string} sessionId - Existing avatar session identifier.
   */
  setSessionId(sessionId) {
    this.sessionId = normalizeOptionalString(sessionId);
  }

  /**
   * Return the current avatar session identifier.
   *
   * @returns {string} Current avatar session identifier or an empty string.
   */
  getSessionId() {
    return this.sessionId;
  }

  /**
   * Return the last cached health payload from the backend.
   *
   * @returns {object | null} Last health payload.
   */
  getHealth() {
    return this.health;
  }

  /**
   * Return the last cached avatar status payload.
   *
   * @returns {object | null} Last avatar status payload.
   */
  getLastAvatarStatus() {
    return this.lastAvatarStatus;
  }

  /**
   * Return the last cached job status payload.
   *
   * @returns {object | null} Last job status payload.
   */
  getLastJobStatus() {
    return this.lastJobStatus;
  }

  /**
   * Request backend health and session metadata.
   *
   * @param {object} [options] - Request options.
   * @param {string} [options.sessionId] - Optional avatar session identifier to reuse.
   * @returns {Promise<object>} Backend health payload.
   * @throws {Error} When the request fails.
   */
  async fetchHealth(options = {}) {
    const sessionId = normalizeOptionalString(options.sessionId);
    const payload = await this.requestJson({
      method: AVATAR_CLIENT_HTTP_METHODS.GET,
      path: AVATAR_CLIENT_ENDPOINTS.HEALTH,
      sessionId,
      cache: "no-store",
    });
    this.health = payload;
    return payload;
  }

  /**
   * Create or resume an avatar session.
   *
   * @param {object} [options] - Session options.
   * @param {string} [options.sessionId] - Existing avatar session identifier to resume.
   * @returns {Promise<object>} Session payload including sessionId and health metadata.
   * @throws {Error} When the backend request fails.
   */
  async createSession(options = {}) {
    const requestedSessionId = normalizeOptionalString(options.sessionId) || this.sessionId;
    const healthPayload = await this.fetchHealth({ sessionId: requestedSessionId });
    const resolvedSessionId = normalizeOptionalString(healthPayload.avatarSessionId)
      || requestedSessionId
      || createSessionIdentifier();
    this.sessionId = resolvedSessionId;
    return {
      sessionId: resolvedSessionId,
      health: healthPayload,
      httpStreamUrl: this.buildHttpStreamUrl({ sessionId: resolvedSessionId, useHealthPath: true }),
    };
  }

  /**
   * Alias for createSession to keep the public API concise.
   *
   * @param {object} [options] - Session options.
   * @returns {Promise<object>} Session payload.
   */
  async connect(options = {}) {
    return this.createSession(options);
  }

  /**
   * Fetch the current avatar scheduler status for the active session.
   *
   * @param {object} [options] - Request options.
   * @param {string} [options.sessionId] - Session identifier override.
   * @returns {Promise<object>} Avatar status payload.
   * @throws {Error} When the request fails.
   */
  async fetchAvatarStatus(options = {}) {
    const sessionId = this.requireSessionId(options.sessionId);
    const payload = await this.requestJson({
      method: AVATAR_CLIENT_HTTP_METHODS.GET,
      path: AVATAR_CLIENT_ENDPOINTS.AVATAR_STATUS,
      sessionId,
      cache: "no-store",
    });
    this.lastAvatarStatus = payload;
    return payload;
  }

  /**
   * Fetch status for a specific generated audio job.
   *
   * @param {string} jobId - Avatar job identifier.
   * @param {object} [options] - Request options.
   * @param {string} [options.sessionId] - Session identifier override.
   * @returns {Promise<object>} Job status payload.
   * @throws {Error} When the request fails.
   */
  async fetchJobStatus(jobId, options = {}) {
    const normalizedJobId = normalizeOptionalString(jobId);
    const sessionId = this.requireSessionId(options.sessionId);
    const payload = await this.requestJson({
      method: AVATAR_CLIENT_HTTP_METHODS.GET,
      path: buildJobStatusPath(normalizedJobId),
      sessionId,
      cache: "no-store",
    });
    this.lastJobStatus = payload;
    return payload;
  }

  /**
   * Poll avatar status on a fixed interval.
   *
   * @param {object} options - Polling options.
   * @param {Function} options.onUpdate - Callback for each successful payload.
   * @param {Function} [options.onError] - Callback for request failures.
   * @param {number} [options.intervalMs] - Poll interval in milliseconds.
   * @param {string} [options.sessionId] - Session identifier override.
   * @returns {Function} Stop function.
   */
  startAvatarStatusPolling(options) {
    const normalizedOptions = isObjectLike(options) ? options : {};
    const onUpdate = typeof normalizedOptions.onUpdate === "function" ? normalizedOptions.onUpdate : () => {};
    const onError = typeof normalizedOptions.onError === "function" ? normalizedOptions.onError : () => {};
    const intervalMs = coerceIntegerValue(
      normalizedOptions.intervalMs,
      AVATAR_CLIENT_DEFAULTS.STATUS_POLL_INTERVAL_MS
    );
    const intervalHandle = globalThis.setInterval(async () => {
      try {
        const payload = await this.fetchAvatarStatus({ sessionId: normalizedOptions.sessionId });
        onUpdate(payload);
      } catch (error) {
        onError(error);
      }
    }, intervalMs);
    return () => {
      globalThis.clearInterval(intervalHandle);
    };
  }

  /**
   * Build the continuous MP4 stream URL for the current session.
   *
   * @param {object} [options] - URL options.
   * @param {string} [options.sessionId] - Session identifier override.
   * @param {boolean} [options.useHealthPath] - Use the path exposed by the health payload when available.
   * @returns {string} Absolute HTTP stream URL.
   */
  buildHttpStreamUrl(options = {}) {
    const sessionId = this.requireSessionId(options.sessionId);
    const path = options.useHealthPath && isObjectLike(this.health) && isNonEmptyString(this.health.avatarVideoHttpUrl)
      ? this.health.avatarVideoHttpUrl
      : AVATAR_CLIENT_ENDPOINTS.AVATAR_VIDEO_HTTP;
    return this.buildMediaUrl(path, sessionId);
  }

  /**
   * Build standard API headers with bearer token and avatar session identifier.
   *
   * @param {object} [options] - Header options.
   * @param {string} [options.sessionId] - Optional avatar session identifier.
   * @param {string} [options.contentType] - Optional content type.
   * @returns {Headers} Prepared headers object.
   */
  buildRequestHeaders(options = {}) {
    const normalizedOptions = isObjectLike(options) ? options : {};
    const headers = new Headers();
    const authorizationValue = buildAuthorizationValue(this.token);
    if (authorizationValue) {
      headers.set(AVATAR_CLIENT_HEADERS.AUTHORIZATION, authorizationValue);
    }
    const sessionId = normalizeOptionalString(normalizedOptions.sessionId);
    if (sessionId) {
      headers.set(AVATAR_CLIENT_HEADERS.SESSION_ID, sessionId);
    }
    const contentType = normalizeOptionalString(normalizedOptions.contentType);
    if (contentType) {
      headers.set(AVATAR_CLIENT_HEADERS.CONTENT_TYPE, contentType);
    }
    return headers;
  }

  /**
   * Stop the currently active managed HTTP stream, if any.
   */
  stopManagedHttpStream() {
    if (isObjectLike(this.activeHttpStreamController) && typeof this.activeHttpStreamController.stop === "function") {
      this.activeHttpStreamController.stop();
    }
    this.activeHttpStreamController = null;
  }

  /**
   * Attach the continuous HTTP stream to a video element.
   *
   * @param {object} options - Stream options.
   * @param {HTMLVideoElement} options.videoElement - Target video element.
   * @param {string} [options.sessionId] - Session identifier override.
   * @param {boolean} [options.autoplay] - Start playback immediately.
   * @param {boolean} [options.muted] - Set the muted property on the video element.
   * @param {boolean} [options.playsInline] - Set the playsInline property on the video element.
   * @returns {Promise<object>} Stream metadata.
   * @throws {Error} When the target element is invalid.
   */
  async connectHttpStream(options) {
    const normalizedOptions = isObjectLike(options) ? options : {};
    const videoElement = normalizedOptions.videoElement;
    if (!isMediaElementLike(videoElement)) {
      throw createAvatarClientError(AVATAR_CLIENT_ERROR_CODES.INVALID_MEDIA_TARGET);
    }
    const sessionId = this.requireSessionId(normalizedOptions.sessionId);
    this.stopManagedHttpStream();
    const streamUrl = this.buildHttpStreamUrl({ sessionId, useHealthPath: true });
    videoElement.muted = coerceBooleanValue(normalizedOptions.muted, AVATAR_CLIENT_DEFAULTS.HTTP_STREAM_MUTED);
    videoElement.playsInline = coerceBooleanValue(
      normalizedOptions.playsInline,
      AVATAR_CLIENT_DEFAULTS.HTTP_STREAM_PLAYS_INLINE
    );
    videoElement.autoplay = coerceBooleanValue(
      normalizedOptions.autoplay,
      AVATAR_CLIENT_DEFAULTS.HTTP_STREAM_AUTOPLAY
    );
    videoElement.srcObject = null;
    videoElement.src = streamUrl;
    videoElement.load();
    let playbackError = null;
    const stop = () => {
      videoElement.pause();
      videoElement.removeAttribute("src");
      videoElement.srcObject = null;
      videoElement.load();
    };
    this.activeHttpStreamController = {
      stop,
      streamUrl,
    };
    if (videoElement.autoplay) {
      try {
        await videoElement.play();
      } catch (error) {
        playbackError = error;
      }
    }
    this.activeVideoElement = videoElement;
    return {
      transport: AVATAR_CLIENT_TRANSPORTS.HTTP,
      sessionId,
      streamUrl,
      playbackError,
    };
  }

  /**
   * Start live playback using the unified HTTP MP4 transport.
   *
   * @param {object} options - Stream options.
   * @param {string} [options.transport] - Transport identifier.
   * @returns {Promise<object>} Stream metadata.
   * @throws {Error} When the transport is invalid.
   */
  async startStream(options = {}) {
    const normalizedOptions = isObjectLike(options) ? options : {};
    const requestedTransport = normalizeOptionalString(normalizedOptions.transport)
      || normalizeOptionalString(this.health && this.health.avatarTransport)
      || AVATAR_CLIENT_TRANSPORTS.HTTP;
    const transport = normalizeStreamTransport(requestedTransport);
    if (transport === AVATAR_CLIENT_TRANSPORTS.HTTP) {
      return this.connectHttpStream(normalizedOptions);
    }
    throw createAvatarClientError(AVATAR_CLIENT_ERROR_CODES.LEGACY_API_DISABLED, {
      api: "startStream",
      transport: requestedTransport,
    });
  }

  /**
   * List source template packs available on the backend.
   *
   * @returns {Promise<object>} Template pack listing payload.
   * @throws {Error} When the request fails.
   */
  async listSourceTemplates() {
    return this.requestJson({
      method: AVATAR_CLIENT_HTTP_METHODS.GET,
      path: this.resolveSourceTemplatesUrl(),
    });
  }

  /**
   * Build a source template pack from an uploaded source image, source video, or existing source frame.
   *
   * @param {object} options - Template creation options.
   * @param {Blob | File} [options.sourceImage] - Source image file.
   * @param {Blob | File} [options.sourceVideo] - Source video file.
   * @param {string} [options.sourceFrame] - Existing backend source frame path.
   * @param {string} [options.templateName] - Optional template pack name.
   * @returns {Promise<object>} Template creation payload.
   * @throws {Error} When no source input is provided.
   */
  async createSourceTemplate(options = {}) {
    const normalizedOptions = isObjectLike(options) ? options : {};
    const formData = new this.formDataConstructor();
    const sourceImage = normalizedOptions.sourceImage || null;
    const sourceVideo = normalizedOptions.sourceVideo || null;
    const sourceFrame = normalizeOptionalString(normalizedOptions.sourceFrame);
    if (!sourceImage && !sourceVideo && !sourceFrame) {
      throw createAvatarClientError(AVATAR_CLIENT_ERROR_CODES.MISSING_SOURCE_INPUT);
    }
    appendBinaryToFormData(formData, AVATAR_CLIENT_SOURCE_FIELDS.SOURCE_IMAGE, sourceImage, "source-image");
    appendBinaryToFormData(formData, AVATAR_CLIENT_SOURCE_FIELDS.SOURCE_VIDEO, sourceVideo, "source-video");
    appendScalarToFormData(formData, AVATAR_CLIENT_SOURCE_FIELDS.SOURCE_FRAME, sourceFrame);
    appendScalarToFormData(
      formData,
      AVATAR_CLIENT_SOURCE_FIELDS.TEMPLATE_NAME,
      normalizeOptionalString(normalizedOptions.templateName)
    );
    return this.requestJson({
      method: AVATAR_CLIENT_HTTP_METHODS.POST,
      path: this.resolveSourceTemplatesUrl(),
      body: formData,
    });
  }

  /**
   * Enqueue an audio clip for live avatar playback.
   *
   * @param {object} options - Audio upload options.
   * @param {Blob | File} options.audio - Audio binary.
   * @param {string} [options.sessionId] - Session identifier override.
   * @param {string} [options.template] - Alias for sourceTemplatePack.
   * @param {string} [options.sourceTemplatePack] - Source template pack identifier.
   * @param {string} [options.audioTuningPreset] - Backend audio tuning preset identifier.
   * @param {object} [options.params] - Parameter overrides for generation.
   * @returns {Promise<object>} Job payload returned by the backend.
   * @throws {Error} When audio is missing or the request fails.
   */
  async uploadAudio(options = {}) {
    const normalizedOptions = isObjectLike(options) ? options : {};
    const sessionId = this.requireSessionId(normalizedOptions.sessionId);
    const audioBinary = normalizedOptions.audio || null;
    if (!audioBinary) {
      throw createAvatarClientError(AVATAR_CLIENT_ERROR_CODES.MISSING_AUDIO_INPUT);
    }
    const formData = new this.formDataConstructor();
    appendBinaryToFormData(formData, AVATAR_CLIENT_SOURCE_FIELDS.AUDIO, audioBinary, "avatar-audio");

    const sourceTemplatePack = normalizeOptionalString(
      resolveFirstDefinedValue([normalizedOptions.sourceTemplatePack, normalizedOptions.template])
    );
    if (!sourceTemplatePack) {
      throw createAvatarClientError(AVATAR_CLIENT_ERROR_CODES.MISSING_SOURCE_TEMPLATE_PACK);
    }
    const audioTuningPreset = normalizeOptionalString(normalizedOptions.audioTuningPreset);
    appendScalarToFormData(formData, AVATAR_CLIENT_SOURCE_FIELDS.SOURCE_TEMPLATE_PACK, sourceTemplatePack);

    if (audioTuningPreset) {
      appendScalarToFormData(formData, AVATAR_CLIENT_SOURCE_FIELDS.AUDIO_TUNING_PRESET, audioTuningPreset);
    }

    const normalizedParameters = normalizeUploadParameters(this.health, normalizedOptions.params);
    for (const [parameterName, specification] of Object.entries(AVATAR_CLIENT_UPLOAD_PARAMETER_SPECS)) {
      appendScalarToFormData(formData, specification.fieldName, normalizedParameters[parameterName]);
    }

    const payload = await this.requestJson({
      method: AVATAR_CLIENT_HTTP_METHODS.POST,
      path: this.resolveEnqueueAudioUrl(),
      sessionId,
      body: formData,
    });
    this.lastJobStatus = payload;
    return payload;
  }

  /**
   * Close active browser-managed media state created by this instance.
   */
  dispose() {
    this.stopManagedHttpStream();
    if (this.activePeerConnection && typeof this.activePeerConnection.close === "function") {
      this.activePeerConnection.close();
    }
    this.activePeerConnection = null;
    if (isMediaElementLike(this.activeVideoElement)) {
      this.activeVideoElement.pause();
      this.activeVideoElement.removeAttribute("src");
      this.activeVideoElement.srcObject = null;
      this.activeVideoElement.load();
    }
    this.activeVideoElement = null;
  }

  /**
   * Resolve the enqueue endpoint exposed by the backend.
   *
   * @returns {string} Relative or absolute enqueue URL.
   */
  resolveEnqueueAudioUrl() {
    if (isObjectLike(this.health) && isNonEmptyString(this.health.enqueueAudioUrl)) {
      return this.health.enqueueAudioUrl;
    }
    return AVATAR_CLIENT_ENDPOINTS.AVATAR_ENQUEUE;
  }

  /**
   * Resolve the source template endpoint exposed by the backend.
   *
   * @returns {string} Relative or absolute source template URL.
   */
  resolveSourceTemplatesUrl() {
    if (isObjectLike(this.health) && isNonEmptyString(this.health.sourceTemplatePacksUrl)) {
      return this.health.sourceTemplatePacksUrl;
    }
    return AVATAR_CLIENT_ENDPOINTS.SOURCE_TEMPLATES;
  }

  /**
   * Build a media URL with token and session query parameters.
   *
   * @param {string} pathOrUrl - Relative or absolute media URL.
   * @param {string} sessionId - Avatar session identifier.
   * @returns {string} Absolute media URL.
   */
  buildMediaUrl(pathOrUrl, sessionId) {
    const mediaUrl = new URL(buildAbsoluteUrl(this.apiBaseUrl, pathOrUrl));
    if (this.token) {
      mediaUrl.searchParams.set(AVATAR_CLIENT_QUERY_KEYS.TOKEN, this.token);
    }
    if (sessionId) {
      mediaUrl.searchParams.set(AVATAR_CLIENT_QUERY_KEYS.SESSION_ID, sessionId);
    }
    return mediaUrl.toString();
  }

  /**
   * Ensure the client has a session identifier.
   *
   * @param {string} [sessionIdOverride] - Optional explicit session identifier.
   * @returns {string} Resolved session identifier.
   * @throws {Error} When the session identifier is missing.
   */
  requireSessionId(sessionIdOverride) {
    const sessionId = normalizeOptionalString(sessionIdOverride) || this.sessionId;
    if (!sessionId) {
      throw createAvatarClientError(AVATAR_CLIENT_ERROR_CODES.MISSING_SESSION);
    }
    return sessionId;
  }

  /**
   * Perform a JSON-capable HTTP request against the avatar API.
   *
   * @param {object} options - Request options.
   * @param {string} options.method - HTTP method.
   * @param {string} options.path - Relative or absolute path.
   * @param {string} [options.sessionId] - Session identifier override.
   * @param {object | FormData} [options.body] - Request body.
   * @param {string} [options.contentType] - Explicit content type.
   * @param {string} [options.cache] - Fetch cache mode.
   * @returns {Promise<object>} Parsed JSON payload.
   * @throws {Error} When the request fails.
   */
  async requestJson(options) {
    const normalizedOptions = isObjectLike(options) ? options : {};
    const sessionId = normalizeOptionalString(normalizedOptions.sessionId);
    const headers = this.buildRequestHeaders({
      sessionId,
      contentType: normalizedOptions.contentType === AVATAR_CLIENT_CONTENT_TYPES.JSON
        ? AVATAR_CLIENT_CONTENT_TYPES.JSON
        : "",
    });

    let requestBody = normalizedOptions.body;
    const isFormDataBody = requestBody instanceof this.formDataConstructor;
    if (requestBody && !isFormDataBody && normalizedOptions.contentType === AVATAR_CLIENT_CONTENT_TYPES.JSON) {
      requestBody = JSON.stringify(requestBody);
    }

    const response = await this.fetchImplementation(buildAbsoluteUrl(this.apiBaseUrl, normalizedOptions.path), {
      method: normalizedOptions.method || AVATAR_CLIENT_HTTP_METHODS.GET,
      headers,
      body: requestBody,
      cache: normalizedOptions.cache,
    });
    if (!response.ok) {
      const responseText = await response.text();
      throw createAvatarClientError(
        AVATAR_CLIENT_ERROR_CODES.HTTP_REQUEST_FAILED,
        {
          method: normalizedOptions.method || AVATAR_CLIENT_HTTP_METHODS.GET,
          path: normalizedOptions.path,
          status: response.status,
          body: responseText,
        }
      );
    }
    return response.json();
  }
}

/**
 * Factory helper for concise frontend integration.
 *
 * @param {object} options - Avatar client configuration.
 * @returns {AvatarClient} Avatar client instance.
 */
function createAvatarClient(options = {}) {
  return new AvatarClient(options);
}

const avatarClientModule = Object.freeze({
  AVATAR_CLIENT_CONSTANTS: Object.freeze({
    GLOBAL_NAME: AVATAR_CLIENT_GLOBAL_NAME,
    DEFAULTS: AVATAR_CLIENT_DEFAULTS,
    ENDPOINTS: AVATAR_CLIENT_ENDPOINTS,
    ERROR_CODES: AVATAR_CLIENT_ERROR_CODES,
    HEADERS: AVATAR_CLIENT_HEADERS,
    QUERY_KEYS: AVATAR_CLIENT_QUERY_KEYS,
    TRANSPORTS: AVATAR_CLIENT_TRANSPORTS,
  }),
  AvatarClient,
  AvatarClientError,
  createAvatarClient,
});

export {
  avatarClientModule as default,
  AVATAR_CLIENT_DEFAULTS,
  AVATAR_CLIENT_ENDPOINTS,
  AVATAR_CLIENT_ERROR_CODES,
  AVATAR_CLIENT_HEADERS,
  AVATAR_CLIENT_QUERY_KEYS,
  AVATAR_CLIENT_TRANSPORTS,
  AvatarClient,
  AvatarClientError,
  createAvatarClient,
};
