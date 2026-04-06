import assert from "node:assert/strict";
import { File } from "node:buffer";
import { once } from "node:events";
import http from "node:http";
import test from "node:test";
import { Readable } from "node:stream";

import { AVATAR_CLIENT_ERROR_CODES, createAvatarClient } from "../dist/avatar-client.esm.js";

const MOCK_VALUES = Object.freeze({
  API_TOKEN: "test-token",
  AUDIO_PRESET_ID: "energetic",
  CREATED_TEMPLATE_NAME: "marketing-avatar",
  DEFAULT_SESSION_ID: "mock-session",
  JOB_TOTAL_FRAMES: 24,
  SEEDED_TEMPLATE_ID: "template-seeded",
  SEEDED_TEMPLATE_NAME: "Template Seeded",
});

const MOCK_TIMINGS_MS = Object.freeze({
  COMPLETE: 460,
  PROCESSING_START: 80,
  STREAM_CHUNK_INTERVAL: 35,
});

function delay(milliseconds) {
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

function createMockVideoElement() {
  return {
    autoplay: false,
    currentTime: 0,
    buffered: {
      length: 0,
      end() {
        return 0;
      },
      start() {
        return 0;
      },
    },
    loadCount: 0,
    muted: false,
    pauseCount: 0,
    playCount: 0,
    playbackRate: 1,
    playsInline: false,
    src: "",
    srcObject: null,
    load() {
      this.loadCount += 1;
    },
    pause() {
      this.pauseCount += 1;
    },
    async play() {
      this.playCount += 1;
    },
    removeAttribute(attributeName) {
      if (attributeName === "src") {
        this.src = "";
      }
    },
  };
}

function createJsonResponse(response, payload, statusCode = 200) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json",
  });
  response.end(JSON.stringify(payload));
}

function createErrorResponse(response, statusCode, detail) {
  createJsonResponse(response, { detail }, statusCode);
}

function createMp4Box(type, payloadText) {
  const payloadBuffer = Buffer.from(payloadText, "utf8");
  const boxBuffer = Buffer.alloc(8 + payloadBuffer.length);
  boxBuffer.writeUInt32BE(boxBuffer.length, 0);
  boxBuffer.write(type, 4, 4, "ascii");
  payloadBuffer.copy(boxBuffer, 8);
  return boxBuffer;
}

function parseMp4Boxes(buffer) {
  const boxes = [];
  let offset = 0;
  while (offset + 8 <= buffer.length) {
    const boxSize = buffer.readUInt32BE(offset);
    if (boxSize < 8 || offset + boxSize > buffer.length) {
      break;
    }
    boxes.push({
      payload: buffer.subarray(offset + 8, offset + boxSize),
      size: boxSize,
      type: buffer.subarray(offset + 4, offset + 8).toString("ascii"),
    });
    offset += boxSize;
  }
  return boxes;
}

async function readMp4Boxes(response, minimumBoxCount) {
  assert.ok(response.body, "stream response must expose a body");
  const reader = response.body.getReader();
  const chunks = [];
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      if (value && value.length > 0) {
        chunks.push(Buffer.from(value));
        const combinedBuffer = Buffer.concat(chunks);
        const boxes = parseMp4Boxes(combinedBuffer);
        if (boxes.length >= minimumBoxCount) {
          return {
            boxes,
            combinedBuffer,
          };
        }
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
  const combinedBuffer = Buffer.concat(chunks);
  return {
    boxes: parseMp4Boxes(combinedBuffer),
    combinedBuffer,
  };
}

async function parseRequestBody(request) {
  if (request.method === "GET" || request.method === "HEAD") {
    return null;
  }
  const normalizedHeaders = Object.fromEntries(
    Object.entries(request.headers).map(([headerName, headerValue]) => [
      headerName,
      Array.isArray(headerValue) ? headerValue.join(", ") : String(headerValue || ""),
    ])
  );
  const normalizedRequest = new Request(`http://127.0.0.1${request.url}`, {
    body: Readable.toWeb(request),
    duplex: "half",
    headers: normalizedHeaders,
    method: request.method,
  });
  const contentType = String(request.headers["content-type"] || "");
  if (contentType.startsWith("multipart/form-data")) {
    const formData = await normalizedRequest.formData();
    const entries = {};
    for (const [fieldName, fieldValue] of formData.entries()) {
      if (typeof fieldValue === "string") {
        entries[fieldName] = fieldValue;
      } else {
        entries[fieldName] = {
          name: fieldValue.name,
          size: fieldValue.size,
          type: fieldValue.type,
        };
      }
    }
    return entries;
  }
  if (contentType.includes("application/json")) {
    return normalizedRequest.json();
  }
  return normalizedRequest.text();
}

function normalizeBearerToken(request) {
  const authorizationHeader = String(request.headers.authorization || "").trim();
  if (!authorizationHeader.toLowerCase().startsWith("bearer ")) {
    return "";
  }
  return authorizationHeader.slice(7).trim();
}

function resolveSessionId(request, requestUrl) {
  return String(
    request.headers["x-avatar-session-id"]
      || requestUrl.searchParams.get("sessionId")
      || MOCK_VALUES.DEFAULT_SESSION_ID
  ).trim();
}

function buildTemplateRecord({ createdAtMs, id, name, sourceType }) {
  return {
    createdAtMs,
    fileName: `${id}.pkl`,
    id,
    name,
    previewUrl: `/media/templates/${id}.png`,
    sourceType,
  };
}

function getOrCreateSession(state, sessionId) {
  const normalizedSessionId = String(sessionId || MOCK_VALUES.DEFAULT_SESSION_ID).trim() || MOCK_VALUES.DEFAULT_SESSION_ID;
  const existingSession = state.sessions.get(normalizedSessionId);
  if (existingSession) {
    return existingSession;
  }
  const createdSession = {
    sequence: 0,
    sessionId: normalizedSessionId,
  };
  state.sessions.set(normalizedSessionId, createdSession);
  return createdSession;
}

function resolveJobRuntime(job, nowMs) {
  const elapsedMs = Math.max(0, nowMs - job.createdAtMs);
  if (elapsedMs < MOCK_TIMINGS_MS.PROCESSING_START) {
    return {
      finishedAtMs: 0,
      frameIndex: 0,
      message: "queued",
      progress: 0,
      running: false,
      startedAtMs: 0,
      state: "queued",
    };
  }
  if (elapsedMs < MOCK_TIMINGS_MS.COMPLETE) {
    const processingElapsedMs = elapsedMs - MOCK_TIMINGS_MS.PROCESSING_START;
    const processingSpanMs = MOCK_TIMINGS_MS.COMPLETE - MOCK_TIMINGS_MS.PROCESSING_START;
    const progress = Math.min(0.98, Math.max(0.05, processingElapsedMs / processingSpanMs));
    return {
      finishedAtMs: 0,
      frameIndex: Math.max(1, Math.round(progress * MOCK_VALUES.JOB_TOTAL_FRAMES)),
      message: "rendering",
      progress,
      running: true,
      startedAtMs: job.createdAtMs + MOCK_TIMINGS_MS.PROCESSING_START,
      state: "processing",
    };
  }
  return {
    finishedAtMs: job.createdAtMs + MOCK_TIMINGS_MS.COMPLETE,
    frameIndex: MOCK_VALUES.JOB_TOTAL_FRAMES,
    message: "completed",
    progress: 1,
    running: false,
    startedAtMs: job.createdAtMs + MOCK_TIMINGS_MS.PROCESSING_START,
    state: "done",
  };
}

function listSessionJobs(state, sessionId) {
  return [...state.jobs.values()]
    .filter((job) => job.sessionId === sessionId)
    .sort((leftJob, rightJob) => leftJob.createdAtMs - rightJob.createdAtMs);
}

function resolveCurrentJob(state, sessionId) {
  const nowMs = Date.now();
  return listSessionJobs(state, sessionId).find((job) => resolveJobRuntime(job, nowMs).state !== "done") || null;
}

function buildHealthPayload(state, sessionId) {
  getOrCreateSession(state, sessionId);
  return {
    allowCustomSourceFrame: true,
    audioTuningPresets: {
      [MOCK_VALUES.AUDIO_PRESET_ID]: {
        animationRegion: "lip",
        audioLipSyncAssist: true,
        drivingMultiplier: 1.15,
      },
    },
    avatarSessionId: sessionId,
    avatarTransport: "http",
    avatarVideoHttpUrl: "/api/avatar/video.mp4",
    createSourceTemplatePackUrl: "/api/source-templates",
    defaultAnimationRegion: "all",
    defaultAudioMotionStride: 2,
    defaultMode: "preview",
    enqueueAudioUrl: "/api/avatar/enqueue",
    sourceTemplatePacksUrl: "/api/source-templates",
  };
}

function buildJobPayload(state, job) {
  const runtime = resolveJobRuntime(job, Date.now());
  const templateRecord = state.templates.get(job.templateId) || null;
  return {
    animationRegion: String(job.enqueueForm.animation_region || "all"),
    audioDurationSec: 1.2,
    audioLipSyncAssist: String(job.enqueueForm.audio_lip_sync_assist || "false") === "true",
    audioMotionStride: Number(job.enqueueForm.motion_stride || 2),
    audioTuningPreset: String(job.enqueueForm.audio_tuning_preset || ""),
    avatarSessionId: job.sessionId,
    createdAtMs: job.createdAtMs,
    exitCode: runtime.state === "done" ? 0 : null,
    finishedAtMs: runtime.finishedAtMs,
    jobId: job.jobId,
    logUrl: `/api/jobs/${job.jobId}/log`,
    mode: String(job.enqueueForm.mode || "preview"),
    previewComposition: null,
    queuePosition: runtime.state === "queued" ? 0 : 0,
    reportUrl: `/api/jobs/${job.jobId}/report`,
    resultConcatUrl: runtime.state === "done" ? `/jobs/${job.jobId}/result_concat.mp4` : "",
    resultVideoUrl: runtime.state === "done" ? `/jobs/${job.jobId}/result.mp4` : "",
    running: runtime.running,
    sourceFrameUrl: templateRecord ? templateRecord.previewUrl : "",
    sourceInputKind: "template_pack",
    sourceMediaType: templateRecord ? templateRecord.sourceType : "image",
    sourceTemplatePack: templateRecord,
    sourceTemplatePackId: job.templateId,
    startedAtMs: runtime.startedAtMs,
    state: runtime.state,
    status: {
      frameIndex: runtime.frameIndex,
      frameTotal: MOCK_VALUES.JOB_TOTAL_FRAMES,
      message: runtime.message,
      progress: runtime.progress,
      updatedAtMs: Date.now(),
    },
    statusUrl: `/api/jobs/${job.jobId}/status`,
  };
}

function buildAvatarStatusPayload(state, sessionId) {
  const session = getOrCreateSession(state, sessionId);
  session.sequence += 1;
  const currentJob = resolveCurrentJob(state, sessionId);
  const runtime = currentJob ? resolveJobRuntime(currentJob, Date.now()) : null;
  const activeJobs = listSessionJobs(state, sessionId).filter(
    (job) => resolveJobRuntime(job, Date.now()).state !== "done"
  );
  const templateRecord = currentJob ? state.templates.get(currentJob.templateId) : null;
  return {
    avatarSessionId: sessionId,
    bufferedStartProgress: 0.25,
    currentJobAudioDurationSec: currentJob ? 1.2 : 0,
    currentJobDrivingMediaUrl: currentJob ? `/jobs/${currentJob.jobId}/audio.wav` : "",
    currentJobId: currentJob ? currentJob.jobId : "",
    currentJobPreviewComposition: null,
    currentJobSourceFrameUrl: templateRecord ? templateRecord.previewUrl : "",
    currentJobSourceMediaType: templateRecord ? templateRecord.sourceType : "",
    currentJobStartedAtMs: runtime ? runtime.startedAtMs : 0,
    currentJobEndsAtMs: runtime && runtime.finishedAtMs ? runtime.finishedAtMs : 0,
    idleVideoAvailable: true,
    idleVideoUrl: "/idle/avatar_idle.mp4",
    idleStartedAtMs: Date.now() - 5_000,
    mode: runtime && runtime.state === "processing" ? "talking" : "idle",
    queueDepth: currentJob ? Math.max(0, activeJobs.length - 1) : 0,
    runningJobId: currentJob ? currentJob.jobId : "",
    sequence: session.sequence,
    status: "ok",
  };
}

function isLegacyPath(pathname) {
  return pathname === "/api/webrtc/offer"
    || pathname === "/ws/avatar"
    || pathname === "/ws/avatar/video"
    || pathname === "/ws/jobs/job_1"
    || pathname === "/ws/jobs/job_1/video"
    || /\/api\/jobs\/[^/]+\/stream\.mjpg$/.test(pathname)
    || /\/ws\/jobs\/[^/]+$/.test(pathname)
    || /\/ws\/jobs\/[^/]+\/video$/.test(pathname);
}

function ensureAuthorized(state, request, requestUrl, response, options = {}) {
  const normalizedOptions = options || {};
  const headerToken = normalizeBearerToken(request);
  const queryToken = String(requestUrl.searchParams.get("token") || "");
  const acceptedToken = headerToken || (normalizedOptions.allowQueryToken ? queryToken : "");
  if (acceptedToken === MOCK_VALUES.API_TOKEN) {
    return true;
  }
  state.authFailures.push({
    allowQueryToken: Boolean(normalizedOptions.allowQueryToken),
    path: requestUrl.pathname,
    queryToken,
    token: headerToken,
  });
  createErrorResponse(response, 401, "Unauthorized");
  return false;
}

async function createMockAvatarServer() {
  const seededTemplate = buildTemplateRecord({
    createdAtMs: Date.now(),
    id: MOCK_VALUES.SEEDED_TEMPLATE_ID,
    name: MOCK_VALUES.SEEDED_TEMPLATE_NAME,
    sourceType: "image",
  });
  const state = {
    authFailures: [],
    enqueueForms: [],
    jobs: new Map(),
    legacyRequests: [],
    requests: [],
    sessions: new Map(),
    streamRequests: [],
    templateForms: [],
    templates: new Map([[seededTemplate.id, seededTemplate]]),
  };

  const server = http.createServer(async (request, response) => {
    const requestUrl = new URL(request.url, "http://127.0.0.1");
    const sessionId = resolveSessionId(request, requestUrl);
    const requestBody = await parseRequestBody(request);
    state.requests.push({
      authorization: String(request.headers.authorization || ""),
      body: requestBody,
      method: request.method,
      path: requestUrl.pathname,
      query: Object.fromEntries(requestUrl.searchParams.entries()),
      sessionId,
    });

    if (isLegacyPath(requestUrl.pathname)) {
      state.legacyRequests.push({
        method: request.method,
        path: requestUrl.pathname,
      });
      createErrorResponse(
        response,
        410,
        "Legacy transport disabled. Use GET /api/avatar/video.mp4 for video and HTTP status polling endpoints."
      );
      return;
    }

    if (request.method === "GET" && requestUrl.pathname === "/api/health") {
      if (!ensureAuthorized(state, request, requestUrl, response)) {
        return;
      }
      createJsonResponse(response, buildHealthPayload(state, sessionId));
      return;
    }

    if (request.method === "GET" && requestUrl.pathname === "/api/avatar/status") {
      if (!ensureAuthorized(state, request, requestUrl, response)) {
        return;
      }
      createJsonResponse(response, buildAvatarStatusPayload(state, sessionId));
      return;
    }

    if (request.method === "GET" && requestUrl.pathname === "/api/source-templates") {
      if (!ensureAuthorized(state, request, requestUrl, response)) {
        return;
      }
      createJsonResponse(response, {
        items: [...state.templates.values()].sort((leftItem, rightItem) => leftItem.name.localeCompare(rightItem.name)),
        presetIds: [MOCK_VALUES.AUDIO_PRESET_ID],
      });
      return;
    }

    if (request.method === "POST" && requestUrl.pathname === "/api/source-templates") {
      if (!ensureAuthorized(state, request, requestUrl, response)) {
        return;
      }
      state.templateForms.push(requestBody);
      const sourceType = requestBody.source_video ? "video" : "image";
      const templateId = `template-built-${state.templateForms.length}`;
      const templateRecord = buildTemplateRecord({
        createdAtMs: Date.now(),
        id: templateId,
        name: String(requestBody.template_name || MOCK_VALUES.CREATED_TEMPLATE_NAME),
        sourceType,
      });
      state.templates.set(templateId, templateRecord);
      createJsonResponse(response, {
        item: templateRecord,
      });
      return;
    }

    if (request.method === "POST" && requestUrl.pathname === "/api/avatar/enqueue") {
      if (!ensureAuthorized(state, request, requestUrl, response)) {
        return;
      }
      state.enqueueForms.push(requestBody);
      const normalizedTemplateId = String(requestBody.source_template_pack || "").trim();
      if (!normalizedTemplateId) {
        createErrorResponse(response, 400, "source_template_pack is required");
        return;
      }
      if (!state.templates.has(normalizedTemplateId)) {
        createErrorResponse(response, 400, `Unknown template: ${normalizedTemplateId}`);
        return;
      }
      if (requestBody.source_image || requestBody.source_video || requestBody.source_frame) {
        createErrorResponse(response, 400, "Legacy runtime source overrides are disabled");
        return;
      }
      const jobId = `job_${state.enqueueForms.length}`;
      const jobRecord = {
        createdAtMs: Date.now(),
        enqueueForm: requestBody,
        jobId,
        sessionId,
        templateId: normalizedTemplateId,
      };
      state.jobs.set(jobId, jobRecord);
      createJsonResponse(response, buildJobPayload(state, jobRecord));
      return;
    }

    if (request.method === "GET" && /^\/api\/jobs\/[^/]+\/status$/.test(requestUrl.pathname)) {
      if (!ensureAuthorized(state, request, requestUrl, response)) {
        return;
      }
      const jobId = requestUrl.pathname.split("/")[3];
      const jobRecord = state.jobs.get(jobId);
      if (!jobRecord || jobRecord.sessionId !== sessionId) {
        createErrorResponse(response, 404, "Job not found");
        return;
      }
      createJsonResponse(response, buildJobPayload(state, jobRecord));
      return;
    }

    if (request.method === "GET" && requestUrl.pathname === "/api/avatar/video.mp4") {
      if (!ensureAuthorized(state, request, requestUrl, response, { allowQueryToken: true })) {
        return;
      }
      state.streamRequests.push({
        path: requestUrl.pathname,
        query: Object.fromEntries(requestUrl.searchParams.entries()),
        sessionId,
      });
      response.writeHead(200, {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Content-Type": "video/mp4",
        "X-Accel-Buffering": "no",
      });

      let closed = false;
      request.on("close", () => {
        closed = true;
      });

      const streamBoxes = [
        createMp4Box("ftyp", "isommock"),
        createMp4Box("moov", `session:${sessionId}`),
        createMp4Box("mdat", `phase:${resolveCurrentJob(state, sessionId) ? "talking" : "idle"}:frame:1`),
        createMp4Box("mdat", `phase:${resolveCurrentJob(state, sessionId) ? "talking" : "idle"}:frame:2`),
      ];

      for (const streamBox of streamBoxes) {
        if (closed) {
          return;
        }
        response.write(streamBox);
        await delay(MOCK_TIMINGS_MS.STREAM_CHUNK_INTERVAL);
      }
      response.end();
      return;
    }

    createErrorResponse(response, 404, "not-found");
  });

  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();

  return {
    async close() {
      server.close();
      await once(server, "close");
    },
    state,
    url: `http://${address.address}:${address.port}`,
  };
}

test("avatar client works against a stateful mock server that mirrors the realtime HTTP contract", async () => {
  const mockServer = await createMockAvatarServer();
  const client = createAvatarClient({
    apiBaseUrl: mockServer.url,
    token: MOCK_VALUES.API_TOKEN,
  });
  const sessionId = "browser-e2e-session";

  try {
    const sessionPayload = await client.createSession({ sessionId });
    assert.equal(sessionPayload.sessionId, sessionId);
    assert.match(sessionPayload.httpStreamUrl, /sessionId=browser-e2e-session/);
    assert.match(sessionPayload.httpStreamUrl, /token=test-token/);

    const idleAvatarStatus = await client.fetchAvatarStatus();
    assert.equal(idleAvatarStatus.mode, "idle");
    assert.equal(idleAvatarStatus.currentJobId, "");
    assert.equal(idleAvatarStatus.queueDepth, 0);

    const listedTemplates = await client.listSourceTemplates();
    assert.equal(listedTemplates.items.length, 1);
    assert.equal(listedTemplates.items[0].id, MOCK_VALUES.SEEDED_TEMPLATE_ID);

    const createdTemplate = await client.createSourceTemplate({
      sourceImage: new File(["image"], "avatar.png", { type: "image/png" }),
      templateName: MOCK_VALUES.CREATED_TEMPLATE_NAME,
    });
    assert.equal(createdTemplate.item.name, MOCK_VALUES.CREATED_TEMPLATE_NAME);
    assert.equal(mockServer.state.templateForms[0].source_image.name, "avatar.png");
    assert.equal(mockServer.state.templateForms[0].template_name, MOCK_VALUES.CREATED_TEMPLATE_NAME);

    const relistedTemplates = await client.listSourceTemplates();
    assert.equal(relistedTemplates.items.length, 2);
    assert.ok(relistedTemplates.items.some((template) => template.id === createdTemplate.item.id));

    const uploadedAudio = await client.uploadAudio({
      audio: new File(["audio"], "voice.wav", { type: "audio/wav" }),
      audioTuningPreset: MOCK_VALUES.AUDIO_PRESET_ID,
      params: {
        animationRegion: "lip",
        audioLipSyncAssist: true,
        drivingMultiplier: 1.5,
        mode: "full",
        motionStride: 3,
      },
      sourceTemplatePack: createdTemplate.item.id,
    });
    assert.equal(uploadedAudio.jobId, "job_1");
    assert.equal(mockServer.state.enqueueForms[0].audio.name, "voice.wav");
    assert.equal(mockServer.state.enqueueForms[0].source_template_pack, createdTemplate.item.id);
    assert.equal(mockServer.state.enqueueForms[0].audio_tuning_preset, MOCK_VALUES.AUDIO_PRESET_ID);
    assert.equal(mockServer.state.enqueueForms[0].mode, "full");
    assert.equal(mockServer.state.enqueueForms[0].motion_stride, "3");
    assert.equal(mockServer.state.enqueueForms[0].audio_lip_sync_assist, "true");
    assert.equal(mockServer.state.enqueueForms[0].driving_multiplier, "1.5");
    assert.equal("source_image" in mockServer.state.enqueueForms[0], false);
    assert.equal("source_video" in mockServer.state.enqueueForms[0], false);
    assert.equal("source_frame" in mockServer.state.enqueueForms[0], false);

    await assert.rejects(
      client.uploadAudio({
        audio: new File(["audio"], "voice.wav", { type: "audio/wav" }),
      }),
      (error) => error && error.code === AVATAR_CLIENT_ERROR_CODES.MISSING_SOURCE_TEMPLATE_PACK
    );

    const httpVideoElement = createMockVideoElement();
    const httpStreamPayload = await client.startStream({
      autoplay: true,
      muted: false,
      playsInline: true,
      videoElement: httpVideoElement,
    });
    assert.equal(httpStreamPayload.transport, "http");
    assert.equal(httpVideoElement.playCount, 1);
    assert.equal(httpVideoElement.muted, false);
    assert.equal(httpVideoElement.playsInline, true);
    assert.match(httpStreamPayload.streamUrl, /\/api\/avatar\/video\.mp4\?/);
    assert.equal(httpVideoElement.src, httpStreamPayload.streamUrl);

    await delay(MOCK_TIMINGS_MS.PROCESSING_START + 40);

    const talkingAvatarStatus = await client.fetchAvatarStatus();
    assert.equal(talkingAvatarStatus.mode, "talking");
    assert.equal(talkingAvatarStatus.currentJobId, uploadedAudio.jobId);
    assert.equal(talkingAvatarStatus.queueDepth, 0);

    const runningJobStatus = await client.fetchJobStatus(uploadedAudio.jobId);
    assert.equal(runningJobStatus.state, "processing");
    assert.ok(runningJobStatus.status.progress > 0);
    assert.ok(runningJobStatus.status.progress < 1);
    assert.ok(runningJobStatus.status.frameIndex > 0);

    const streamResponse = await fetch(httpStreamPayload.streamUrl);
    assert.equal(streamResponse.status, 200);
    assert.equal(streamResponse.headers.get("content-type"), "video/mp4");
    assert.match(String(streamResponse.headers.get("cache-control") || ""), /no-store/);
    const streamBoxes = await readMp4Boxes(streamResponse, 3);
    assert.ok(streamBoxes.boxes.length >= 3);
    assert.equal(streamBoxes.boxes[0].type, "ftyp");
    assert.equal(streamBoxes.boxes[1].type, "moov");
    assert.equal(streamBoxes.boxes[2].type, "mdat");
    assert.match(streamBoxes.boxes[1].payload.toString("utf8"), /session:browser-e2e-session/);
    assert.match(streamBoxes.combinedBuffer.toString("utf8"), /phase:talking/);

    await delay(MOCK_TIMINGS_MS.COMPLETE);

    const completedJobStatus = await client.fetchJobStatus(uploadedAudio.jobId);
    assert.equal(completedJobStatus.state, "done");
    assert.equal(completedJobStatus.running, false);
    assert.equal(completedJobStatus.status.progress, 1);
    assert.equal(completedJobStatus.resultVideoUrl, `/jobs/${uploadedAudio.jobId}/result.mp4`);
    assert.equal(completedJobStatus.resultConcatUrl, `/jobs/${uploadedAudio.jobId}/result_concat.mp4`);

    const finalAvatarStatus = await client.fetchAvatarStatus();
    assert.equal(finalAvatarStatus.mode, "idle");
    assert.equal(finalAvatarStatus.currentJobId, "");
    assert.equal(finalAvatarStatus.queueDepth, 0);

    await assert.rejects(
      client.startStream({
        transport: "webrtc",
        videoElement: createMockVideoElement(),
      }),
      (error) => error && error.code === AVATAR_CLIENT_ERROR_CODES.LEGACY_API_DISABLED
    );

    const pollUpdates = [];
    const stopPolling = client.startAvatarStatusPolling({
      intervalMs: 25,
      onUpdate(payload) {
        pollUpdates.push(payload.sequence);
      },
    });
    await delay(110);
    stopPolling();
    assert.ok(pollUpdates.length >= 3);
    assert.ok(pollUpdates.every((sequence, index) => index === 0 || sequence > pollUpdates[index - 1]));

    client.dispose();
    assert.equal(httpVideoElement.src, "");
    assert.equal(httpVideoElement.srcObject, null);

    const apiRequests = mockServer.state.requests.filter((requestRecord) => requestRecord.path !== "/api/avatar/video.mp4");
    assert.ok(apiRequests.every((requestRecord) => requestRecord.authorization === `Bearer ${MOCK_VALUES.API_TOKEN}`));
    assert.ok(mockServer.state.streamRequests.length >= 1);
    assert.ok(mockServer.state.streamRequests.every((requestRecord) => requestRecord.query.token === MOCK_VALUES.API_TOKEN));
    assert.ok(mockServer.state.streamRequests.every((requestRecord) => requestRecord.query.sessionId === sessionId));
    assert.deepEqual(mockServer.state.legacyRequests, []);
    assert.deepEqual(mockServer.state.authFailures, []);
  } finally {
    client.dispose();
    await mockServer.close();
  }
});

test("mock server rejects legacy transport endpoints like the production API", async () => {
  const mockServer = await createMockAvatarServer();
  try {
    const response = await fetch(`${mockServer.url}/api/webrtc/offer`, {
      headers: {
        Authorization: `Bearer ${MOCK_VALUES.API_TOKEN}`,
        "X-Avatar-Session-Id": "legacy-check",
      },
      method: "POST",
    });
    assert.equal(response.status, 410);
    const payload = await response.json();
    assert.match(String(payload.detail || ""), /Legacy transport disabled/);
  } finally {
    await mockServer.close();
  }
});
