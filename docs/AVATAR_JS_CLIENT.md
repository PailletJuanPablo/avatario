# Avatar JavaScript Client

`src/avatar-client.js` is a lightweight frontend library for the realtime avatar API exposed by this repository.

It covers:

- session creation and reuse
- live video playback over HTTP MP4
- live playback over WebRTC
- audio upload and enqueue
- source template listing and creation
- avatar/job status polling and websocket helpers

## Build Outputs

Run:

```bash
npm run build:avatar-client
```

This generates:

- `dist/avatar-client.esm.js`
- `dist/avatar-client.browser.js`

## ESM Usage

```js
import { createAvatarClient } from "animation-realtime-trt/avatar-client";

const client = createAvatarClient({
  apiBaseUrl: "http://127.0.0.1:8010",
  token: "",
});

const { sessionId } = await client.createSession();

await client.startStream({
  transport: "http",
  videoElement: document.querySelector("video"),
  muted: true,
  autoplay: true,
});

await client.uploadAudio({
  audio: fileInput.files[0],
  template: "demo-template",
  params: {
    mode: "preview",
    animationRegion: "all",
    motionStride: 2,
  },
});
```

## Browser Global Usage

```html
<script src="/dist/avatar-client.browser.js"></script>
<script>
  const client = window.AvatarClientSDK.createAvatarClient({
    apiBaseUrl: "http://127.0.0.1:8010"
  });

  async function boot() {
    await client.createSession();
    await client.startStream({
      transport: "http",
      videoElement: document.getElementById("avatarVideo"),
      muted: true,
      autoplay: true
    });
  }

  boot();
</script>
```

## Upload Audio With a Template

```js
await client.uploadAudio({
  audio,
  template: "sales-avatar",
  params: {
    mode: "full",
    animationRegion: "lip",
    audioLipSyncAssist: true,
    audioMotionTuningEnabled: true,
    drivingMultiplier: 1.15,
  },
});
```

## Upload Audio With Explicit Source Media

```js
await client.uploadAudio({
  audio,
  sourceImage,
  params: {
    mode: "preview",
    stitching: true,
    relativeMotion: true,
    pasteBack: true,
  },
});
```

## Create a Source Template

```js
const templatePayload = await client.createSourceTemplate({
  sourceImage,
  templateName: "marketing-avatar",
});

console.log(templatePayload.item.id);
```

## WebRTC Playback

```js
await client.createSession();

await client.startStream({
  transport: "webrtc",
  videoElement: document.querySelector("video"),
});
```

## Main API Surface

- `createSession(options)`
- `connect(options)`
- `fetchHealth(options)`
- `fetchAvatarStatus(options)`
- `fetchJobStatus(jobId, options)`
- `startAvatarStatusPolling(options)`
- `startStream(options)`
- `connectHttpStream(options)`
- `connectWebRtc(options)`
- `buildHttpStreamUrl(options)`
- `uploadAudio(options)`
- `listSourceTemplates()`
- `createSourceTemplate(options)`
- `openAvatarStatusSocket(options)`
- `openJobStatusSocket(jobId, options)`
- `dispose()`
