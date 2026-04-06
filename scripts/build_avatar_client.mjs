import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const BUILD_FILE_NAMES = Object.freeze({
  BROWSER: "avatar-client.browser.js",
  ESM: "avatar-client.esm.js",
});

const BUILD_CONSTANTS = Object.freeze({
  GLOBAL_NAME: "AvatarClientSDK",
  EXPORT_BLOCK_PATTERN: /export\s*\{[\s\S]*?\};\s*$/,
});

const currentFilePath = fileURLToPath(import.meta.url);
const scriptsDirectoryPath = path.dirname(currentFilePath);
const projectRootPath = path.resolve(scriptsDirectoryPath, "..");
const sourceFilePath = path.join(projectRootPath, "src", "avatar-client.js");
const distributionDirectoryPath = path.join(projectRootPath, "dist");
const esmOutputFilePath = path.join(distributionDirectoryPath, BUILD_FILE_NAMES.ESM);
const browserOutputFilePath = path.join(distributionDirectoryPath, BUILD_FILE_NAMES.BROWSER);

function buildBrowserBundle(sourceCode) {
  if (!BUILD_CONSTANTS.EXPORT_BLOCK_PATTERN.test(sourceCode)) {
    throw new Error("The avatar client source does not end with an export block.");
  }
  const sourceWithoutExports = sourceCode.replace(BUILD_CONSTANTS.EXPORT_BLOCK_PATTERN, "").trimEnd();
  return [
    ";(function (globalScope) {",
    '  "use strict";',
    "",
    sourceWithoutExports,
    "",
    `  globalScope[${JSON.stringify(BUILD_CONSTANTS.GLOBAL_NAME)}] = avatarClientModule;`,
    "})(typeof window !== \"undefined\" ? window : globalThis);",
    "",
  ].join("\n");
}

async function buildAvatarClient() {
  const sourceCode = await readFile(sourceFilePath, "utf8");
  const browserBundle = buildBrowserBundle(sourceCode);
  await mkdir(distributionDirectoryPath, { recursive: true });
  await writeFile(esmOutputFilePath, sourceCode, "utf8");
  await writeFile(browserOutputFilePath, browserBundle, "utf8");
  process.stdout.write(
    `Built ${path.relative(projectRootPath, esmOutputFilePath)} and ${path.relative(projectRootPath, browserOutputFilePath)}\n`
  );
}

await buildAvatarClient();
