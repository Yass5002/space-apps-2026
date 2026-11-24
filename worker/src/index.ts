import { DurableObject } from "cloudflare:workers";

export interface Env {
  AZURE_OPENAI_ENDPOINT: string;
  AZURE_OPENAI_DEPLOYMENT: string;
  AZURE_OPENAI_API_VERSION: string;
  AZURE_OPENAI_KEY: string;
  AZURE_SPEECH_REGION: string;
  AZURE_SPEECH_KEY: string;
  ALLOWED_ORIGINS: string;
  RATE_LIMITER: DurableObjectNamespace<RateLimiter>;
}

import { SYSTEM_PROMPT } from "../../src/assistant/systemPrompt";

const MAX_CHAT_BODY_BYTES = 64 * 1024; // 64 KB
const MAX_TTS_BODY_BYTES = 8 * 1024; // 8 KB
const MAX_MESSAGES = 40;
const MAX_MESSAGE_CONTENT_CHARS = 4000;
const MAX_TOOL_CONTENT_CHARS = 16000;
const MAX_COMPLETION_TOKENS = 400;
const MAX_TTS_TEXT_CHARS = 600;
const MAX_TOOLS = 20;
const MAX_TOOL_CALLS_PER_MSG = 20;
const TOOL_NAME_RE = /^[a-zA-Z0-9_-]{1,128}$/;

const ALLOWED_ROLES = new Set(["system", "user", "assistant", "tool"]);
const ALLOWED_CHAT_KEYS = new Set([
  "messages",
  "tools",
  "tool_choice",
  "temperature",
  "max_completion_tokens",
]);
const ALLOWED_USER_OR_SYSTEM_MSG_KEYS = new Set(["role", "content"]);
const ALLOWED_TOOL_MSG_KEYS = new Set(["role", "content", "tool_call_id"]);
const ALLOWED_ASSISTANT_MSG_KEYS = new Set([
  "role",
  "content",
  "tool_calls",
  "refusal",
  "annotations",
  "audio",
  "function_call",
]);
const ALLOWED_TOOL_SPEC_KEYS = new Set(["type", "function"]);
const ALLOWED_TOOL_FN_KEYS = new Set(["name", "description", "parameters", "strict"]);
const ALLOWED_TOOL_CHOICE_STRINGS = new Set(["auto", "none", "required"]);

const ALLOWED_TTS_KEYS = new Set(["text", "voice", "urgency"]);
const DEFAULT_VOICE = "en-US-EmmaMultilingualNeural";
const ALLOWED_VOICES = new Set([
  "en-US-AndrewMultilingualNeural",
  "en-US-AvaMultilingualNeural",
  "en-US-BrianMultilingualNeural",
  "en-US-EmmaMultilingualNeural",
]);
const ALLOWED_URGENCIES = new Set(["nominal", "advisory", "caution", "emergency"]);

type SpeechUrgency = "nominal" | "advisory" | "caution" | "emergency";

interface SanitizedToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

interface SanitizedMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_call_id?: string;
  tool_calls?: SanitizedToolCall[];
}

interface SanitizedToolSpec {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
    strict?: boolean;
  };
}

type SanitizedToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; function: { name: string } };

interface SanitizedChatBody {
  messages: SanitizedMessage[];
  tools?: SanitizedToolSpec[];
  tool_choice?: SanitizedToolChoice;
  temperature?: number;
  max_completion_tokens: number;
}

interface SanitizedTtsBody {
  text: string;
  voice: string;
  urgency: SpeechUrgency;
}

function getAllowedOrigins(env: Env): string[] {
  return (env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function isAllowedOrigin(origin: string | null, allowed: string[]): origin is string {
  return Boolean(origin && allowed.includes(origin));
}

function corsHeaders(origin: string | null, allowed: string[]): Record<string, string> {
  if (!isAllowedOrigin(origin, allowed)) {
    return { Vary: "Origin" };
  }
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function jsonResponse(
  body: unknown,
  status: number,
  headers: Record<string, string> = {}
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
  });
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasOnlyKeys(obj: Record<string, unknown>, allowed: Set<string>): boolean {
  for (const key of Object.keys(obj)) {
    if (!allowed.has(key)) return false;
  }
  return true;
}

async function readBoundedJsonObject(
  request: Request,
  maxBytes: number
): Promise<{ ok: true; data: Record<string, unknown> } | { ok: false; error: string }> {
  const contentLengthHeader = request.headers.get("Content-Length");
  if (contentLengthHeader !== null) {
    const declared = Number(contentLengthHeader);
    if (!Number.isFinite(declared) || declared < 0 || declared > maxBytes) {
      return { ok: false, error: "Request body exceeds size limit" };
    }
  }

  if (!request.body) {
    return { ok: false, error: "Missing request body" };
  }

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        totalBytes += value.byteLength;
        if (totalBytes > maxBytes) {
          await reader.cancel();
          return { ok: false, error: "Request body exceeds size limit" };
        }
        chunks.push(value);
      }
    }
  } catch {
    return { ok: false, error: "Failed to read request body" };
  }

  const combined = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(combined));
  } catch {
    return { ok: false, error: "Invalid JSON body" };
  }

  if (!isPlainObject(parsed)) {
    return { ok: false, error: "Request body must be a JSON object" };
  }

  return { ok: true, data: parsed };
}

function validateToolCalls(raw: unknown): SanitizedToolCall[] | null {
  if (!Array.isArray(raw) || raw.length > MAX_TOOL_CALLS_PER_MSG) {
    return null;
  }

  const sanitized: SanitizedToolCall[] = [];
  for (const item of raw) {
    if (!isPlainObject(item)) return null;
    if (typeof item.id !== "string" || item.id.length === 0 || item.id.length > 256) {
      return null;
    }
    if (item.type !== "function") return null;
    if (!isPlainObject(item.function)) return null;
    const fn = item.function;
    if (typeof fn.name !== "string" || !TOOL_NAME_RE.test(fn.name)) return null;
    if (typeof fn.arguments !== "string" || fn.arguments.length > MAX_MESSAGE_CONTENT_CHARS) {
      return null;
    }
    sanitized.push({
      id: item.id,
      type: "function",
      function: {
        name: fn.name,
        arguments: fn.arguments,
      },
    });
  }
  return sanitized;
}

function validateChatBody(
  raw: Record<string, unknown>
): { ok: true; body: SanitizedChatBody } | { ok: false; error: string } {
  if (!hasOnlyKeys(raw, ALLOWED_CHAT_KEYS)) {
    return { ok: false, error: "Request contains unsupported fields" };
  }

  if (!Array.isArray(raw.messages) || raw.messages.length === 0 || raw.messages.length > MAX_MESSAGES) {
    return { ok: false, error: `messages must be an array of 1 to ${MAX_MESSAGES} items` };
  }

  const sanitizedMessages: SanitizedMessage[] = [];
  for (const item of raw.messages) {
    if (!isPlainObject(item)) {
      return { ok: false, error: "Each message must be an object" };
    }
    if (typeof item.role !== "string" || !ALLOWED_ROLES.has(item.role)) {
      return { ok: false, error: "Invalid message role" };
    }

    if (item.role === "system") {
      // The Worker owns the system prompt; client-supplied ones are ignored.
      continue;
    }

    if (item.role === "user") {
      if (!hasOnlyKeys(item, ALLOWED_USER_OR_SYSTEM_MSG_KEYS)) {
        return { ok: false, error: `Unexpected field on ${item.role} message` };
      }
      if (typeof item.content !== "string" || item.content.length > MAX_MESSAGE_CONTENT_CHARS) {
        return {
          ok: false,
          error: `${item.role} message content must be a string up to ${MAX_MESSAGE_CONTENT_CHARS} chars`,
        };
      }
      sanitizedMessages.push({
        role: item.role,
        content: item.content,
      });
    } else if (item.role === "tool") {
      if (!hasOnlyKeys(item, ALLOWED_TOOL_MSG_KEYS)) {
        return { ok: false, error: "Unexpected field on tool message" };
      }
      if (typeof item.content !== "string" || item.content.length > MAX_TOOL_CONTENT_CHARS) {
        return {
          ok: false,
          error: `tool message content must be a string up to ${MAX_TOOL_CONTENT_CHARS} chars`,
        };
      }
      if (
        typeof item.tool_call_id !== "string" ||
        item.tool_call_id.length === 0 ||
        item.tool_call_id.length > 256
      ) {
        return { ok: false, error: "tool message requires a valid tool_call_id" };
      }
      sanitizedMessages.push({
        role: "tool",
        tool_call_id: item.tool_call_id,
        content: item.content,
      });
    } else {
      // assistant
      if (!hasOnlyKeys(item, ALLOWED_ASSISTANT_MSG_KEYS)) {
        return { ok: false, error: "Unexpected field on assistant message" };
      }
      let content: string | null;
      if (typeof item.content === "string") {
        if (item.content.length > MAX_MESSAGE_CONTENT_CHARS) {
          return {
            ok: false,
            error: `assistant message content exceeds ${MAX_MESSAGE_CONTENT_CHARS} chars`,
          };
        }
        content = item.content;
      } else if (item.content === null || item.content === undefined) {
        content = null;
      } else {
        return { ok: false, error: "Invalid assistant message content" };
      }

      let toolCalls: SanitizedToolCall[] | undefined;
      if (item.tool_calls !== undefined) {
        const validated = validateToolCalls(item.tool_calls);
        if (!validated) {
          return { ok: false, error: "Invalid tool_calls on assistant message" };
        }
        if (validated.length > 0) {
          toolCalls = validated;
        }
      }

      if (content === null && !toolCalls) {
        return { ok: false, error: "assistant message must include content or tool_calls" };
      }

      const msg: SanitizedMessage = { role: "assistant", content };
      if (toolCalls) {
        msg.tool_calls = toolCalls;
      }
      sanitizedMessages.push(msg);
    }
  }

  if (!sanitizedMessages.some((m) => m.role === "user" || m.role === "tool")) {
    return { ok: false, error: "messages must include a user or tool message" };
  }
  sanitizedMessages.unshift({ role: "system", content: SYSTEM_PROMPT });

  let sanitizedTools: SanitizedToolSpec[] | undefined;
  if (raw.tools !== undefined) {
    if (!Array.isArray(raw.tools) || raw.tools.length > MAX_TOOLS) {
      return { ok: false, error: `tools must be an array of at most ${MAX_TOOLS} items` };
    }
    sanitizedTools = [];
    for (const tool of raw.tools) {
      if (!isPlainObject(tool) || !hasOnlyKeys(tool, ALLOWED_TOOL_SPEC_KEYS)) {
        return { ok: false, error: "Invalid tool specification" };
      }
      if (tool.type !== "function" || !isPlainObject(tool.function)) {
        return { ok: false, error: "Tool must have type 'function' and a function definition" };
      }
      const fn = tool.function;
      if (!hasOnlyKeys(fn, ALLOWED_TOOL_FN_KEYS)) {
        return { ok: false, error: "Unexpected field in tool function definition" };
      }
      if (typeof fn.name !== "string" || !TOOL_NAME_RE.test(fn.name)) {
        return { ok: false, error: "Invalid tool function name" };
      }
      const sanitizedFn: SanitizedToolSpec["function"] = { name: fn.name };
      if (fn.description !== undefined) {
        if (typeof fn.description !== "string" || fn.description.length > 1000) {
          return { ok: false, error: "Invalid tool function description" };
        }
        sanitizedFn.description = fn.description;
      }
      if (fn.parameters !== undefined) {
        if (!isPlainObject(fn.parameters)) {
          return { ok: false, error: "Tool function parameters must be an object" };
        }
        sanitizedFn.parameters = fn.parameters;
      }
      if (fn.strict !== undefined) {
        if (typeof fn.strict !== "boolean") {
          return { ok: false, error: "Tool function strict flag must be a boolean" };
        }
        sanitizedFn.strict = fn.strict;
      }
      sanitizedTools.push({ type: "function", function: sanitizedFn });
    }
  }

  let sanitizedToolChoice: SanitizedToolChoice | undefined;
  if (raw.tool_choice !== undefined) {
    if (typeof raw.tool_choice === "string") {
      if (!ALLOWED_TOOL_CHOICE_STRINGS.has(raw.tool_choice)) {
        return { ok: false, error: "Invalid tool_choice value" };
      }
      sanitizedToolChoice = raw.tool_choice as "auto" | "none" | "required";
    } else if (isPlainObject(raw.tool_choice)) {
      if (
        !hasOnlyKeys(raw.tool_choice, ALLOWED_TOOL_SPEC_KEYS) ||
        raw.tool_choice.type !== "function" ||
        !isPlainObject(raw.tool_choice.function)
      ) {
        return { ok: false, error: "Invalid tool_choice object" };
      }
      const fnKeys = Object.keys(raw.tool_choice.function);
      const fnName = raw.tool_choice.function.name;
      if (
        fnKeys.length !== 1 ||
        fnKeys[0] !== "name" ||
        typeof fnName !== "string" ||
        !TOOL_NAME_RE.test(fnName)
      ) {
        return { ok: false, error: "Invalid tool_choice function name" };
      }
      sanitizedToolChoice = { type: "function", function: { name: fnName } };
    } else {
      return { ok: false, error: "Invalid tool_choice type" };
    }
  }

  let temperature: number | undefined;
  if (raw.temperature !== undefined) {
    if (typeof raw.temperature !== "number" || !Number.isFinite(raw.temperature)) {
      return { ok: false, error: "temperature must be a finite number" };
    }
    temperature = Math.min(1, Math.max(0, raw.temperature));
  }

  let maxCompletionTokens = MAX_COMPLETION_TOKENS;
  if (raw.max_completion_tokens !== undefined) {
    if (
      typeof raw.max_completion_tokens !== "number" ||
      !Number.isInteger(raw.max_completion_tokens) ||
      raw.max_completion_tokens < 1
    ) {
      return { ok: false, error: "max_completion_tokens must be a positive integer" };
    }
    maxCompletionTokens = Math.min(MAX_COMPLETION_TOKENS, raw.max_completion_tokens);
  }

  const upstreamBody: SanitizedChatBody = {
    messages: sanitizedMessages,
    max_completion_tokens: maxCompletionTokens,
  };
  if (sanitizedTools !== undefined) {
    upstreamBody.tools = sanitizedTools;
  }
  if (sanitizedToolChoice !== undefined) {
    upstreamBody.tool_choice = sanitizedToolChoice;
  }
  if (temperature !== undefined) {
    upstreamBody.temperature = temperature;
  }

  return { ok: true, body: upstreamBody };
}

function validateTtsBody(
  raw: Record<string, unknown>
): { ok: true; body: SanitizedTtsBody } | { ok: false; error: string } {
  if (!hasOnlyKeys(raw, ALLOWED_TTS_KEYS)) {
    return { ok: false, error: "Request contains unsupported fields" };
  }

  if (
    typeof raw.text !== "string" ||
    raw.text.trim().length === 0 ||
    raw.text.length > MAX_TTS_TEXT_CHARS
  ) {
    return {
      ok: false,
      error: `text must be a non-empty string up to ${MAX_TTS_TEXT_CHARS} chars`,
    };
  }

  const voice = raw.voice === undefined ? DEFAULT_VOICE : raw.voice;
  if (typeof voice !== "string" || !ALLOWED_VOICES.has(voice)) {
    return { ok: false, error: "Unsupported voice" };
  }

  const urgency = raw.urgency === undefined ? "nominal" : raw.urgency;
  if (typeof urgency !== "string" || !ALLOWED_URGENCIES.has(urgency)) {
    return { ok: false, error: "Unsupported urgency" };
  }

  return {
    ok: true,
    body: {
      text: raw.text,
      voice,
      urgency: urgency as SpeechUrgency,
    },
  };
}

// Single shared rate limiter (one Durable Object instance), so limits hold no
// matter which Cloudflare server a request lands on.
export class RateLimiter extends DurableObject {
  private hits = new Map<string, { count: number; resetAt: number }>();

  async hit(key: string, limit: number, windowMs: number): Promise<boolean> {
    const now = Date.now();
    let entry = this.hits.get(key);
    if (!entry || now >= entry.resetAt) {
      entry = { count: 0, resetAt: now + windowMs };
      this.hits.set(key, entry);
    }
    entry.count += 1;
    if (this.hits.size > 5000) {
      for (const [k, v] of this.hits) if (now >= v.resetAt) this.hits.delete(k);
    }
    return entry.count <= limit;
  }
}

const PER_IP_LIMIT = 20; // requests per minute per IP, per endpoint
const GLOBAL_LIMIT = 120; // requests per minute across everyone, per endpoint

async function allowRequest(env: Env, kind: "chat" | "tts", ip: string): Promise<boolean> {
  const limiter = env.RATE_LIMITER.get(env.RATE_LIMITER.idFromName("global"));
  const [perIp, global] = await Promise.all([
    limiter.hit(`${kind}:${ip}`, PER_IP_LIMIT, 60_000),
    limiter.hit(`${kind}:all`, GLOBAL_LIMIT, 60_000),
  ]);
  return perIp && global;
}

function getClientIp(request: Request): string {
  return request.headers.get("CF-Connecting-IP")?.trim() || "unknown";
}

function escapeXml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const allowedOrigins = getAllowedOrigins(env);
    const origin = request.headers.get("Origin");
    const cors = corsHeaders(origin, allowedOrigins);

    if (request.method === "OPTIONS") {
      if (!isAllowedOrigin(origin, allowedOrigins)) {
        return jsonResponse({ error: "Forbidden origin" }, 403, { Vary: "Origin" });
      }
      return new Response(null, { status: 204, headers: cors });
    }

    const url = new URL(request.url);

    if (url.pathname === "/api/health" && request.method === "GET") {
      return jsonResponse(
        { ok: Boolean(env.AZURE_OPENAI_KEY && env.AZURE_SPEECH_KEY) },
        200,
        cors
      );
    }

    if (url.pathname === "/api/chat" && request.method === "POST") {
      if (!isAllowedOrigin(origin, allowedOrigins)) {
        return jsonResponse({ error: "Forbidden origin" }, 403, { Vary: "Origin" });
      }

      const success = await allowRequest(env, "chat", getClientIp(request));
      if (!success) {
        return jsonResponse({ error: "Rate limit exceeded" }, 429, {
          ...cors,
          "Retry-After": "60",
        });
      }

      const rawBody = await readBoundedJsonObject(request, MAX_CHAT_BODY_BYTES);
      if (!rawBody.ok) {
        return jsonResponse({ error: rawBody.error }, 400, cors);
      }

      const validated = validateChatBody(rawBody.data);
      if (!validated.ok) {
        return jsonResponse({ error: validated.error }, 400, cors);
      }

      if (!env.AZURE_OPENAI_KEY) {
        return jsonResponse({ error: "Service not configured" }, 503, cors);
      }

      const endpoint = env.AZURE_OPENAI_ENDPOINT.replace(/\/+$/, "");
      const targetUrl = `${endpoint}/openai/deployments/${encodeURIComponent(
        env.AZURE_OPENAI_DEPLOYMENT
      )}/chat/completions?api-version=${encodeURIComponent(env.AZURE_OPENAI_API_VERSION)}`;

      const upstream = await fetch(targetUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "api-key": env.AZURE_OPENAI_KEY,
        },
        body: JSON.stringify(validated.body),
      });

      const text = await upstream.text();
      return new Response(text, {
        status: upstream.status,
        headers: { ...cors, "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/api/tts" && request.method === "POST") {
      if (!isAllowedOrigin(origin, allowedOrigins)) {
        return jsonResponse({ error: "Forbidden origin" }, 403, { Vary: "Origin" });
      }

      const success = await allowRequest(env, "tts", getClientIp(request));
      if (!success) {
        return jsonResponse({ error: "Rate limit exceeded" }, 429, {
          ...cors,
          "Retry-After": "60",
        });
      }

      const rawBody = await readBoundedJsonObject(request, MAX_TTS_BODY_BYTES);
      if (!rawBody.ok) {
        return jsonResponse({ error: rawBody.error }, 400, cors);
      }

      const validated = validateTtsBody(rawBody.data);
      if (!validated.ok) {
        return jsonResponse({ error: validated.error }, 400, cors);
      }

      if (!env.AZURE_SPEECH_KEY) {
        return jsonResponse({ error: "Service not configured" }, 503, cors);
      }

      const { text, voice, urgency } = validated.body;
      const langMatch = voice.match(/^([a-z]{2}-[A-Z]{2})/);
      const lang = langMatch ? langMatch[1] : "en-US";
      const prosodyRate =
        urgency === "emergency"
          ? "+14%"
          : urgency === "caution"
            ? "+7%"
            : urgency === "advisory"
              ? "+3%"
              : "0%";
      const prosodyPitch =
        urgency === "emergency" ? "+2%" : urgency === "caution" ? "+1%" : "0%";
      const ssml = `<speak version='1.0' xml:lang='${lang}'><voice xml:lang='${lang}' name='${escapeXml(
        voice
      )}'><prosody rate='${prosodyRate}' pitch='${prosodyPitch}'>${escapeXml(text)}</prosody></voice></speak>`;

      const ttsUrl = `https://${env.AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1`;
      const res = await fetch(ttsUrl, {
        method: "POST",
        headers: {
          "Ocp-Apim-Subscription-Key": env.AZURE_SPEECH_KEY,
          "Content-Type": "application/ssml+xml",
          "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
          "User-Agent": "arasaka-health-assistant",
        },
        body: ssml,
      });

      if (!res.ok) {
        const err = await res.text();
        return new Response(err, {
          status: res.status,
          headers: { ...cors, "Content-Type": "text/plain" },
        });
      }

      return new Response(res.body, {
        status: 200,
        headers: {
          ...cors,
          "Content-Type": "audio/mpeg",
          "Cache-Control": "no-store",
        },
      });
    }

    return jsonResponse({ error: "Not found" }, 404, cors);
  },
};
