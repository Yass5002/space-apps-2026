export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
}

export interface ChatToolSpec {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface ProxyHealth {
  ok: boolean;
  deployment?: string;
  speechRegion?: string;
  hasLlmKey?: boolean;
  hasSpeechKey?: boolean;
}

export const DEFAULT_PROXY_BASE_URL =
  import.meta.env.VITE_PROXY_URL ||
  (typeof window !== "undefined" && window.location.hostname === "nasa.yssn.tech"
    ? "https://nasa-api.yssn.tech"
    : "/api-proxy");

export class AzureProxyClient {
  private baseUrl: string;

  constructor(baseUrl = DEFAULT_PROXY_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  public setBaseUrl(url: string): void {
    this.baseUrl = url.replace(/\/+$/, "");
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  public async checkHealth(): Promise<ProxyHealth> {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`, { method: "GET" });
      if (!res.ok) return { ok: false };
      return await res.json();
    } catch {
      return { ok: false };
    }
  }

  public async chatCompletion(
    messages: ChatMessage[],
    tools: ChatToolSpec[],
    signal?: AbortSignal
  ): Promise<{
    message: ChatMessage;
  }> {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        // The proxy injects the system prompt server-side.
        messages: messages.filter((m) => m.role !== "system"),
        tools,
        tool_choice: "auto",
        temperature: 0.2,
        max_completion_tokens: 280,
      }),
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`Azure chat proxy HTTP ${res.status}: ${txt}`);
    }

    const data = await res.json();
    const choice = data?.choices?.[0]?.message;
    if (!choice) {
      throw new Error("Invalid Azure chat response structure");
    }
    return { message: choice };
  }

  public async synthesizeSpeech(
    text: string,
    voice: string,
    urgency: "nominal" | "advisory" | "caution" | "emergency" = "nominal",
    signal?: AbortSignal
  ): Promise<ArrayBuffer> {
    const res = await fetch(`${this.baseUrl}/api/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({ text, voice, urgency }),
    });
    if (!res.ok) {
      throw new Error(`Azure TTS HTTP ${res.status}`);
    }
    return await res.arrayBuffer();
  }
}
