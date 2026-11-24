import type { AssistantEngineBridge } from "../engine/assistantApi";
import { ENGINE_LIMITS } from "../engine/config";
import type { AlertObject, IndicatorId, ProtocolId, SuitActionName } from "../engine/types";
import { AzureProxyClient, type ChatMessage, type ChatToolSpec } from "./azureClient";
import { lookupManualTopic } from "./flightManual";
import { SYSTEM_PROMPT } from "./systemPrompt";
import type { SpeechUrgency, VoiceService } from "./voiceService";

export interface AssistantCaptionState {
  speaker: "ASSISTANT" | "CREW";
  text: string;
  interimTranscript?: string;
  timestampT: number;
  alertClass: 0 | 1 | 2 | 3;
  isBusy: boolean;
  isListening: boolean;
  isSpeaking?: boolean;
  voiceConnected: boolean;
  cloudStatus: "ready" | "fallback" | "checking";
  escalationStage?: 0 | 1 | 2 | 3;
  unresponsiveEscalated?: boolean;
}

const ASSISTANT_TOOLS: ChatToolSpec[] = [
  {
    type: "function",
    function: {
      name: "get_state",
      description:
        "Get current live telemetry, EVA context, active alerts, and current safety step.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "lookup_manual",
      description:
        "Look up authoritative NASA-STD-3001 limits, Plan B terrain/suit contingencies, and clinical rationale from the Operations Manual.",
      parameters: {
        type: "object",
        properties: {
          topic: {
            type: "string",
            enum: [
              "radiation_spe",
              "suit_depressurization",
              "co2_scrubber",
              "hypoxia_pio2",
              "exertion_cardio",
              "orthostatic",
              "sleep_circadian",
              "bone_exercise",
              "immune_sans",
              "all_limits",
            ],
          },
        },
        required: ["topic"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "summarise",
      description: "Summarise an indicator over a recent time window in minutes.",
      parameters: {
        type: "object",
        properties: {
          indicator: { type: "string" },
          windowMinutes: { type: "number" },
        },
        required: ["indicator", "windowMinutes"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_alerts",
      description: "List active alerts ranked by class and urgency.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "explain",
      description: "Get the recorded evidence chain for an alert ID.",
      parameters: {
        type: "object",
        properties: { alertId: { type: "string" } },
        required: ["alertId"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "compare_to_plan",
      description: "Compare dose_cumulative, sleep_hours, exercise_sessions, ppco2, or pio2 to NASA/mission limits.",
      parameters: {
        type: "object",
        properties: {
          metric: {
            type: "string",
            enum: ["dose_cumulative", "sleep_hours", "exercise_sessions", "ppco2", "pio2"],
          },
        },
        required: ["metric"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "start_protocol",
      description: "Start a guided step-by-step response protocol.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            enum: [
              "shelter",
              "air_check",
              "exertion_recovery",
              "orthostatic_response",
              "resistive_exercise",
              "sleep_reset",
              "calm_breathing",
            ],
          },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "complete_step",
      description: "Confirm completion of the current protocol step and advance to the next step.",
      parameters: {
        type: "object",
        properties: {
          protocol: { type: "string" },
          step: { type: "number" },
        },
        required: ["protocol", "step"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "acknowledge",
      description: "Mark an alert as acknowledged.",
      parameters: {
        type: "object",
        properties: { alertId: { type: "string" } },
        required: ["alertId"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "update_action_plan",
      description:
        "Replace the current HUD protocol instruction with an improvised Plan B when Sir reports Plan A is impossible or unsafe (e.g. too far from habitat, rover dead).",
      parameters: {
        type: "object",
        properties: {
          title: { type: "string", description: "Short Plan B banner e.g. PLAN B: TERRAIN SHIELD" },
          currentAction: { type: "string", description: "Immediate concise physical instruction for the visor HUD" },
          confirmationPrompt: { type: "string", description: "What Sir should confirm once done" },
          reason: { type: "string", description: "Why Plan A was adapted" },
          protocolId: { type: "string" },
        },
        required: ["title", "currentAction"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "execute_suit_action",
      description:
        "Command immediate EVA suit hardware or alarm mitigation (shielding posture, secondary O2 tank, open-loop purge, seal assist, rest mode, silence alarm, hail flight surgeon).",
      parameters: {
        type: "object",
        properties: {
          action: {
            type: "string",
            enum: [
              "engage_terrain_shield",
              "switch_secondary_o2",
              "open_loop_o2_purge",
              "seal_suit_breach",
              "rest_recovery_mode",
              "silence_alarm",
              "hail_flight_surgeon",
            ],
          },
        },
        required: ["action"],
      },
    },
  },
];

function formatMissionClock(day: number, todSec: number): string {
  const h = Math.floor(todSec / 3600) % 24;
  const m = Math.floor((todSec % 3600) / 60);
  return `Day ${day} ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function classToUrgency(alertClass: 0 | 1 | 2 | 3): SpeechUrgency {
  if (alertClass === 1) return "emergency";
  if (alertClass === 2) return "caution";
  if (alertClass === 3) return "advisory";
  return "nominal";
}

export class AssistantController {
  private readonly bridge: AssistantEngineBridge;
  public readonly client: AzureProxyClient;
  public readonly voice: VoiceService;

  private history: ChatMessage[] = [{ role: "system", content: SYSTEM_PROMPT }];
  private lastAnnouncedAlertIds = new Set<string>();
  private announcedDetectorClass = new Map<string, number>();
  private lastDetectorAnnouncedAtMs = new Map<string, number>();
  private captionListeners = new Set<(cap: AssistantCaptionState) => void>();

  // Prevents step-completion synchronous notify() from causing duplicate alert speech
  private suppressAlertSync = false;
  private alertSpeechGen = 0;
  private askSeq = 0;
  private activeAskAbort: AbortController | null = null;
  private activeAlertAbort: AbortController | null = null;

  // Wall-clock NASA No-Response Escalation Ladder state
  private watchdogTimer: ReturnType<typeof setInterval> | null = null;
  private hazardTrackingKey: string | null = null;
  private hazardUnconfirmedSinceMs = 0;
  private lastCrewInteractionMs = Date.now();
  private lastAiBusyAtMs = 0;
  private lastStage3PulseMs = 0;
  private escalationStage: 0 | 1 | 2 | 3 = 0;
  private lastSpokenReply = "All suit systems look good, Sir.";

  public caption: AssistantCaptionState = {
    speaker: "ASSISTANT",
    text: "All suit systems look good, Sir.",
    interimTranscript: "",
    timestampT: 0,
    alertClass: 0,
    isBusy: false,
    isListening: false,
    isSpeaking: false,
    voiceConnected: false,
    cloudStatus: "checking",
    escalationStage: 0,
    unresponsiveEscalated: false,
  };

  constructor(
    bridge: AssistantEngineBridge,
    client: AzureProxyClient,
    voice: VoiceService
  ) {
    this.bridge = bridge;
    this.client = client;
    this.voice = voice;
    this.voice.onSpeakingChange = (speaking) => {
      this.emitCaption({ isSpeaking: speaking });
    };
  }

  private sanitizeReply(text: string): string {
    return text
      .replace(/\s*\u2014\s*/g, ", ")
      .replace(/\u2014/g, "-")
      .replace(/\(\s*\d+\s+more\s+(?:waiting|queued)\.?\s*\)/gi, "")
      .replace(/(?:^|(?<=[.!?]\s+))Next(?:\s+step)?:\s*/gi, "")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  /**
   * Cancels in-flight LLM calls and queued/active AI audio so user confirmations
   * or new queries never lose a race against delayed background replies.
   */
  private cancelPendingAiWork(stopAudio = true): void {
    this.alertSpeechGen += 1;
    if (this.activeAlertAbort) {
      try {
        this.activeAlertAbort.abort();
      } catch {
        // ignore
      }
      this.activeAlertAbort = null;
    }

    this.askSeq += 1;
    if (this.activeAskAbort) {
      try {
        this.activeAskAbort.abort();
      } catch {
        // ignore
      }
      this.activeAskAbort = null;
    }

    if (stopAudio) {
      this.voice.stopSpeech();
    }
  }

  public subscribe(fn: (cap: AssistantCaptionState) => void): () => void {
    this.captionListeners.add(fn);
    fn(this.caption);
    return () => {
      this.captionListeners.delete(fn);
    };
  }

  private emitCaption(patch: Partial<AssistantCaptionState>): void {
    const cleanPatch =
      typeof patch.text === "string" ? { ...patch, text: this.sanitizeReply(patch.text) } : patch;
    if (cleanPatch.speaker === "ASSISTANT" && cleanPatch.text) {
      this.lastSpokenReply = cleanPatch.text;
    }
    this.caption = { ...this.caption, ...cleanPatch };
    for (const fn of this.captionListeners) {
      fn(this.caption);
    }
  }

  public async probeCloud(): Promise<void> {
    this.emitCaption({ cloudStatus: "checking" });
    const res = await this.client.checkHealth();
    this.emitCaption({ cloudStatus: res.ok ? "ready" : "fallback" });
  }

  public setVoiceConnected(connected: boolean): void {
    this.voice.enabled = connected;
    if (connected) {
      this.voice.unlockAudio();
      this.voice.fx.setMuted(false);
    } else {
      this.voice.stopSpeech();
      this.voice.fx.setMuted(true);
    }
    this.emitCaption({ voiceConnected: connected });
  }

  /**
   * Silent nominal suit entry: plays short boot wake chord, opens mic if enabled,
   * and stays quiet during nominal conditions unless spoken to.
   */
  public enterSuitSilent(openMic = false): void {
    this.voice.fx.playBootChime();
    const live = this.bridge.get_state();
    this.emitCaption({
      speaker: "ASSISTANT",
      text: "All suit systems look good, Sir.",
      timestampT: live.t,
      alertClass: live.overallClass,
    });
    if (openMic) {
      this.startOpenMic();
    }
    this.startWatchdog();
  }

  public startOpenMic(): boolean {
    if (this.voice.listening) return true;
    this.voice.unlockAudio();
    if (!this.voice.enabled) {
      this.setVoiceConnected(true);
    }
    return this.voice.startContinuousListening({
      onInterim: (interim) => {
        this.emitCaption({ interimTranscript: interim });
      },
      onBargeIn: () => {
        this.markCrewResponsive();
      },
      onFinal: (transcript) => {
        this.emitCaption({ interimTranscript: "" });
        this.handleVoiceUtterance(transcript);
      },
      onStateChange: (listening, errorNote) => {
        this.emitCaption({
          isListening: listening,
          interimTranscript: listening ? this.caption.interimTranscript : "",
          ...(errorNote ? { speaker: "ASSISTANT", text: `${errorNote}, Sir.` } : {}),
        });
      },
    });
  }

  public stopOpenMic(): void {
    this.voice.stopListening();
    this.emitCaption({ isListening: false, interimTranscript: "" });
  }

  public toggleListening(): void {
    if (this.voice.listening) {
      this.stopOpenMic();
    } else {
      this.startOpenMic();
    }
  }

  private handleVoiceUtterance(transcript: string): void {
    const clean = transcript.trim().toLowerCase();
    if (!clean) return;

    if (/^(mic off|stop listening|close mic|standby mic)/.test(clean)) {
      this.stopOpenMic();
      this.emitCaption({ speaker: "ASSISTANT", text: "Microphone muted, Sir." });
      return;
    }

    if (
      /^(stop talking|quiet|silence|mute|belay that)\b/.test(clean) &&
      !/\b(alarm|alarms|alert|alerts|beep|siren|noise)\b/.test(clean)
    ) {
      this.cancelPendingAiWork(true);
      this.markCrewResponsive();
      this.emitCaption({ isBusy: false });
      return;
    }

    void this.ask(transcript);
  }

  public startWatchdog(): void {
    if (this.watchdogTimer) return;
    this.watchdogTimer = setInterval(() => {
      this.evaluateUnresponsiveEscalation();
    }, 1000);
  }

  public dispose(): void {
    if (this.watchdogTimer) {
      clearInterval(this.watchdogTimer);
      this.watchdogTimer = null;
    }
    this.cancelPendingAiWork(true);
    this.voice.stopListening();
    this.voice.stopSpeech();
    this.voice.fx.dispose();
  }

  private markCrewResponsive(): void {
    this.lastCrewInteractionMs = Date.now();
    if (this.escalationStage !== 0 || this.caption.unresponsiveEscalated) {
      this.escalationStage = 0;
      this.emitCaption({ escalationStage: 0, unresponsiveEscalated: false });
    }
  }

  private evaluateUnresponsiveEscalation(): void {
    const live = this.bridge.get_state();
    const activeProto = live.activeProtocol && !live.activeProtocol.completed ? live.activeProtocol : null;
    const primary = live.rankedPrimaryAlert;
    const hasUrgentHazard =
      Boolean(activeProto) || Boolean(primary && primary.class <= 2 && primary.state !== "acknowledged");

    if (!hasUrgentHazard) {
      this.hazardTrackingKey = null;
      if (this.escalationStage !== 0 || this.caption.unresponsiveEscalated) {
        this.escalationStage = 0;
        this.emitCaption({ escalationStage: 0, unresponsiveEscalated: false });
      }
      return;
    }

    const key = activeProto
      ? `${activeProto.id}:${activeProto.currentStepIndex}`
      : `${primary!.id}:${primary!.class}`;
    const now = Date.now();
    // Never count silence or stack escalation prompts while AI is actively speaking or synthesizing
    if (this.voice.busySpeaking) {
      this.lastAiBusyAtMs = now;
      return;
    }

    if (this.hazardTrackingKey !== key) {
      this.hazardTrackingKey = key;
      this.hazardUnconfirmedSinceMs = now;
      this.escalationStage = 0;
      return;
    }

    if (this.voice.busySpeaking) {
      return;
    }

    const silentSince = Math.max(
      this.hazardUnconfirmedSinceMs,
      this.lastCrewInteractionMs,
      this.lastAiBusyAtMs
    );
    const elapsedSec = (now - silentSince) / 1000;

    // Stage 1 (~25s): Direct action check-in without repeating value dumps
    if (this.escalationStage === 0 && elapsedSec >= 25) {
      this.escalationStage = 1;
      const stepHint = activeProto?.contingencyTitle
        ? activeProto.currentStep.instruction
        : "Tap Done or tell me if something is blocking you.";
      const msg = `Sir, status check. ${stepHint}`;
      this.voice.fx.playAlertTone(live.overallClass || 2);
      this.emitCaption({
        speaker: "ASSISTANT",
        text: msg,
        timestampT: live.t,
        alertClass: live.overallClass,
        escalationStage: 1,
      });
      if (this.voice.enabled && this.voice.autoSpeakAlerts) {
        void this.voice.speak(msg, classToUrgency(live.overallClass), false);
      }
      return;
    }

    // Stage 2 (~50s): Incapacitation comms check
    if (this.escalationStage === 1 && elapsedSec >= 50) {
      this.escalationStage = 2;
      const msg = `Sir, please check in. Let me know you are okay or confirm the current step.`;
      this.voice.fx.playAlertTone(1);
      this.emitCaption({
        speaker: "ASSISTANT",
        text: msg,
        timestampT: live.t,
        alertClass: 1,
        escalationStage: 2,
      });
      if (this.voice.enabled && this.voice.autoSpeakAlerts) {
        void this.voice.speak(msg, "emergency", false);
      }
      return;
    }

    // Stage 3 (~75s+): Log unresponsive crew & hail Habitat / Flight Surgeon
    if (this.escalationStage === 2 && elapsedSec >= 75) {
      this.escalationStage = 3;
      this.lastStage3PulseMs = now;
      const protoId = activeProto?.id ?? primary?.protocolId ?? "none";
      this.bridge.log_escalation(protoId, 3, `Crew unresponsive >75s on ${primary?.title ?? protoId}`);
      this.bridge.execute_suit_action("hail_flight_surgeon");
      const msg = `No answer yet, Sir. Sending an update to the medical team (12-minute signal delay).`;
      this.voice.fx.playEscalationPulse();
      this.emitCaption({
        speaker: "ASSISTANT",
        text: msg,
        timestampT: live.t,
        alertClass: 1,
        escalationStage: 3,
        unresponsiveEscalated: true,
      });
      if (this.voice.enabled && this.voice.autoSpeakAlerts) {
        void this.voice.speak(msg, "emergency", false);
      }
      return;
    }

    if (this.escalationStage === 3 && now - this.lastStage3PulseMs >= 25000) {
      this.lastStage3PulseMs = now;
      this.voice.fx.playEscalationPulse();
    }
  }

  private buildTacticalCallout(
    primary: AlertObject,
    stepInstruction: string
  ): string {
    const leadMap: Record<string, string> = {
      rad_spe_rate: "Radiation from a solar flare is rising, Sir.",
      rad_dose_cum: "You have reached the long-term radiation limit, Sir.",
      env_pressure_bounds: "Your suit pressure is dropping from a leak, Sir.",
      env_pio2_low: "Oxygen inside your helmet is dropping, Sir.",
      env_ppco2_1h: "Carbon dioxide is building up inside your helmet, Sir.",
      cv_context_hr: "Your heart rate is running high for this activity, Sir.",
      cv_orthostatic: "Your blood pressure looks unsteady after standing up, Sir.",
      cascade_coupled: "More than one suit system needs attention, Sir.",
    };
    const head = leadMap[primary.detectorId] ?? `${primary.title}, Sir.`;
    const action = stepInstruction || primary.suggestedActions[0] || "Check suit status.";
    return `${head} ${action}`;
  }

  private async generateDynamicAlertCallout(
    gen: number,
    primary: AlertObject,
    stepInstruction: string,
    fallbackMsg: string,
    abortSignal: AbortSignal
  ): Promise<void> {
    let spoken = fallbackMsg;
    const timeoutController = new AbortController();
    const onParentAbort = () => timeoutController.abort();
    abortSignal.addEventListener("abort", onParentAbort, { once: true });
    const timer = setTimeout(() => timeoutController.abort(), 1600);

    try {
      const live = this.bridge.get_state();
      const clock = formatMissionClock(live.missionDay, live.timeOfDaySec);
      const alertSummary = `${primary.title} (Level ${primary.class}): ${primary.whyLine}`;
      const promptMsg: ChatMessage = {
        role: "user",
        content:
          `[LIVE ALERT at ${clock}] ${alertSummary}. Guide Sir toward: "${stepInstruction || primary.suggestedActions[0] || "Check suit"}". ` +
          `Give Sir an immediate 1-2 sentence spoken callout (under 25 words): plain everyday English, action first, calm and helpful. Never mention queues, routing, waiting items, step numbers, or say "Next:". No em dashes.`,
      };
      const { message } = await this.client.chatCompletion(
        [this.history[0], ...this.history.slice(-4), promptMsg],
        [],
        timeoutController.signal
      );
      if (abortSignal.aborted || gen !== this.alertSpeechGen) return;
      const aiText = message.content?.trim();
      if (aiText && aiText.length >= 8) {
        spoken = this.sanitizeReply(aiText);
        this.history = [
          this.history[0],
          ...this.history.slice(-6),
          { role: "assistant", content: spoken },
        ];
      }
      this.emitCaption({ cloudStatus: "ready" });
    } catch {
      if (abortSignal.aborted || gen !== this.alertSpeechGen) return;
      spoken = this.sanitizeReply(fallbackMsg);
    } finally {
      clearTimeout(timer);
      abortSignal.removeEventListener("abort", onParentAbort);
    }

    if (abortSignal.aborted || gen !== this.alertSpeechGen) return;
    spoken = this.sanitizeReply(spoken);
    this.emitCaption({
      speaker: "ASSISTANT",
      text: spoken,
      timestampT: this.bridge.get_state().t,
      alertClass: primary.class,
      escalationStage: 0,
      unresponsiveEscalated: false,
    });
    if (this.voice.enabled && this.voice.autoSpeakAlerts) {
      void this.voice.speak(spoken, classToUrgency(primary.class), false);
    }
  }

  /**
   * Handles newly raised alerts deterministically (< 2 s per ENG-DET-008):
   * starts/preempts top protocol, queues all concurrent secondary protocols,
   * and NEVER speaks unprompted when nominal.
   */
  public onAlertsUpdated(alerts: AlertObject[]): void {
    const activeIds = new Set(alerts.map((a) => a.id));
    const activeDetectors = new Set(alerts.map((a) => a.detectorId));
    for (const oldId of this.lastAnnouncedAlertIds) {
      if (!activeIds.has(oldId)) {
        this.lastAnnouncedAlertIds.delete(oldId);
      }
    }
    for (const det of Array.from(this.announcedDetectorClass.keys())) {
      if (!activeDetectors.has(det)) {
        this.announcedDetectorClass.delete(det);
      }
    }

    if (alerts.length === 0) {
      this.alertSpeechGen++;
      if (this.activeAlertAbort) {
        try {
          this.activeAlertAbort.abort();
        } catch {
          // ignore
        }
        this.activeAlertAbort = null;
      }
      this.voice.clearPendingQueue();
      this.escalationStage = 0;
      if (this.caption.unresponsiveEscalated || this.caption.alertClass !== 0) {
        const live = this.bridge.get_state();
        this.emitCaption({
          alertClass: 0,
          escalationStage: 0,
          unresponsiveEscalated: false,
          speaker: "ASSISTANT",
          text: "Everything is back to normal, Sir.",
          timestampT: live.t,
        });
      }
      return;
    }

    // Queue EVERY distinct concurrent alert protocol cleanly so dual hazards never vanish
    const distinctProtocols: ProtocolId[] = [];
    for (const alr of alerts) {
      if (alr.protocolId && !distinctProtocols.includes(alr.protocolId)) {
        distinctProtocols.push(alr.protocolId);
      }
    }

    const primary = alerts[0];
    const nowWall = Date.now();
    const prevDetClass = this.announcedDetectorClass.get(primary.detectorId);
    const lastDetAt = this.lastDetectorAnnouncedAtMs.get(primary.detectorId) ?? 0;
    const escalatedToEmergency =
      prevDetClass !== undefined && prevDetClass > primary.class && primary.class === 1 && nowWall - lastDetAt >= 8000;
    const isUnannouncedPrimary =
      (!this.lastAnnouncedAlertIds.has(primary.id) && prevDetClass === undefined) || escalatedToEmergency;

    if (!isUnannouncedPrimary) {
      for (const pid of distinctProtocols) {
        this.bridge.queue_protocol(pid, true);
      }
      this.lastAnnouncedAlertIds.add(primary.id);
      return;
    }

    // Record ALL currently active alerts so dismantling cascade_coupled doesn't re-announce already-queued alerts
    for (const alr of alerts) {
      this.lastAnnouncedAlertIds.add(alr.id);
      const existingCls = this.announcedDetectorClass.get(alr.detectorId);
      if (existingCls === undefined || alr.class < existingCls) {
        this.announcedDetectorClass.set(alr.detectorId, alr.class);
      }
      this.lastDetectorAnnouncedAtMs.set(alr.detectorId, nowWall);
    }

    // If this alert arrived synchronously during confirmCurrentProtocolStep, let confirmCurrentProtocolStep speak cleanly once!
    if (this.suppressAlertSync) {
      for (const pid of distinctProtocols) {
        this.bridge.queue_protocol(pid, true);
      }
      return;
    }

    const liveBefore = this.bridge.get_state();
    let stepInstruction = "";
    if (primary.protocolId) {
      const currentProto = liveBefore.activeProtocol;
      const currentProtoClass = currentProto
        ? Math.min(
            3,
            ...(alerts.filter((a) => a.protocolId === currentProto.id).map((a) => a.class) || [3])
          )
        : 99;
      const shouldStartOrPreempt =
        !currentProto ||
        currentProto.completed ||
        (currentProto.id !== primary.protocolId && primary.class < currentProtoClass);

      if (shouldStartOrPreempt) {
        const started = this.bridge.start_protocol(primary.protocolId, true);
        stepInstruction = started.currentStep.instruction;
      } else {
        stepInstruction = currentProto.currentStep.instruction;
      }
    }

    for (const pid of distinctProtocols) {
      this.bridge.queue_protocol(pid, true);
    }

    const fallbackMsg = this.buildTacticalCallout(primary, stepInstruction);

    const live = this.bridge.get_state();
    this.bridge.log_activity("activity_start", { event: "alert_announced", alertId: primary.id });
    this.hazardUnconfirmedSinceMs = nowWall;
    this.lastAiBusyAtMs = nowWall;
    this.escalationStage = 0;

    if (this.voice.enabled) {
      this.voice.fx.playAlertTone(primary.class);
    }

    const useLiveLlm =
      this.caption.cloudStatus !== "fallback" &&
      !this.client.getBaseUrl().includes("127.0.0.1:0");

    if (!useLiveLlm) {
      this.emitCaption({
        speaker: "ASSISTANT",
        text: fallbackMsg,
        timestampT: live.t,
        alertClass: primary.class,
        escalationStage: 0,
        unresponsiveEscalated: false,
      });
      if (this.voice.enabled && this.voice.autoSpeakAlerts) {
        void this.voice.speak(fallbackMsg, classToUrgency(primary.class), false);
      }
      return;
    }

    // Live AI proxy mode: update alert severity banner now and emit one clean spoken callout once ready
    this.emitCaption({
      alertClass: primary.class,
      escalationStage: 0,
      unresponsiveEscalated: false,
    });
    if (this.activeAlertAbort) {
      try {
        this.activeAlertAbort.abort();
      } catch {
        // ignore
      }
    }
    const alertAbort = new AbortController();
    this.activeAlertAbort = alertAbort;
    const gen = ++this.alertSpeechGen;
    void this.generateDynamicAlertCallout(gen, primary, stepInstruction, fallbackMsg, alertAbort.signal);
  }

  public confirmCurrentProtocolStep(): void {
    this.markCrewResponsive();
    this.cancelPendingAiWork(true);

    const live = this.bridge.get_state();
    const proto = live.activeProtocol;
    if (!proto || proto.completed) return;

    const confirmedStep = proto.currentStepIndex;
    const confirmedProtoId = proto.id;

    let updated: ReturnType<AssistantEngineBridge["complete_step"]> = null;
    this.suppressAlertSync = true;
    try {
      updated = this.bridge.complete_step(confirmedProtoId, confirmedStep);
    } finally {
      this.suppressAlertSync = false;
    }
    if (!updated) return;

    this.voice.fx.playStepConfirm();

    const postState = this.bridge.get_state();
    let reply: string;
    if (updated.completed && postState.activeProtocol && !postState.activeProtocol.completed && postState.activeProtocol.id !== updated.id) {
      const resumed = postState.activeProtocol;
      reply = `${updated.name} is handled, Sir. Now, ${resumed.currentStep.instruction}`;
    } else if (updated.completed) {
      reply = `That alert is clear, Sir.`;
    } else {
      reply = `Got it, Sir. ${updated.currentStep.instruction}`;
    }

    this.hazardUnconfirmedSinceMs = Date.now();
    this.emitCaption({
      speaker: "ASSISTANT",
      text: reply,
      timestampT: postState.t,
      alertClass: postState.overallClass,
      isBusy: false,
      escalationStage: 0,
      unresponsiveEscalated: false,
    });
    if (this.voice.enabled) {
      void this.voice.speak(reply, classToUrgency(postState.overallClass), false);
    }
  }

  public async ask(question: string): Promise<string> {
    const trimmed = question.trim();
    if (!trimmed) return "";

    this.markCrewResponsive();
    if (this.voice.enabled) {
      this.voice.unlockAudio();
    }

    const live = this.bridge.get_state();
    if (!/[a-z0-9]/i.test(trimmed)) {
      const unclearReply = "I did not catch that, Sir. How can I help?";
      this.emitCaption({
        speaker: "ASSISTANT",
        text: unclearReply,
        timestampT: live.t,
        alertClass: live.overallClass,
        isBusy: false,
      });
      return unclearReply;
    }

    this.cancelPendingAiWork(true);
    const myAskSeq = this.askSeq;
    const abortController = new AbortController();
    this.activeAskAbort = abortController;

    this.emitCaption({
      speaker: "CREW",
      text: trimmed,
      interimTranscript: "",
      timestampT: live.t,
      isBusy: true,
    });

    // Fast "repeat that" command without touching protocol state
    if (/\b(repeat that|say again|repeat|what did you say|pardon|come again)\b/i.test(trimmed)) {
      const repeated = this.sanitizeReply(this.lastSpokenReply || "All suit systems look good, Sir.");
      this.emitCaption({
        speaker: "ASSISTANT",
        text: repeated,
        timestampT: this.bridge.get_state().t,
        alertClass: this.bridge.get_state().overallClass,
        isBusy: false,
      });
      if (this.voice.enabled) {
        void this.voice.speak(repeated, classToUrgency(this.bridge.get_state().overallClass), false);
      }
      if (this.activeAskAbort === abortController) this.activeAskAbort = null;
      return repeated;
    }

    // Fast silence/mute alarm command
    if (
      /\b(mute|silence|quiet|shut up|turn off|stop)\s+(the\s+)?(alarm|alarms|beep|beeping|alerts|siren|noise|it)\b|\b(mute alarm|silence alarm|stop alarm|quiet alarm|silence alerts)\b/i.test(
        trimmed
      )
    ) {
      this.bridge.execute_suit_action("silence_alarm");
      const silenceReply = "Alarms quieted, Sir.";
      this.emitCaption({
        speaker: "ASSISTANT",
        text: silenceReply,
        timestampT: this.bridge.get_state().t,
        alertClass: this.bridge.get_state().overallClass,
        isBusy: false,
      });
      if (this.activeAskAbort === abortController) this.activeAskAbort = null;
      return silenceReply;
    }

    // Fast stop-talking voice/text command
    if (/^(stop talking|shut up|be quiet|quiet please|silence)\b/i.test(trimmed)) {
      const quietReply = "Understood, Sir. Staying quiet.";
      this.emitCaption({
        speaker: "ASSISTANT",
        text: quietReply,
        timestampT: this.bridge.get_state().t,
        alertClass: this.bridge.get_state().overallClass,
        isBusy: false,
      });
      if (this.activeAskAbort === abortController) this.activeAskAbort = null;
      return quietReply;
    }

    // Fast voice/text confirmation shortcut (strictly guard against queries, checks, or Plan B requests)
    const normalizedCmd = trimmed
      .toLowerCase()
      .replace(/[^a-z0-9'\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    const wordCount = normalizedCmd ? normalizedCmd.split(" ").length : 0;
    const hasObstacleOrQuestion =
      trimmed.endsWith("?") ||
      /^(what|why|how|when|where|who|which|tell|show|explain|check|read|give|list|is|are|do|does|can|could|would|should|switch|use|try|no|nope|nah|wait)\b/i.test(
        normalizedCmd
      ) ||
      /\b(can't|cannot|won't|wont|not|no|unable|impossible|far|stuck|trapped|blocked|broken|dead|jammed|hurt|dizzy|faint|pain|wait|why|how|what|when|where|which|didn't|didnt|doesn't|doesnt|plan b|plan c|backup plan|other way|instead|alternative|still|check)\b/i.test(
        normalizedCmd
      );
    const isShortAck =
      wordCount <= 4 &&
      /\b(done|confirm|confirmed|yes|yeah|yep|ok|okay|roger|copy|affirmative|got it|understood|all set|finished|completed|complete)\b/i.test(
        normalizedCmd
      );
    const isActionCompletePhrase =
      /\b(made it|inside|in cover|in shelter|behind cover|at the habitat|in the airlock|patched|fixed|sealed|locked|braced|resting|settled|stabilized|in position|all done|step done|step complete|step completed|i am okay|i'm okay)\b/i.test(
        normalizedCmd
      );

    if (
      !hasObstacleOrQuestion &&
      live.activeProtocol &&
      !live.activeProtocol.completed &&
      (isShortAck || isActionCompletePhrase)
    ) {
      if (this.activeAskAbort === abortController) this.activeAskAbort = null;
      this.confirmCurrentProtocolStep();
      this.emitCaption({ isBusy: false });
      return this.caption.text;
    }

    if (
      !hasObstacleOrQuestion &&
      !live.activeProtocol &&
      live.rankedPrimaryAlert &&
      (isShortAck || /\b(acknowledge|acknowledged)\b/i.test(normalizedCmd))
    ) {
      if (this.activeAskAbort === abortController) this.activeAskAbort = null;
      this.bridge.acknowledge(live.rankedPrimaryAlert.id);
      this.voice.fx.playStepConfirm();
      const ackReply = `Got it, Sir. Keeping an eye on that for you.`;
      this.emitCaption({
        speaker: "ASSISTANT",
        text: ackReply,
        timestampT: this.bridge.get_state().t,
        alertClass: this.bridge.get_state().overallClass,
        isBusy: false,
      });
      if (this.voice.enabled) {
        void this.voice.speak(ackReply, classToUrgency(this.bridge.get_state().overallClass), false);
      }
      return ackReply;
    }

    let reply = "";
    try {
      reply = await this.runAzureToolLoop(trimmed, myAskSeq, abortController.signal);
      if (abortController.signal.aborted || myAskSeq !== this.askSeq) {
        return "";
      }
      this.emitCaption({ cloudStatus: "ready" });
    } catch {
      if (abortController.signal.aborted || myAskSeq !== this.askSeq) {
        return "";
      }
      reply = this.buildDeterministicFallback(trimmed);
      this.emitCaption({ cloudStatus: "fallback" });
    } finally {
      if (this.activeAskAbort === abortController) {
        this.activeAskAbort = null;
      }
    }

    if (abortController.signal.aborted || myAskSeq !== this.askSeq) {
      return "";
    }

    reply = this.sanitizeReply(reply);
    const post = this.bridge.get_state();
    this.emitCaption({
      speaker: "ASSISTANT",
      text: reply,
      timestampT: post.t,
      alertClass: post.overallClass,
      isBusy: false,
    });
    if (this.voice.enabled) {
      void this.voice.speak(reply, classToUrgency(post.overallClass), false);
    }
    return reply;
  }

  private async runAzureToolLoop(
    userText: string,
    myAskSeq: number,
    signal: AbortSignal
  ): Promise<string> {
    const messages: ChatMessage[] = [
      ...this.history.slice(-10),
      { role: "user", content: userText },
    ];

    for (let round = 0; round < 4; round++) {
      if (signal.aborted || myAskSeq !== this.askSeq) {
        throw new Error("Request cancelled");
      }
      const { message } = await this.client.chatCompletion(messages, ASSISTANT_TOOLS, signal);
      if (signal.aborted || myAskSeq !== this.askSeq) {
        throw new Error("Request cancelled");
      }
      messages.push(message);

      if (!message.tool_calls || message.tool_calls.length === 0) {
        const finalContent = this.sanitizeReply(
          message.content?.trim() || this.buildDeterministicFallback(userText)
        );
        this.history = [this.history[0], ...messages.slice(-8)];
        return finalContent;
      }

      const orderedCalls = [...message.tool_calls].sort((a, b) => {
        const rank = (n: string) => (n === "update_action_plan" ? 0 : n === "execute_suit_action" ? 1 : 2);
        return rank(a.function.name) - rank(b.function.name);
      });
      for (const tc of orderedCalls) {
        if (signal.aborted || myAskSeq !== this.askSeq) {
          throw new Error("Request cancelled");
        }
        const result = this.executeTool(tc.function.name, tc.function.arguments);
        messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: JSON.stringify(result),
        });
      }
    }

    if (signal.aborted || myAskSeq !== this.askSeq) {
      throw new Error("Request cancelled");
    }
    return this.buildDeterministicFallback(userText);
  }

  private executeTool(name: string, rawArgs: string): unknown {
    let args: Record<string, any> = {};
    try {
      args = rawArgs ? JSON.parse(rawArgs) : {};
    } catch {
      args = {};
    }

    switch (name) {
      case "get_state": {
        const s = this.bridge.get_state();
        return {
          missionDay: s.missionDay,
          clock: formatMissionClock(s.missionDay, s.timeOfDaySec),
          context: s.context,
          overallClass: s.overallClass,
          careerDosePercent: s.careerDosePercent,
          speEventDoseMsv: s.speEventDoseMsv,
          currentSafetyStep:
            s.activeProtocol && !s.activeProtocol.completed
              ? {
                  id: s.activeProtocol.id,
                  title: s.activeProtocol.name,
                  stepIndex: s.activeProtocol.currentStepIndex,
                  instruction: s.activeProtocol.currentStep.instruction,
                  isBackupPlan: Boolean(s.activeProtocol.contingencyTitle),
                }
              : null,
          otherActiveIssues: s.activeAlerts.slice(1).map((a) => ({
            title: a.title,
            severity: a.class,
          })),
          activeAlerts: s.activeAlerts,
          primaryVitals: {
            dose_rate: s.indicators.dose_rate,
            dose_cumulative: s.indicators.dose_cumulative,
            ppco2: s.indicators.ppco2,
            pio2: s.indicators.pio2,
            pressure:
              s.context.activity === "eva"
                ? s.indicators.suit_pressure
                : s.indicators.cabin_pressure,
            heart_rate: s.indicators.heart_rate,
            hrv_rmssd: s.indicators.hrv_rmssd,
            sleep_hours: s.indicators.sleep_hours,
            bone_loss_lumbar: s.indicators.bone_loss_lumbar,
            immune_index: s.indicators.immune_index,
            cognitive_drift: s.indicators.cognitive_drift,
            sans_risk: s.indicators.sans_risk,
          },
        };
      }
      case "summarise":
        return this.bridge.summarise(
          args.indicator as IndicatorId,
          Number(args.windowMinutes ?? 60)
        );
      case "get_alerts":
        return this.bridge.get_alerts("active");
      case "explain":
        return this.bridge.explain(String(args.alertId || ""));
      case "compare_to_plan":
        return this.bridge.compare_to_plan(args.metric);
      case "start_protocol":
        return this.bridge.start_protocol(args.name as ProtocolId);
      case "complete_step": {
        const proto = args.protocol as ProtocolId;
        const step = Number(args.step ?? 0);
        let res: ReturnType<AssistantEngineBridge["complete_step"]> = null;
        this.suppressAlertSync = true;
        try {
          res = this.bridge.complete_step(proto, step);
        } finally {
          this.suppressAlertSync = false;
        }
        if (res) {
          this.markCrewResponsive();
          this.hazardUnconfirmedSinceMs = Date.now();
          this.escalationStage = 0;
          this.voice.fx.playStepConfirm();
        }
        return res;
      }
      case "acknowledge":
        return this.bridge.acknowledge(String(args.alertId || ""));
      case "lookup_manual":
        return {
          topic: String(args.topic || "all_limits"),
          content: lookupManualTopic(String(args.topic || "all_limits")),
        };
      case "update_action_plan": {
        const updated = this.bridge.update_action_plan(
          String(args.title || "PLAN B ADAPTATION"),
          String(args.currentAction || "Execute improvised contingency."),
          args.confirmationPrompt ? String(args.confirmationPrompt) : undefined,
          args.reason ? String(args.reason) : undefined,
          args.protocolId ? (String(args.protocolId) as ProtocolId) : undefined
        );
        this.markCrewResponsive();
        this.hazardUnconfirmedSinceMs = Date.now();
        this.escalationStage = 0;
        this.voice.fx.playStepConfirm();
        return updated;
      }
      case "execute_suit_action": {
        const act = String(args.action || "") as SuitActionName;
        const res = this.bridge.execute_suit_action(act);
        if (res.ok) {
          this.voice.fx.playStepConfirm();
        }
        return res;
      }
      default:
        return { error: `Unknown tool: ${name}` };
    }
  }

  /**
   * Guarantees grounded, adaptive, non-diagnostic replies if offline or before proxy setup (V-11).
   */
  public buildDeterministicFallback(question: string): string {
    const s = this.bridge.get_state();
    const clk = formatMissionClock(s.missionDay, s.timeOfDaySec);
    const trimmed = question.trim();
    const q = trimmed.toLowerCase();

    if (!trimmed) {
      return s.rankedPrimaryAlert
        ? `Still here, Sir. ${s.activeProtocol?.currentStep.instruction ?? s.rankedPrimaryAlert.whyLine}`
        : "Right here, Sir. Tell me what you need.";
    }

    // Physical distress / symptoms: stay warm, non-diagnostic, and prioritize immediate safety
    if (/\b(dizzy|lightheaded|faint|nausea|vomit|throwing up|sick|unwell|headache|chest pain|can't breathe|hard to breathe|scared|panic|afraid|dying|die|hurt|pain|bleeding|injured)\b/.test(q)) {
      const safetyHint = s.activeProtocol
        ? ` Stop moving first and focus on this step: ${s.activeProtocol.currentStep.instruction}`
        : " Pause where you are, lock your boots, and take slow steady breaths.";
      const vitals = `Heart rate is ${Math.round(s.indicators.heart_rate.value)} bpm and helmet oxygen is ${s.indicators.pio2.value.toFixed(0)} mmHg.`;
      return `Stay with me, Sir.${safetyHint} ${vitals} I only read suit sensors and cannot give a medical diagnosis, so let the medical team know if you keep feeling unwell.`;
    }

    // Adaptive offline contingency if crew reports obstacle to Plan A (and escalates Plan B -> Plan C)
    if (
      /\b(can't|cannot|won't|wont|impossible|too far|stuck|trapped|blocked|jammed|broken|no crater|no rover|no cover|flat ground|no tape|no cuff|no patch|no spare|nothing nearby|still leaking|doesn't work|doesnt work|failed|plan b|plan c|not working|didn't work|didnt work|other way|alternative|no other)\b/.test(
        q
      ) &&
      (s.rankedPrimaryAlert || (s.activeProtocol && !s.activeProtocol.completed))
    ) {
      const area =
        s.rankedPrimaryAlert?.area ??
        (s.activeProtocol?.id === "shelter"
          ? "radiation"
          : s.activeProtocol?.id === "air_check"
            ? "environment"
            : "cardiovascular");
      const alreadyPlanB = Boolean(s.activeProtocol?.contingencyTitle?.includes("PLAN B"));
      const isLeakContext =
        s.indicators.suit_pressure.value < 35 ||
        Boolean(s.activeProtocol?.contingencyTitle?.includes("SUIT PATCH")) ||
        s.activeAlerts.some((a) => a.detectorId === "env_pressure_bounds");
      if (area === "radiation") {
        if (alreadyPlanB) {
          this.bridge.update_action_plan(
            "PLAN C: FLAT GROUND SHIELDING",
            "Kneel low facing away from the sun, stack pack equipment behind your backpack, and pull loose dust against your legs.",
            "Low posture holding",
            "No crater wall or rover nearby",
            "shelter"
          );
          this.bridge.execute_suit_action("engage_terrain_shield");
          return "Understood, Sir. Moving to Plan C. Get as low as you can with your backpack facing the sun, put your pack gear behind your back, and hold position until the flare eases.";
        }
        this.bridge.update_action_plan(
          "PLAN B: CRATER OR ROVER COVER",
          "Drop behind the nearest crater wall or rover body, crouch with your backpack facing the sun, and stay low.",
          "In cover and braced",
          "Habitat is too far to reach before the flare peaks",
          "shelter"
        );
        this.bridge.execute_suit_action("engage_terrain_shield");
        return "Understood, Sir. Switching to Plan B. Take cover behind the nearest crater wall or rover body with your backpack facing the sun now. Shielding is active.";
      }
      if (area === "environment" && isLeakContext) {
        if (alreadyPlanB) {
          this.bridge.update_action_plan(
            "PLAN C: PINCH & HOLD STILL",
            "Pinch the torn fabric shut with your glove, lock your boots, and stay completely still so backup oxygen lasts longer.",
            "Pressure holding steady",
            "Cannot apply a full patch right now",
            "air_check"
          );
          this.bridge.execute_suit_action("seal_suit_breach");
          this.bridge.execute_suit_action("switch_secondary_o2");
          return "Understood, Sir. Moving to Plan C. Pinch the tear shut with your glove and stay completely still to stretch your backup oxygen while pressure stabilizes.";
        }
        this.bridge.update_action_plan(
          "PLAN B: SUIT PATCH & BACKUP O2",
          "Clamp the torn suit seam with your emergency cuff or tape. Backup oxygen is now active.",
          "Leak slowed and pressure holding",
          "Too far from the airlock for an immediate return",
          "air_check"
        );
        this.bridge.execute_suit_action("seal_suit_breach");
        this.bridge.execute_suit_action("switch_secondary_o2");
        return "Understood, Sir. Switching to Plan B. Backup oxygen is running. Seal the suit tear with your emergency cuff or tape now and tell me once pressure steadies.";
      }
      if (area === "environment") {
        if (alreadyPlanB) {
          this.bridge.update_action_plan(
            "PLAN C: SLOW BREATHING HOLD",
            "Stop walking, lock your boots, and take slow steady breaths while fresh oxygen flushes your helmet.",
            "Air feels clear",
            "Moving makes helmet air take longer to clear",
            "air_check"
          );
          this.bridge.execute_suit_action("open_loop_o2_purge");
          return "Understood, Sir. Moving to Plan C. Stop moving right now and take slow, steady breaths while fresh oxygen clears your helmet.";
        }
        this.bridge.update_action_plan(
          "PLAN B: FRESH AIR FLUSH",
          "Keep fresh oxygen flushing through your helmet and slow your breathing to save supplies.",
          "Breathing feels clear",
          "Standard scrubber recovery is not enough right now",
          "air_check"
        );
        this.bridge.execute_suit_action("open_loop_o2_purge");
        return "Understood, Sir. Fresh helmet oxygen flush is active. Stop moving and take slow breaths while your air clears.";
      }
      const cvProto: ProtocolId =
        s.activeProtocol?.id === "orthostatic_response" ? "orthostatic_response" : "exertion_recovery";
      if (alreadyPlanB) {
        this.bridge.update_action_plan(
          "PLAN C: FULL STATIC REST",
          "Keep both boots locked, relax your shoulders, and wait silently until heart rate drops.",
          "Resting calmly",
          "Recovery taking longer than expected",
          cvProto
        );
        this.bridge.execute_suit_action("rest_recovery_mode");
        return "Understood, Sir. Moving to Plan C. Stay completely still with your boots locked and focus on slow breathing until your heart rate comes down.";
      }
      this.bridge.update_action_plan(
        "PLAN B: STOP AND REST",
        "Stop moving, lock your boot brakes, breathe slowly, and wait for your heart rate to settle.",
        "Resting and steady",
        "Cannot continue walking to recover",
        cvProto
      );
      this.bridge.execute_suit_action("rest_recovery_mode");
      return "Understood, Sir. Switching to Plan B. Stop moving, lock your boots, and let your heart rate settle.";
    }

    if (/dose|rad|shelter|dangerous|how long|flare|solar/.test(q)) {
      const rate = s.indicators.dose_rate;
      const cum = s.indicators.dose_cumulative;
      return `Sir, at ${clk}, radiation dose rate is ${rate.value.toFixed(2)} mSv/day and total trip dose is ${cum.value.toFixed(1)} mSv (${s.careerDosePercent.toFixed(1)}% of the 600 mSv career limit).`;
    }

    if (/heart|hr|bpm|pulse|exercise/.test(q)) {
      const hr = s.indicators.heart_rate;
      const limit = ENGINE_LIMITS.hrLimitByActivity[s.context.activity].value;
      const status =
        hr.value <= limit
          ? `within the ${limit} bpm limit expected for ${s.context.activity}`
          : `above the ${limit} bpm limit for ${s.context.activity}`;
      return `Sir, at ${clk}, heart rate is ${Math.round(hr.value)} bpm (${hr.kind}), which is ${status}.`;
    }

    if (/co2|air|oxygen|pio2|pressure|breathe|leak/.test(q)) {
      const co2 = s.indicators.ppco2;
      const o2 = s.indicators.pio2;
      const press = s.indicators.suit_pressure;
      return `Sir, at ${clk}, suit pressure is ${press.value.toFixed(1)} kPa, CO2 level is ${co2.value.toFixed(2)} mmHg, and helmet oxygen is ${o2.value.toFixed(0)} mmHg.`;
    }

    if (/bone|immune|sleep|vision|sans|cognitive/.test(q)) {
      const lum = s.indicators.bone_loss_lumbar;
      const imm = s.indicators.immune_index;
      return `Sir, at ${clk}, estimated lumbar bone density is ${lum.value.toFixed(2)}% of pre-flight (${lum.uncertainty?.low} to ${lum.uncertainty?.high}%, modeled estimate) and the demo immune score is ${imm.value.toFixed(0)} out of 100.`;
    }

    if (/^(hi|hello|hey|good morning|good evening|are you there|can you hear me|who are you|test|testing)\b/.test(q)) {
      return s.rankedPrimaryAlert
        ? `Right here, Sir. We still have an active alert: ${s.rankedPrimaryAlert.title}. ${s.activeProtocol?.currentStep.instruction ?? ""}`.trim()
        : `Right here, Sir (${clk}). Your voice channel is clear and all suit systems look good.`;
    }

    if (/\b(else|other|queued|queue|waiting|more|remaining)\b/.test(q)) {
      if (s.pausedProtocols.length > 0 || s.activeAlerts.length > 1) {
        const secondary = s.activeAlerts
          .slice(1, 3)
          .map((a) => a.title)
          .join(", ");
        return `Focus on ${s.rankedPrimaryAlert?.title ?? "your current action"} first, Sir. You also have ${secondary || "another active suit alert"} to handle after this.`;
      }
      if (s.rankedPrimaryAlert) {
        return `Nothing else active right now, Sir. Just focus on ${s.rankedPrimaryAlert.title}: ${s.activeProtocol?.currentStep.instruction ?? "follow the highlighted step."}`;
      }
      return `All quiet besides your main vitals, Sir (${clk}). Suit systems look good.`;
    }

    if (/\b(what should i do|what is happening|what's happening|why|help|status|report)\b/.test(q)) {
      if (s.rankedPrimaryAlert) {
        const stepHint = s.activeProtocol
          ? ` ${s.activeProtocol.currentStep.instruction}`
          : "";
        return `Sir, ${s.rankedPrimaryAlert.whyLine}.${stepHint}`;
      }
      const hr = s.indicators.heart_rate;
      const pio2 = s.indicators.pio2;
      const press = s.indicators.suit_pressure;
      return `Everything looks good right now, Sir (${clk}). Heart rate is ${Math.round(hr.value)} bpm, suit pressure is ${press.value.toFixed(1)} kPa, and helmet oxygen is ${pio2.value.toFixed(0)} mmHg.`;
    }

    if (s.rankedPrimaryAlert) {
      const alr = s.rankedPrimaryAlert;
      const stepHint = s.activeProtocol
        ? ` ${s.activeProtocol.currentStep.instruction}`
        : "";
      return `Sir, ${alr.whyLine}.${stepHint}`;
    }

    const hr = s.indicators.heart_rate;
    const pio2 = s.indicators.pio2;
    const press = s.indicators.suit_pressure;
    return `All suit systems look good, Sir (${clk}). Heart rate is ${Math.round(hr.value)} bpm, suit pressure is ${press.value.toFixed(1)} kPa, and helmet oxygen is ${pio2.value.toFixed(0)} mmHg.`;
  }
}
