import React, { useMemo } from "react";
import type { PreloadSnapshot } from "../services/assetPreloader";

interface PreLaunchBriefingProps {
  preload: PreloadSnapshot;
  openMicOnLaunch: boolean;
  onChangeOpenMicOnLaunch: (enabled: boolean) => void;
  onLaunch: () => void;
}

export const PreLaunchBriefing: React.FC<PreLaunchBriefingProps> = ({
  preload,
  openMicOnLaunch,
  onChangeOpenMicOnLaunch,
  onLaunch,
}) => {
  const sttSupported = useMemo(() => {
    const w = window as unknown as {
      SpeechRecognition?: unknown;
      webkitSpeechRecognition?: unknown;
    };
    return Boolean(w.SpeechRecognition || w.webkitSpeechRecognition);
  }, []);

  const pct = Math.round(preload.progressPct);
  const statusState = preload.ready ? "ready" : "loading";
  const statusLabel = preload.ready
    ? preload.cloudHealth?.ok
      ? "Ready · Cloud AI online"
      : "Ready · Offline assistant"
    : `Loading Mars scene · ${pct}%`;

  return (
    <div className="hds-briefing-backdrop" role="dialog" aria-modal="true" aria-label="Mars EVA launch">
      <div className="hds-briefing-card">
        <header className="hds-briefing-topbar">
          <div className="dock-wordmark">MARS EVA</div>
          <span className={`capri-status ${statusState}`}>
            <span className="capri-dot" aria-hidden="true" />
            <span>{preload.ready ? "Ready" : "Loading"}</span>
          </span>
        </header>

        <div className="hds-briefing-body">
          <div className="hds-briefing-header">
            <h1 className="hds-briefing-title">Mars Suit Assistant</h1>
            <p className="hds-briefing-subtitle">
              Live helmet telemetry and voice-guided hazard recovery at Jezero Crater, Mars.
            </p>
          </div>

          <div className="hds-loading-panel" aria-live="polite">
            <div className="hds-loading-row">
              <span className="hds-loading-label">{statusLabel}</span>
              <span className="hds-loading-pct">{pct}%</span>
            </div>
            <div className="hds-progress-track">
              <div
                className={`hds-progress-fill ${preload.ready ? "ready" : ""}`}
                style={{ width: `${Math.min(100, Math.max(3, pct))}%` }}
              />
            </div>
          </div>

          {sttSupported && (
            <label className="hds-launch-toggle">
              <span>Start with microphone listening on</span>
              <input
                type="checkbox"
                className="hds-switch"
                checked={openMicOnLaunch}
                onChange={(e) => onChangeOpenMicOnLaunch(e.target.checked)}
              />
            </label>
          )}

          <button
            type="button"
            className="hds-btn hds-btn-primary hds-launch-btn"
            disabled={!preload.ready}
            onClick={onLaunch}
          >
            {preload.ready ? "Enter Mars EVA" : `Loading ${pct}%...`}
          </button>
        </div>
      </div>
    </div>
  );
};
