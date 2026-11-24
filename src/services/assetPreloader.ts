import type { AzureProxyClient, ProxyHealth } from "../assistant/azureClient";

export interface PreloadItemStatus {
  id: string;
  label: string;
  status: "pending" | "loading" | "ready" | "degraded";
  detail?: string;
}

export interface PreloadSnapshot {
  progressPct: number;
  ready: boolean;
  items: PreloadItemStatus[];
  cloudHealth: ProxyHealth | null;
}

/**
 * Decoded HTMLImageElement cache populated before VisorScene mounts so
 * Three.js creates panoramic GPU textures synchronously with zero pop-in.
 */
export const PRELOADED_IMAGES = new Map<string, HTMLImageElement>();

function loadDecodedImage(url: string): Promise<HTMLImageElement> {
  const cached = PRELOADED_IMAGES.get(url);
  if (cached && cached.complete && cached.naturalWidth > 0) {
    return Promise.resolve(cached);
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.onload = async () => {
      try {
        if (typeof img.decode === "function") {
          await img.decode();
        }
      } catch {
        // Ignore decode rejection if already rasterizable
      }
      PRELOADED_IMAGES.set(url, img);
      resolve(img);
    };
    img.onerror = () => reject(new Error(`Failed to preload ${url}`));
    img.src = url;
  });
}

/**
 * Loads all visual assets, fonts, and AI/voice service health checks
 * before revealing the 3D EVA visor scene or starting simulation telemetry.
 */
export async function runMissionAssetPreload(
  azure: AzureProxyClient,
  onProgress: (snapshot: PreloadSnapshot) => void
): Promise<PreloadSnapshot> {
  const base = import.meta.env.BASE_URL;
  const assetList: Array<{ id: string; label: string; url: string }> = [
    { id: "sky", label: "Martian Sky Dome Panorama (Jezero Crater)", url: `${base}mars/mars-sky.webp` },
    { id: "terrain", label: "Perseverance Surface Terrain Photogrammetry", url: `${base}mars/mars-terrain.webp` },
    { id: "depth", label: "Parallax Depth Map & Shadow Relief", url: `${base}mars/mars-depth.png` },
  ];

  const items: PreloadItemStatus[] = [
    ...assetList.map((a) => ({ id: a.id, label: a.label, status: "pending" as const })),
    { id: "fonts", label: "HUD Typography & Visor Glyphs", status: "pending" as const },
    { id: "cloud", label: "On-Board AI & Neural Suit Voice Link", status: "pending" as const },
  ];

  let completedSteps = 0;
  const totalSteps = items.length;
  let cloudHealth: ProxyHealth | null = null;

  const emit = (ready = false) => {
    onProgress({
      progressPct: Math.min(100, Math.round((completedSteps / totalSteps) * 100)),
      ready,
      items: items.map((x) => ({ ...x })),
      cloudHealth,
    });
  };

  emit(false);

  // Load scene image assets concurrently while updating per-item progress
  const imageJobs = assetList.map(async (asset, idx) => {
    items[idx].status = "loading";
    emit(false);
    try {
      await loadDecodedImage(asset.url);
      items[idx].status = "ready";
      items[idx].detail = "Cached";
    } catch {
      items[idx].status = "degraded";
      items[idx].detail = "Fallback stream";
    } finally {
      completedSteps += 1;
      emit(false);
    }
  });

  const fontJob = (async () => {
    const item = items[3];
    item.status = "loading";
    emit(false);
    try {
      if (typeof document !== "undefined" && document.fonts?.ready) {
        await Promise.race([
          document.fonts.ready,
          new Promise<void>((res) => setTimeout(res, 2200)),
        ]);
      }
      item.status = "ready";
      item.detail = "Loaded";
    } catch {
      item.status = "degraded";
      item.detail = "System font fallback";
    } finally {
      completedSteps += 1;
      emit(false);
    }
  })();

  const cloudJob = (async () => {
    const item = items[4];
    item.status = "loading";
    emit(false);
    try {
      cloudHealth = await Promise.race([
        azure.checkHealth(),
        new Promise<ProxyHealth>((res) =>
          setTimeout(() => res({ ok: false }), 2600)
        ),
      ]);
      if (cloudHealth.ok) {
        item.status = "ready";
        item.detail = "Cloud AI + Neural Voice";
      } else {
        item.status = "degraded";
        item.detail = "Local Deterministic Engine";
      }
    } catch {
      item.status = "degraded";
      item.detail = "Local Deterministic Engine";
    } finally {
      completedSteps += 1;
      emit(false);
    }
  })();

  await Promise.all([...imageJobs, fontJob, cloudJob]);

  const finalSnapshot: PreloadSnapshot = {
    progressPct: 100,
    ready: true,
    items: items.map((x) => ({ ...x })),
    cloudHealth,
  };
  onProgress(finalSnapshot);
  return finalSnapshot;
}
