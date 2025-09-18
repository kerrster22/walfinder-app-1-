// lib/dangerNLG.ts
export type Danger = "none" | "low" | "med" | "high";
export type Box = { x: number; y: number; w: number; h: number };
export type Det = {
  class: string;
  conf: number;
  box: Box;
  danger: Danger;
  count?: number; // for "team"
};

function dirWord(b: Box): string {
  const cx = b.x + b.w / 2;
  if (cx < 0.15) return "far left";
  if (cx < 0.32) return "left";
  if (cx < 0.48) return "slightly left";
  if (cx <= 0.52) return "center";
  if (cx <= 0.68) return "slightly right";
  if (cx <= 0.85) return "right";
  return "far right";
}

function proxWord(b: Box): string {
  const a = b.w * b.h; // proxy for distance
  if (a > 0.32) return "immediately ahead";
  if (a > 0.18) return "very close";
  if (a > 0.09) return "nearby";
  return "ahead";
}

function niceLabel(cls: string, count?: number): string {
  if (cls === "team") return `a group of ${count ?? "several"} people`;
  return cls.replace(/_/g, " ");
}

function leadFor(danger: Danger | "team") {
  if (danger === "high" || danger === "team") return "Alert";
  if (danger === "med") return "Caution";
  return "Notice";
}

export function describeThreat(d: Det): string {
  const kind = d.class === "team" ? ("team" as const) : d.danger;
  const lead = leadFor(kind);
  const where = dirWord(d.box);
  const prox = proxWord(d.box);
  const what = niceLabel(d.class, d.count);
  const tail =
    d.class === "team" || d.danger === "high"
      ? "Please proceed with caution."
      : "Be careful.";
  // Lightly varied, natural cadence with short sentences reads better on Web Speech.
  return `${lead}: ${what} ${where}, ${prox}. ${tail}`;
}

export function summarizeEvent(dets: Det[]): string | null {
  if (!dets?.length) return null;
  dets.sort((a, b) => {
    const rank = (x: Danger) => (x === "high" ? 3 : x === "med" ? 2 : x === "low" ? 1 : 0);
    const ra = a.class === "team" ? 3 : rank(a.danger);
    const rb = b.class === "team" ? 3 : rank(b.danger);
    const r = rb - ra;
    if (r) return r;
    return b.box.w * b.box.h - a.box.w * a.box.h;
  });
  const top = dets.slice(0, 2);
  if (top.length === 1) return describeThreat(top[0]);
  return `${describeThreat(top[0])} ${describeThreat(top[1])}`;
}

/** Coarse signature to avoid retriggers from tiny motion (12% bins). */
export function coarseSignature(dets: Det[]): string {
  const q = (v: number) => Math.round(v / 0.12); // bigger bins = fewer false "changes"
  const parts = dets
    .map((d) => {
      const bx = d.box;
      const cx = bx.x + bx.w / 2;
      const cy = bx.y + bx.h / 2;
      return [d.class, d.danger, q(cx), q(cy), q(bx.w), q(bx.h)].join(":");
    })
    .sort();
  return parts.join("|");
}
