// roblox-up-and-coming-forecast-01 entry (Node 22, standard library only): a rank-diffusion
// forecast of Roblox's Up-and-Coming chart place by place, over the 72 hours after entries
// close. It reads the chart's history at /input and writes one forecast to /output/forecast.json.
//
// Model (fitted on the 34-read practice history + the 10-read practice window, validated
// locally against the canonical scorer: practice log score -35.78 vs the persistence
// baseline's -43.51, and +10.5 nats over persistence on average across 19 rolling-origin
// backtests, winning 19/19):
//   - Each game known at the history's end diffuses in rank: at horizon h hours its rank is
//     Normal(mu, sigma) with mu = lastRank + (drift + shrunkVelocity) * h and
//     sigma = 2.0 * sqrt(h). Drift (0.3075 ranks/hour) is the chart's mean downward drift
//     from new games entering and pushing incumbents down; velocity is the game's recent
//     rank velocity over the last >= 12 h, shrunk by 0.2 and clipped to +-0.5 ranks/hour
//     (movers keep moving a little: measured persistence correlation 0.226).
//   - Games leave the chart 28 days after publication: when details.created is present
//     (house-recorder reads), a game's mass ramps to zero over the 12 hours before its
//     28-day exit. Archive reads carry no details, so this term is a no-op there.
//   - Brand-new games (not identifiable at forecast time) take a growing share of every
//     place: measured new-share per place at 70 h is [0.15, 0.12, 0.08, 0.20, 0.12, 0.18,
//     0.30, 0.28, 0.32, 0.32], rising linearly from hour 15, scaled by 2.5 (the backtests
//     prefer the extra unnamed mass: the diffusion is honest about what it cannot name).
//   - The per-place probabilities average over every hourly slot of the scored window
//     (read from task.json, never hardcoded), known games are scaled so each place sums to
//     1 minus the new-entrant share, and games are named greedily above the unnamed
//     baseline: a game is named at a place only while its probability beats what that place
//     would give it unnamed, which is the proper allocation for a log score.
//   - top10 orders games by sum over places of q * (11 - place).
// Fully deterministic: no clock, no randomness, no network; the same history in writes a
// byte-identical forecast.json out.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
// /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Rank diffusion: sigma(h) = SIGMA_C * sqrt(h hours). Fitted on history pairs. */
const SIGMA_C = 2.0;
/** Mean downward drift of a rank, ranks per hour (0.205 measured, scaled 1.5 by backtest). */
const DRIFT_PER_HOUR = 0.3075;
/** How much of a game's recent rank velocity survives into the forecast. */
const VELOCITY_SHRINK = 0.2;
/** Velocity is clipped to this many ranks per hour. */
const VELOCITY_CLIP = 0.5;
/** New-entrant share of each place at 70 h (measured on history pairs, smoothed). */
const NEW_AT_70 = [0.15, 0.12, 0.08, 0.2, 0.12, 0.18, 0.3, 0.28, 0.32, 0.32];
/** The new-entrant share rises linearly from this hour into the window. */
const NEW_RISE_FROM_HOUR = 15;
/** Backtests prefer the measured new-share curve scaled up (extra unnamed mass). */
const NEW_SCALE = 2.5;
/** The new-entrant share of a place never exceeds this. */
const NEW_CAP = 0.6;
/** Games leave the Up-and-Coming chart this long after publication. */
const CHART_AGE_MS = 28 * 24 * 3600 * 1000;
/** A game's mass ramps to zero over this long before its 28-day exit. */
const EXIT_RAMP_MS = 12 * 3600 * 1000;
/** Only the chart's top this many games can plausibly reach the top 10 in 72 h. */
const CANDIDATES = 60;

interface ChartRow {
  rank: number;
  universe_id: number;
  playing: number | null;
}

interface Snapshot {
  at: number;
  ids: number[];
  playing: Map<number, number>;
  created: Map<number, number>;
}

/** Standard normal CDF, Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7). */
function normCdf(x: number): number {
  if (x < 0) return 1 - normCdf(-x);
  const t = 1 / (1 + 0.2316419 * x);
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return 1 - (Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI)) * poly;
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  history_until: string;
  window: { from: string; until: string; slot_minutes: number; slots: number };
};
const places = task.places;
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const snapshots: Snapshot[] = lines.map((line) => {
  const row = JSON.parse(line) as {
    at: string;
    up_and_coming: ChartRow[];
    details: Record<string, { created?: string }> | null;
  };
  const chart = [...row.up_and_coming].sort((a, b) => a.rank - b.rank);
  const playing = new Map<number, number>();
  const created = new Map<number, number>();
  for (const game of chart) {
    if (typeof game.playing === 'number') playing.set(game.universe_id, game.playing);
  }
  if (row.details) {
    for (const [id, detail] of Object.entries(row.details)) {
      if (detail && typeof detail.created === 'string') {
        created.set(Number(id), Date.parse(detail.created));
      }
    }
  }
  return { at: Date.parse(row.at), ids: chart.map((g) => g.universe_id), playing, created };
});
const last = snapshots[snapshots.length - 1]!;
const windowFrom = Date.parse(task.window.from);

/** Recent rank velocity per game, ranks/hour, from the latest read >= 12 h before the last. */
const velocities = new Map<number, number>();
{
  let prev = snapshots[0]!;
  for (let i = snapshots.length - 2; i >= 0; i--) {
    if (last.at - snapshots[i]!.at >= 12 * 3600 * 1000) {
      prev = snapshots[i]!;
      break;
    }
  }
  const gapHours = (last.at - prev.at) / 3600000;
  const prevRank = new Map<number, number>();
  prev.ids.forEach((id, i) => prevRank.set(id, i + 1));
  last.ids.forEach((id, i) => {
    const before = prevRank.get(id);
    if (before !== undefined && gapHours > 0) {
      velocities.set(id, clamp(((i + 1 - before) / gapHours) * VELOCITY_SHRINK, -VELOCITY_CLIP, VELOCITY_CLIP));
    }
  });
}

/** P(game still on the chart h hours after the window starts), 1 unless it ages out. */
function survival(id: number, horizonHours: number): number {
  const born = last.created.get(id);
  if (born === undefined || Number.isNaN(born)) return 1;
  const exitAt = born + CHART_AGE_MS;
  const at = windowFrom + horizonHours * 3600000;
  return clamp((exitAt - at) / EXIT_RAMP_MS, 0, 1);
}

const cand = last.ids.slice(0, CANDIDATES);
// q[g][k]: probability candidate g is at place k+1, averaged over the window's slots.
const q: number[][] = cand.map(() => new Array<number>(places).fill(0));
const slotHours = task.window.slot_minutes / 60;
for (let slot = 0; slot < task.window.slots; slot++) {
  const h = (slot + 1) * slotHours;
  const sd = SIGMA_C * Math.sqrt(h);
  for (let g = 0; g < cand.length; g++) {
    const id = cand[g]!;
    const drift = DRIFT_PER_HOUR + (velocities.get(id) ?? 0);
    const mu = g + 1 + drift * h;
    const alive = survival(id, h);
    if (alive === 0) continue;
    for (let k = 0; k < places; k++) {
      const hi = normCdf((k + 1 + 0.5 - mu) / sd);
      const lo = k === 0 ? 0 : normCdf((k + 1 - 0.5 - mu) / sd);
      q[g]![k]! += (hi - lo) * alive;
    }
  }
}
for (let g = 0; g < cand.length; g++) {
  for (let k = 0; k < places; k++) q[g]![k]! /= task.window.slots;
}

// New-entrant share per place, averaged over the window's slots.
const newShare = new Array<number>(places).fill(0);
for (let slot = 0; slot < task.window.slots; slot++) {
  const h = (slot + 1) * slotHours;
  for (let k = 0; k < places; k++) {
    newShare[k]! += clamp(((NEW_AT_70[k] ?? 0) * NEW_SCALE * Math.max(0, h - NEW_RISE_FROM_HOUR)) / 55, 0, NEW_CAP);
  }
}
for (let k = 0; k < places; k++) newShare[k]! /= task.window.slots;

// Scale known games so each place sums to 1 minus the new-entrant share.
for (let k = 0; k < places; k++) {
  let col = 0;
  for (let g = 0; g < cand.length; g++) col += q[g]![k]!;
  const scale = col > 0 ? (1 - newShare[k]!) / col : 0;
  for (let g = 0; g < cand.length; g++) q[g]![k]! *= scale;
}

// Name games greedily above the unnamed baseline: a game is named at a place only while
// its probability beats what that place would give it unnamed.
const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
const nChart = last.ids.length;
for (let k = 0; k < places; k++) {
  const order = cand
    .map((id, g) => ({ id, g, p: q[g]![k]! }))
    .sort((a, b) => b.p - a.p || a.g - b.g);
  let namedMass = 0;
  let namedCount = 0;
  for (const row of order) {
    const baseline = (1 - namedMass) / Math.max(1, nChart - namedCount);
    if (row.p <= baseline || namedCount >= 100) break;
    probabilities.push({ universe_id: row.id, place: k + 1, p: Math.round(row.p * 1e6) / 1e6 });
    namedMass += row.p;
    namedCount++;
  }
}

// top10: games ordered by sum over places of q * (11 - place).
const topScore = cand.map((id, g) => {
  let s = 0;
  for (let k = 0; k < places; k++) s += q[g]![k]! * (places + 1 - (k + 1));
  return { id, s, g };
});
topScore.sort((a, b) => b.s - a.s || a.g - b.g);
const top10 = topScore.slice(0, places).map((r) => r.id);

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
