// Polymarket FAA outage attribution forecaster (Node 22, standard library only).
//
// Forecast: for each market, the Yes price at every minute of the 240 minutes
// after the checkpoint, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).
//
// Model: a random walk around the market's last price, with a spread fitted
// to THAT market's own recent behaviour:
//
//   q_tau(h) = clamp(x + z_tau * sigma * sqrt(h), 0, 1), rounded to 6 decimals
//
// where x is the market's price at the checkpoint and sigma is estimated per
// market from the mean absolute deviation of its 1-minute price moves over
// the last 120 minutes, scaled to a standard deviation (mean |move - mean|
// times 1.4826). Prices move in steps — mostly flat, then a tick, now and
// then a jump — so the mean absolute deviation measures the typical minute
// move without letting one big jump inflate the spread the way a plain
// standard deviation would. Sigma is floored (a dead market still gets a
// hair of spread) and capped at 0.004 (a market that just jumped does not
// get absurdly wide quantiles: on the practice cases the 0.008 cap let
// jumpy markets over-widen and cost loss on every quiet minute).
//
// The quantile levels are NOT the normal ones. The 1-minute moves are
// extremely peaked (88% of minutes do not move at all) with very heavy
// tails (kurtosis ~187), so the honest quantiles of the move distribution
// sit closer to the centre at 0.25/0.75 and further out at 0.05/0.95 than
// a normal's: z = [-1.9, -0.5, 0, 0.5, 1.9]. That shape, fitted on the
// practice cases, beats the normal shape on every one of the six cases.
//
// What was tried and rejected on the practice data (all measured, not
// assumed): any drift, positive or negative, and any pull toward a trailing
// mean — the median is the last price, full stop; Student-t quantiles;
// forecasting in logit space; sigma from trailing 240-minute moves, from
// realised variance, or pooled across the case's markets; a wider jump
// guard. Each made the practice loss worse. The loss is dominated by
// news jumps no causal model can foresee: with a perfect ex-post sigma
// the practice loss would be 0.221, and no causal sigma estimator tested
// beat the 120-minute mean absolute deviation, because future volatility
// is nearly unpredictable from past moves here.
//
// Practice loss (mean pinball x100 over p00-p05, verified against the
// server's own numbers: baseline 0.363589, previous entry 0.279972):
// this model scores 0.269930, better than the previous entry on all six
// cases (p00 0.479, p01 0.313, p02 0.333, p03 0.241, p04 0.207, p05 0.111).
//
// Reads every <id>.in in the input directory (the grader runs one case at a
// time, with only that case's file there) and writes <id>.out to the output
// directory: {"forecasts":[{"id","q"}]}, q exactly horizon_minutes rows of
// five non-decreasing numbers in [0,1]. Deterministic, no network, no clock.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/**
 * The spread multipliers at each level. Not the normal quantiles: the
 * 1-minute moves are far more peaked than a normal (88% of minutes do not
 * move) with far heavier tails, so the 0.25/0.75 levels sit closer to the
 * centre and the 0.05/0.95 levels further out than a normal's -1.645/-0.674/
 * 0/0.674/1.645. Fitted on the practice cases; beats the normal shape on
 * all six of them.
 */
const Z = [-1.9, -0.5, 0, 0.5, 1.9];

/** Minutes of recent history used to fit a market's spread. Short, so the fit follows the current regime. */
const SIGMA_WINDOW = 120;
/** Fallback spread when a market has too few observed moves to fit one. */
const SIGMA_FALLBACK = 0.0025;
/** A market with fewer than this many 1-minute moves uses the fallback spread. */
const MIN_MOVES = 10;
/**
 * Floor and cap on the fitted spread, in price units per square-root minute.
 * The 0.004 cap (down from 0.008) is the single biggest win over the
 * previous entry: jumpy markets no longer get quantiles so wide they pay
 * for it on every quiet minute that follows.
 */
const SIGMA_FLOOR = 0.0002;
const SIGMA_CAP = 0.004;
/** A single-minute move larger than this, inside the last JUMP_LOOKBACK minutes,
 *  counts as a jump: the market is in a jumpy regime, so widen its spread. */
const JUMP_THRESHOLD = 0.01;
const JUMP_LOOKBACK = 10;
/** How much a recent jump widens the fitted spread (still capped at SIGMA_CAP). */
const JUMP_WIDEN = 1.5;
/** Mean-absolute-deviation-to-standard-deviation factor. */
const MAD_TO_SD = 1.4826;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/**
 * A market's spread: the mean absolute deviation of its 1-minute moves over
 * the last SIGMA_WINDOW minutes, around their mean, scaled to a standard
 * deviation, floored and capped. Falls back to SIGMA_FALLBACK when there
 * are too few moves to fit.
 */
function fitSigma(prices: number[]): number {
  const n = prices.length;
  const moves = Math.min(SIGMA_WINDOW, n - 1);
  if (moves < MIN_MOVES) return SIGMA_FALLBACK;
  let mean = 0;
  const rets: number[] = new Array(moves);
  for (let i = 0; i < moves; i++) {
    const r = prices[n - moves + i] - prices[n - moves + i - 1];
    rets[i] = r;
    mean += r;
  }
  mean /= moves;
  let mad = 0;
  for (let i = 0; i < moves; i++) mad += Math.abs(rets[i] as number - mean);
  mad /= moves;
  let sigma = mad * MAD_TO_SD;
  // Jump guard: a big single-minute move in the last few minutes means the
  // market is in a jumpy regime (the FAA markets jumped 0.2+ in one minute),
  // so widen the spread instead of trusting the quiet MAD alone.
  const tail = Math.min(JUMP_LOOKBACK, moves);
  for (let i = moves - tail; i < moves; i++) {
    if (Math.abs(rets[i] as number) > JUMP_THRESHOLD) {
      sigma *= JUMP_WIDEN;
      break;
    }
  }
  // Defensive: a non-finite spread (bad input) must never reach the output
  // and invalidate the whole case; fall back to the baseline spread instead.
  if (!Number.isFinite(sigma)) return SIGMA_FALLBACK;
  return Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sigma));
}

/** The forecast of one market: horizon rows of five quantiles around its last price. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;
  const sigma = fitSigma(m.prices);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = sigma * Math.sqrt(h);
    rows.push(Z.map((z) => price6(x + z * s)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
