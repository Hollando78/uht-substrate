import { existsSync, readFileSync } from "node:fs";
import { Command } from "commander";
import type { UHTClient } from "../client.js";
import { bold, cyan, dim, green, red, yellow } from "../output.js";

// --- Types for airgen diff input ---

interface AirgenRequirement {
  ref: string;
  text: string;
  [key: string]: unknown;
}

interface AirgenModified {
  ref: string;
  old_text: string;
  new_text: string;
  [key: string]: unknown;
}

interface AirgenDiff {
  summary?: Record<string, unknown>;
  added?: AirgenRequirement[];
  removed?: AirgenRequirement[];
  modified?: AirgenModified[];
}

// --- Types for classify response ---

interface ClassifyTrait {
  bit: number;
  name: string;
  present: boolean;
  confidence: number;
}

interface ClassifyResult {
  hex_code: string;
  traits: ClassifyTrait[];
  [key: string]: unknown;
}

// --- Output types ---

interface DriftEntry {
  ref: string;
  old_hex: string;
  new_hex: string;
  old_text: string;
  new_text: string;
  flipped_traits: Array<{ bit: number; name: string; direction: "added" | "removed" }>;
}

interface ClassifiedEntry {
  ref: string;
  hex: string;
  text: string;
  top_traits: string[];
}

interface ImpactResult {
  summary: { added: number; removed: number; modified: number; semantic_drift: number };
  drift: DriftEntry[];
  added: ClassifiedEntry[];
  removed: ClassifiedEntry[];
}

// --- Helpers ---

function getActiveTraitNames(traits: ClassifyTrait[]): string[] {
  return traits.filter((t) => t.present && t.name).map((t) => t.name);
}

function findFlippedTraits(
  oldTraits: ClassifyTrait[],
  newTraits: ClassifyTrait[],
): DriftEntry["flipped_traits"] {
  const flipped: DriftEntry["flipped_traits"] = [];
  const oldMap = new Map(oldTraits.map((t) => [t.bit, t]));
  for (const nt of newTraits) {
    const ot = oldMap.get(nt.bit);
    if (!ot) continue;
    if (ot.present && !nt.present && ot.name) {
      flipped.push({ bit: nt.bit, name: ot.name, direction: "removed" });
    } else if (!ot.present && nt.present && nt.name) {
      flipped.push({ bit: nt.bit, name: nt.name, direction: "added" });
    }
  }
  return flipped;
}

// --- Core logic ---

async function runImpact(client: UHTClient, diff: AirgenDiff): Promise<ImpactResult> {
  const added: ClassifiedEntry[] = [];
  const removed: ClassifiedEntry[] = [];
  const drift: DriftEntry[] = [];

  // Classify added requirements
  const addedReqs = diff.added ?? [];
  for (const req of addedReqs) {
    const result = (await client.classify(req.text)) as ClassifyResult;
    added.push({
      ref: req.ref,
      hex: result.hex_code,
      text: req.text,
      top_traits: getActiveTraitNames(result.traits),
    });
  }

  // Classify removed requirements
  const removedReqs = diff.removed ?? [];
  for (const req of removedReqs) {
    const result = (await client.classify(req.text)) as ClassifyResult;
    removed.push({
      ref: req.ref,
      hex: result.hex_code,
      text: req.text,
      top_traits: getActiveTraitNames(result.traits),
    });
  }

  // Classify both sides of modified requirements and detect drift
  const modifiedReqs = diff.modified ?? [];
  for (const req of modifiedReqs) {
    const [oldResult, newResult] = (await Promise.all([
      client.classify(req.old_text),
      client.classify(req.new_text),
    ])) as [ClassifyResult, ClassifyResult];

    if (oldResult.hex_code !== newResult.hex_code) {
      drift.push({
        ref: req.ref,
        old_hex: oldResult.hex_code,
        new_hex: newResult.hex_code,
        old_text: req.old_text,
        new_text: req.new_text,
        flipped_traits: findFlippedTraits(oldResult.traits, newResult.traits),
      });
    }
  }

  return {
    summary: {
      added: addedReqs.length,
      removed: removedReqs.length,
      modified: modifiedReqs.length,
      semantic_drift: drift.length,
    },
    drift,
    added,
    removed,
  };
}

// --- Pretty output ---

function prettyPrintImpact(result: ImpactResult): void {
  const s = result.summary;
  const driftLabel = s.semantic_drift > 0
    ? ` (${yellow(`${s.semantic_drift} with semantic drift`)})`
    : "";
  console.log(
    bold("Summary: ") +
      `${green(`${s.added} added`)}, ${red(`${s.removed} removed`)}, ` +
      `${cyan(`${s.modified} modified`)}${driftLabel}`,
  );

  if (result.drift.length) {
    console.log(bold("\nSemantic Drift:"));
    for (const d of result.drift) {
      console.log(`  ${bold(d.ref)}  ${red(d.old_hex)} → ${green(d.new_hex)}`);
      for (const f of d.flipped_traits) {
        const arrow = f.direction === "added" ? green("+") : red("-");
        console.log(`    ${arrow} ${f.name}`);
      }
    }
  }

  if (result.added.length) {
    console.log(bold("\nAdded:"));
    for (const a of result.added) {
      console.log(`  ${bold(a.ref)}  ${cyan(a.hex)}  ${a.top_traits.join(", ")}`);
    }
  }

  if (result.removed.length) {
    console.log(bold("\nRemoved:"));
    for (const r of result.removed) {
      console.log(`  ${bold(r.ref)}  ${cyan(r.hex)}  ${r.top_traits.join(", ")}`);
    }
  }
}

// --- Command registration ---

export function registerImpactCommand(
  program: Command,
  getClient: () => UHTClient,
  getFormat: () => "json" | "pretty",
): void {
  program
    .command("impact")
    .description("Analyse semantic impact of requirement changes from an airgen diff")
    .requiredOption("--airgen-diff <path>", "Path to airgen diff JSON file (airgen diff --json)")
    .option("--json", "Force JSON output")
    .action(async (opts: { airgenDiff: string; json?: boolean }) => {
      const filePath = opts.airgenDiff;
      if (!existsSync(filePath)) {
        console.error(`Error: File not found: ${filePath}`);
        process.exit(1);
      }

      let diff: AirgenDiff;
      try {
        const raw = readFileSync(filePath, "utf-8");
        diff = JSON.parse(raw) as AirgenDiff;
      } catch {
        console.error(`Error: Could not parse ${filePath} as JSON. Expected airgen diff format: { summary, added, removed, modified }`);
        process.exit(1);
      }

      if (!diff.added && !diff.removed && !diff.modified) {
        console.error(`Error: No added, removed, or modified arrays found in ${filePath}. Expected airgen diff format.`);
        process.exit(1);
      }

      const result = await runImpact(getClient(), diff);

      const format = opts.json ? "json" : getFormat();
      if (format === "json") {
        console.log(JSON.stringify(result, null, 2));
      } else {
        prettyPrintImpact(result);
      }
    });
}
