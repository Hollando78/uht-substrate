const NO_COLOR = !!process.env["NO_COLOR"];

export function bold(s: string): string {
  return NO_COLOR ? s : `\x1b[1m${s}\x1b[0m`;
}

export function dim(s: string): string {
  return NO_COLOR ? s : `\x1b[2m${s}\x1b[0m`;
}

export function cyan(s: string): string {
  return NO_COLOR ? s : `\x1b[36m${s}\x1b[0m`;
}

export function green(s: string): string {
  return NO_COLOR ? s : `\x1b[32m${s}\x1b[0m`;
}

export function yellow(s: string): string {
  return NO_COLOR ? s : `\x1b[33m${s}\x1b[0m`;
}

export function red(s: string): string {
  return NO_COLOR ? s : `\x1b[31m${s}\x1b[0m`;
}

/** Print data as JSON or pretty-formatted. */
export function output(data: unknown, format: "json" | "pretty"): void {
  if (format === "json") {
    console.log(JSON.stringify(data, null, 2));
  } else {
    prettyPrint(data);
  }
}

function prettyPrint(data: unknown): void {
  if (data === null || data === undefined) {
    console.log(dim("(empty)"));
    return;
  }

  if (typeof data !== "object") {
    console.log(String(data));
    return;
  }

  const obj = data as Record<string, unknown>;

  // Classification result
  if ("hex_code" in obj && "traits" in obj) {
    printClassification(obj);
    return;
  }

  // Comparison result
  if ("similarity" in obj && "trait_diff" in obj) {
    printComparison(obj);
    return;
  }
  // Legacy comparison format
  if ("jaccard_similarity" in obj && "shared_traits" in obj) {
    printComparison(obj);
    return;
  }

  // Array of results
  if (Array.isArray(data)) {
    for (const item of data) {
      prettyPrint(item);
      console.log();
    }
    return;
  }

  // Traits definitions
  if ("total_traits" in obj && "layers" in obj) {
    printTraits(obj);
    return;
  }

  // Trait prompts (full set)
  if ("prompt_spec_version" in obj && "prompts" in obj) {
    printTraitPrompts(obj);
    return;
  }

  // Single trait prompt
  if ("system_prompt" in obj && "trait_name" in obj) {
    printSinglePrompt(obj);
    return;
  }

  // Wrap with results/entities/facts arrays
  if ("results" in obj && Array.isArray(obj.results)) {
    prettyPrint(obj.results);
    return;
  }
  if ("entities" in obj && Array.isArray(obj.entities)) {
    printEntitiesTable(obj);
    return;
  }
  if ("facts" in obj && Array.isArray(obj.facts)) {
    printFactsTable(obj.facts as Record<string, unknown>[]);
    return;
  }

  // Generic object
  printObject(obj);
}

function printClassification(obj: Record<string, unknown>): void {
  const label = String(obj.name ?? obj.entity ?? "");
  console.log(`${bold(label)}  ${cyan(String(obj.hex_code))}`);
  if (obj.binary) console.log(dim(String(obj.binary)));
  console.log();

  const traits = obj.traits as Array<Record<string, unknown>> | undefined;
  if (traits) {
    const present = traits.filter((t) => t.present);
    const absent = traits.filter((t) => !t.present);
    if (present.length) {
      console.log(bold("Active traits:"));
      for (const t of present) {
        const name = String(t.name ?? t.trait ?? "");
        const conf = typeof t.confidence === "number" ? ` ${dim(`(${(t.confidence * 100).toFixed(0)}%)`)}` : "";
        console.log(`  ${green("+")} ${name}${conf}`);
      }
    }
    const namedAbsent = absent.filter((t) => t.name || t.trait);
    if (namedAbsent.length) {
      console.log(bold("\nInactive traits:"));
      for (const t of namedAbsent) {
        const name = String(t.name ?? t.trait ?? "");
        console.log(`  ${dim("-")} ${dim(name)}`);
      }
    } else if (absent.length) {
      console.log(dim(`\n${absent.length} inactive traits`));
    }
  }

  const inferred = obj.inferred_properties;
  if (inferred && Array.isArray(inferred)) {
    console.log(bold("\nInferred properties:"));
    for (const p of inferred) {
      if (typeof p === "object" && p !== null) {
        const prop = p as Record<string, unknown>;
        const certainty = prop.certainty ?? prop.confidence ?? "";
        console.log(`  ${bold(String(prop.property ?? prop.name ?? ""))} ${dim(String(certainty))}`);
        if (prop.explanation) console.log(`    ${dim(String(prop.explanation))}`);
      }
    }
  } else if (inferred && typeof inferred === "object") {
    console.log(bold("\nInferred properties:"));
    printObject(inferred as Record<string, unknown>, 2);
  }
}

function printComparison(obj: Record<string, unknown>): void {
  const entityA = String(obj.entity_a ?? "A");
  const entityB = String(obj.entity_b ?? "B");
  console.log(`${bold(entityA)} ${dim("vs")} ${bold(entityB)}`);

  // Handle nested similarity object
  const sim = (obj.similarity ?? obj) as Record<string, unknown>;
  const jaccard = typeof sim.jaccard_similarity === "number" ? sim.jaccard_similarity : 0;
  const hamming = sim.hamming_distance ?? obj.hamming_distance ?? "?";
  const color = jaccard > 0.7 ? green : jaccard > 0.3 ? yellow : red;
  console.log(`Jaccard: ${color((jaccard * 100).toFixed(1) + "%")}  Hamming: ${hamming}`);

  // Handle hex codes
  const hexCodes = obj.hex_codes as Record<string, string> | undefined;
  if (hexCodes) {
    console.log(`${cyan(hexCodes[entityA] ?? "")} / ${cyan(hexCodes[entityB] ?? "")}`);
  }

  // Handle trait_diff or flat shared_traits
  const diff = obj.trait_diff as Record<string, unknown> | undefined;
  const shared = extractTraitNames(diff?.shared_traits ?? obj.shared_traits);
  const aOnly = extractTraitNames(diff?.traits_a_only ?? obj.traits_a_only);
  const bOnly = extractTraitNames(diff?.traits_b_only ?? obj.traits_b_only);

  if (shared.length) {
    console.log(bold("\nShared traits:"));
    console.log(`  ${shared.join(", ")}`);
  }
  if (aOnly.length) {
    console.log(bold(`\n${entityA} only:`));
    console.log(`  ${aOnly.join(", ")}`);
  }
  if (bOnly.length) {
    console.log(bold(`\n${entityB} only:`));
    console.log(`  ${bOnly.join(", ")}`);
  }

  if (obj.comparison) {
    console.log(`\n${dim(String(obj.comparison))}`);
  }
}

function extractTraitNames(val: unknown): string[] {
  if (!val) return [];
  if (Array.isArray(val)) {
    return val.map((t) => {
      if (typeof t === "string") return t;
      if (typeof t === "object" && t !== null) {
        const o = t as Record<string, unknown>;
        return String(o.name ?? o.trait ?? "");
      }
      return String(t);
    }).filter(Boolean);
  }
  return [];
}

function printTable(rows: Record<string, unknown>[]): void {
  if (!rows.length) {
    console.log(dim("(no results)"));
    return;
  }
  const cols = ["name", "hex_code", "source", "created_at"].filter(
    (c) => rows.some((r) => r[c] !== undefined),
  );
  const widths = cols.map((c) =>
    Math.max(c.length, ...rows.map((r) => String(r[c] ?? "").length)),
  );
  console.log(cols.map((c, i) => bold(c.padEnd(widths[i]!))).join("  "));
  console.log(cols.map((_, i) => "-".repeat(widths[i]!)).join("  "));
  for (const row of rows) {
    console.log(cols.map((c, i) => String(row[c] ?? "").padEnd(widths[i]!)).join("  "));
  }
}

function printFactsTable(rows: Record<string, unknown>[]): void {
  if (!rows.length) {
    console.log(dim("(no facts)"));
    return;
  }
  for (const f of rows) {
    console.log(`${cyan(String(f.subject))} ${bold(String(f.predicate))} ${cyan(String(f.object_value))}`);
    const meta = [f.category, f.namespace, f.user_id].filter(Boolean).join(" | ");
    if (meta) console.log(`  ${dim(meta)}`);
    if (f.id) console.log(`  ${dim(String(f.id))}`);
  }
}

function printTraits(obj: Record<string, unknown>): void {
  if (obj.version) console.log(`${bold("Trait version:")} ${cyan(String(obj.version))}`);
  console.log(`${bold("Total traits:")} ${obj.total_traits}  ${dim(String(obj.encoding ?? ""))}`);
  console.log();

  const layers = obj.layers as Record<string, Record<string, unknown>> | undefined;
  if (!layers) return;

  const layerColors: Record<string, (s: string) => string> = {
    physical: red,
    functional: yellow,
    abstract: cyan,
    social: green,
  };

  for (const [layerKey, layer] of Object.entries(layers)) {
    const colorKey = Object.keys(layerColors).find((k) => layerKey.toLowerCase().includes(k));
    const colorFn = colorKey ? layerColors[colorKey]! : bold;
    console.log(colorFn(bold(layerKey)));
    if (layer.description) console.log(`  ${dim(String(layer.description))}`);

    const traits = layer.traits as Array<Record<string, unknown>> | undefined;
    if (traits) {
      for (const t of traits) {
        const bit = String(t.bit ?? "").padStart(2);
        console.log(`  ${dim(bit)}  ${bold(String(t.name ?? ""))} ${dim("—")} ${String(t.short_description ?? "")}`);
      }
    }
    console.log();
  }
}

function printTraitPrompts(obj: Record<string, unknown>): void {
  console.log(`${bold("Prompt spec version:")} ${cyan(String(obj.prompt_spec_version ?? "?"))}`);
  console.log(`${bold("Model:")} ${String(obj.model ?? "?")}  ${bold("Temperature:")} ${String(obj.temperature ?? "?")}`);
  console.log(`${bold("Prompts:")} ${String(obj.count ?? "?")}`);
  console.log();

  const prompts = obj.prompts as Array<Record<string, unknown>> | undefined;
  if (prompts) {
    for (const p of prompts) {
      console.log(`  ${dim(String(p.bit ?? "").padStart(2))}  ${bold(String(p.trait_name ?? ""))} ${dim(`[${p.layer}]`)}`);
    }
  }
  console.log();
  console.log(dim("Use --bit <n> to see the full prompt for a specific trait."));
}

function printSinglePrompt(obj: Record<string, unknown>): void {
  console.log(`${bold(String(obj.trait_name ?? ""))} ${dim(`[bit ${obj.bit}, ${obj.layer}]`)}`);
  console.log();
  console.log(String(obj.system_prompt ?? ""));
  if (obj.user_prompt) {
    console.log();
    console.log(bold("User prompt:"));
    console.log(String(obj.user_prompt));
  }
}

function printEntitiesTable(obj: Record<string, unknown>): void {
  const total = obj.total ?? "?";
  const offset = obj.offset ?? 0;
  const count = obj.count ?? 0;
  const hasMore = obj.has_more ?? false;
  console.log(`${bold("Entities:")} ${count} of ${total}${hasMore ? dim(` (offset ${offset}, more available)`) : ""}`);
  console.log();

  const entities = obj.entities as Record<string, unknown>[] | undefined;
  if (!entities || !entities.length) {
    console.log(dim("(no results)"));
    return;
  }

  for (const e of entities) {
    const hex = cyan(String(e.hex_code ?? ""));
    const name = bold(String(e.name ?? ""));
    const uuid = dim(String(e.uuid ?? "").slice(0, 8));
    console.log(`  ${uuid}  ${hex}  ${name}`);
    if (e.description) console.log(`           ${dim(String(e.description).slice(0, 100))}`);
  }
}

function printObject(obj: Record<string, unknown>, indent = 0): void {
  const pad = " ".repeat(indent);
  for (const [key, value] of Object.entries(obj)) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      console.log(`${pad}${bold(key)}:`);
      printObject(value as Record<string, unknown>, indent + 2);
    } else if (Array.isArray(value)) {
      console.log(`${pad}${bold(key)}: ${value.join(", ")}`);
    } else {
      console.log(`${pad}${bold(key)}: ${String(value)}`);
    }
  }
}
