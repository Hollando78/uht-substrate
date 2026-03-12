import { Command } from "commander";
import type { UHTClient } from "../client.js";
import { output } from "../output.js";

const TRAIT_NAMES = [
  "physical_object", "synthetic", "biological", "powered",
  "structural", "observable", "physical_medium", "active",
  "intentionally_designed", "outputs_effect", "processes_signals", "state_transforming",
  "human_interactive", "system_integrated", "functionally_autonomous", "system_essential",
  "symbolic", "signalling", "rule_governed", "compositional",
  "normative", "meta", "temporal", "digital_virtual",
  "social_construct", "institutionally_defined", "identity_linked", "regulated",
  "economically_significant", "politicised", "ritualised", "ethically_significant",
];

export function registerEntityCommands(
  program: Command,
  getClient: () => UHTClient,
  getFormat: () => "json" | "pretty",
): void {
  const entities = program
    .command("entities")
    .description("Entity management commands");

  entities
    .command("list")
    .description("List entities in the knowledge graph")
    .option("-n, --name <filter>", "Filter by name substring")
    .option("--hex <pattern>", "Filter by hex pattern")
    .option("-l, --limit <n>", "Max results", "50")
    .option("--offset <n>", "Skip N results", "0")
    .action(async (opts: { name?: string; hex?: string; limit: string; offset: string }) => {
      output(
        await getClient().listEntities({
          name_contains: opts.name,
          hex_pattern: opts.hex,
          limit: parseInt(opts.limit),
          offset: parseInt(opts.offset),
        }),
        getFormat(),
      );
    });

  entities
    .command("get")
    .description("Get a single entity by name or UUID")
    .option("-n, --name <name>", "Entity name (case-insensitive)")
    .option("-u, --uuid <uuid>", "Entity UUID")
    .action(async (opts: { name?: string; uuid?: string }) => {
      if (!opts.name && !opts.uuid) {
        console.error("Error: provide --name or --uuid");
        process.exit(1);
      }
      output(await getClient().getEntity(opts), getFormat());
    });

  entities
    .command("delete")
    .description("Delete an entity from the local graph")
    .argument("<name>", "Exact entity name")
    .action(async (name: string) => {
      output(await getClient().deleteEntity(name), getFormat());
    });

  entities
    .command("find-similar")
    .description("[Experimental] Find entities similar to the given entity")
    .argument("<entity>", "Entity to find similar items for")
    .option("-l, --limit <n>", "Max results", "5")
    .option("--min-traits <n>", "Minimum shared traits", "20")
    .action(async (entity: string, opts: { limit: string; minTraits: string }) => {
      output(
        await getClient().findSimilar(entity, {
          limit: parseInt(opts.limit),
          min_shared_traits: parseInt(opts.minTraits),
        }),
        getFormat(),
      );
    });

  entities
    .command("explore")
    .description("[Experimental] Explore semantic neighborhood of an entity")
    .argument("<entity>", "Entity to explore from")
    .option("-m, --metric <metric>", "Similarity metric: embedding, hamming, hybrid", "embedding")
    .option("-l, --limit <n>", "Max results", "10")
    .option("--min-similarity <n>", "Minimum similarity threshold", "0.3")
    .action(async (entity: string, opts: { metric: string; limit: string; minSimilarity: string }) => {
      output(
        await getClient().exploreNeighborhood(entity, {
          metric: opts.metric,
          limit: parseInt(opts.limit),
          min_similarity: parseFloat(opts.minSimilarity),
        }),
        getFormat(),
      );
    });

  entities
    .command("search-traits")
    .description("Search entities by trait pattern (use --<trait-name> or --no-<trait-name>)")
    .option("-l, --limit <n>", "Max results", "100")
    .allowUnknownOption(true)
    .action(async function (this: Command) {
      const rawArgs = this.args;
      const query: Record<string, string> = {};
      let limit = "100";

      for (let i = 0; i < rawArgs.length; i++) {
        const arg = rawArgs[i]!;
        if (arg === "--limit" || arg === "-l") {
          limit = rawArgs[++i] ?? "100";
          continue;
        }
        // --trait-name or --no-trait-name
        const noMatch = arg.match(/^--no-(.+)$/);
        const yesMatch = arg.match(/^--(.+)$/);
        if (noMatch) {
          const trait = noMatch[1]!.replace(/-/g, "_");
          if (TRAIT_NAMES.includes(trait)) query[trait] = "false";
        } else if (yesMatch) {
          const trait = yesMatch[1]!.replace(/-/g, "_");
          if (TRAIT_NAMES.includes(trait)) query[trait] = "true";
        }
      }
      query["limit"] = limit;
      output(await getClient().searchByTraits(query), getFormat());
    });
}
