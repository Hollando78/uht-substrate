import { Command } from "commander";
import type { UHTClient } from "../client.js";
import { output } from "../output.js";

export function registerClassifyCommands(
  program: Command,
  getClient: () => UHTClient,
  getFormat: () => "json" | "pretty",
): void {
  program
    .command("classify")
    .description("Classify an entity and get its hex code")
    .argument("<entity>", "Entity to classify")
    .option("-c, --context <context>", "Additional context/description to guide classification")
    .option("--semantic-priors", "Use semantic prior inference", false)
    .option("-f, --force-refresh", "Skip cache and force fresh classification", false)
    .option("-n, --namespace <ns>", "Namespace code (e.g. 'SE', 'SE:aerospace')")
    .action(async (entity: string, opts: { context?: string; semanticPriors: boolean; forceRefresh: boolean; namespace?: string }) => {
      output(
        await getClient().classify(entity, {
          context: opts.context,
          use_semantic_priors: opts.semanticPriors,
          force_refresh: opts.forceRefresh,
          namespace: opts.namespace,
        }),
        getFormat(),
      );
    });

  program
    .command("infer")
    .description("Infer properties of an entity from its classification")
    .argument("<entity>", "Entity to analyze")
    .action(async (entity: string) => {
      output(await getClient().inferProperties(entity), getFormat());
    });
}
