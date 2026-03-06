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
    .option("-c, --context <context>", "Additional context for classification")
    .option("--semantic-priors", "Use semantic prior inference", false)
    .action(async (entity: string, opts: { context?: string; semanticPriors: boolean }) => {
      output(
        await getClient().classify(entity, {
          context: opts.context,
          use_semantic_priors: opts.semanticPriors,
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
