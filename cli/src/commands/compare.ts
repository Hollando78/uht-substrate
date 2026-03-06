import { Command } from "commander";
import type { UHTClient } from "../client.js";
import { output } from "../output.js";

export function registerCompareCommands(
  program: Command,
  getClient: () => UHTClient,
  getFormat: () => "json" | "pretty",
): void {
  program
    .command("compare")
    .description("Compare two entities")
    .argument("<entity-a>", "First entity")
    .argument("<entity-b>", "Second entity")
    .action(async (entityA: string, entityB: string) => {
      output(await getClient().compare(entityA, entityB), getFormat());
    });

  program
    .command("batch-compare")
    .description("Compare an entity against multiple candidates, ranked by Jaccard")
    .argument("<entity>", "Reference entity")
    .argument("<candidates...>", "Candidate entities to compare against")
    .action(async (entity: string, candidates: string[]) => {
      output(await getClient().batchCompare(entity, candidates), getFormat());
    });
}
