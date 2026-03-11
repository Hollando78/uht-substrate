import { Command } from "commander";
import type { UHTClient } from "../client.js";
import { output } from "../output.js";

export function registerInfoCommands(
  program: Command,
  getClient: () => UHTClient,
  getFormat: () => "json" | "pretty",
): void {
  program
    .command("info")
    .description("Get system information and overview")
    .action(async () => {
      output(await getClient().info(), getFormat());
    });

  program
    .command("traits")
    .description("Get all 32 trait definitions")
    .action(async () => {
      output(await getClient().traits(), getFormat());
    });

  program
    .command("trait-prompts")
    .description("Get classifier prompts sent to the LLM (includes edge cases and examples)")
    .option("-e, --entity <name>", "Entity name to substitute into prompts")
    .option("-d, --description <desc>", "Entity description to substitute")
    .option("-b, --bit <n>", "Single trait bit (1-32), or omit for all")
    .action(async (opts: { entity?: string; description?: string; bit?: string }) => {
      output(
        await getClient().traitPrompts({
          entity_name: opts.entity,
          entity_description: opts.description,
          bit: opts.bit ? parseInt(opts.bit) : undefined,
        }),
        getFormat(),
      );
    });

  program
    .command("patterns")
    .description("Get reasoning patterns for tool orchestration")
    .action(async () => {
      output(await getClient().patterns(), getFormat());
    });
}
