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
    .command("patterns")
    .description("Get reasoning patterns for tool orchestration")
    .action(async () => {
      output(await getClient().patterns(), getFormat());
    });
}
