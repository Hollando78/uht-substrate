import { Command } from "commander";
import type { UHTClient } from "../client.js";
import { output } from "../output.js";

export function registerNamespaceCommands(
  program: Command,
  getClient: () => UHTClient,
  getFormat: () => "json" | "pretty",
): void {
  const ns = program
    .command("namespaces")
    .alias("ns")
    .description("Namespace management commands");

  ns.command("create")
    .description("Create a new namespace")
    .argument("<code>", "Unique namespace code (e.g. SE:aerospace)")
    .argument("<name>", "Human-readable name")
    .option("-d, --description <desc>", "Optional description")
    .action(async (code: string, name: string, opts: { description?: string }) => {
      output(await getClient().createNamespace(code, name, opts.description), getFormat());
    });

  ns.command("list")
    .description("List namespaces")
    .option("-p, --parent <code>", "List children of this namespace")
    .option("--descendants", "Include entire subtree", false)
    .action(async (opts: { parent?: string; descendants: boolean }) => {
      output(
        await getClient().listNamespaces({
          parent: opts.parent,
          include_descendants: opts.descendants,
        }),
        getFormat(),
      );
    });

  ns.command("assign")
    .description("Assign an entity to a namespace")
    .argument("<entity>", "Entity name")
    .argument("<namespace>", "Namespace code")
    .option("--no-primary", "Do not set as primary namespace")
    .action(async (entity: string, namespace: string, opts: { primary: boolean }) => {
      output(await getClient().assignNamespace(entity, namespace, opts.primary), getFormat());
    });
}
