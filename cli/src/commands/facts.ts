import { readFileSync } from "node:fs";
import { Command } from "commander";
import type { UHTClient } from "../client.js";
import { output } from "../output.js";

export function registerFactCommands(
  program: Command,
  getClient: () => UHTClient,
  getFormat: () => "json" | "pretty",
): void {
  const facts = program
    .command("facts")
    .description("Fact management commands");

  facts
    .command("store")
    .description("Store a fact in the knowledge graph")
    .argument("<subject>", "Subject of the fact")
    .argument("<predicate>", "Relationship/predicate")
    .argument("<object>", "Object value")
    .option("-u, --user <id>", "User ID", "default")
    .option("-n, --namespace <ns>", "Namespace")
    .action(async (subject: string, predicate: string, object: string, opts: { user: string; namespace?: string }) => {
      output(
        await getClient().storeFact(subject, predicate, object, {
          user_id: opts.user,
          namespace: opts.namespace,
        }),
        getFormat(),
      );
    });

  facts
    .command("store-bulk")
    .description("Store multiple facts from a JSON file")
    .option("-f, --file <path>", "JSON file with array of facts (or - for stdin)")
    .action(async (opts: { file?: string }) => {
      let data: string;
      if (!opts.file || opts.file === "-") {
        const chunks: Buffer[] = [];
        for await (const chunk of process.stdin) {
          chunks.push(chunk as Buffer);
        }
        data = Buffer.concat(chunks).toString("utf-8");
      } else {
        data = readFileSync(opts.file, "utf-8");
      }
      const parsed = JSON.parse(data) as Array<{ subject: string; predicate: string; object_value: string; user_id?: string; namespace?: string }>;
      output(await getClient().storeFactsBulk(parsed), getFormat());
    });

  facts
    .command("upsert")
    .description("Upsert a fact (create or update)")
    .argument("<subject>", "Subject of the fact")
    .argument("<predicate>", "Relationship/predicate")
    .argument("<object>", "Object value")
    .option("-u, --user <id>", "User ID", "default")
    .option("-n, --namespace <ns>", "Namespace")
    .action(async (subject: string, predicate: string, object: string, opts: { user: string; namespace?: string }) => {
      output(
        await getClient().upsertFact(subject, predicate, object, {
          user_id: opts.user,
          namespace: opts.namespace,
        }),
        getFormat(),
      );
    });

  facts
    .command("query")
    .description("Query facts with flexible filters")
    .option("-s, --subject <subject>", "Filter by subject")
    .option("-o, --object <object>", "Filter by object value")
    .option("-p, --predicate <predicate>", "Filter by predicate")
    .option("-c, --category <category>", "Filter by category")
    .option("-u, --user <id>", "Filter by user ID")
    .option("-n, --namespace <ns>", "Filter by namespace")
    .option("-l, --limit <n>", "Max results", "20")
    .action(async (opts: { subject?: string; object?: string; predicate?: string; category?: string; user?: string; namespace?: string; limit: string }) => {
      output(
        await getClient().queryFacts({
          subject: opts.subject,
          object_value: opts.object,
          predicate: opts.predicate,
          category: opts.category,
          user_id: opts.user,
          namespace: opts.namespace,
          limit: parseInt(opts.limit),
        }),
        getFormat(),
      );
    });

  facts
    .command("update")
    .description("Update an existing fact")
    .argument("<fact-id>", "UUID of the fact")
    .option("-s, --subject <subject>", "New subject")
    .option("-p, --predicate <predicate>", "New predicate")
    .option("-o, --object <object>", "New object value")
    .action(async (factId: string, opts: { subject?: string; predicate?: string; object?: string }) => {
      output(
        await getClient().updateFact(factId, {
          subject: opts.subject,
          predicate: opts.predicate,
          object_value: opts.object,
        }),
        getFormat(),
      );
    });

  facts
    .command("delete")
    .description("Delete a fact from the knowledge graph")
    .argument("<fact-id>", "UUID of the fact")
    .action(async (factId: string) => {
      output(await getClient().deleteFact(factId), getFormat());
    });

  facts
    .command("user-context")
    .description("Get stored facts and preferences for a user")
    .option("-u, --user <id>", "User ID", "default")
    .action(async (opts: { user: string }) => {
      output(await getClient().getUserContext(opts.user), getFormat());
    });

  facts
    .command("namespace-context")
    .description("Get all entities and facts under a namespace")
    .argument("<namespace>", "Namespace code")
    .option("-u, --user <id>", "Filter facts by user ID")
    .action(async (namespace: string, opts: { user?: string }) => {
      output(await getClient().getNamespaceContext(namespace, opts.user), getFormat());
    });
}
