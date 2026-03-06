import { Command } from "commander";
import type { UHTClient } from "../client.js";
import { output } from "../output.js";

export function registerSemanticCommands(
  program: Command,
  getClient: () => UHTClient,
  getFormat: () => "json" | "pretty",
): void {
  program
    .command("search")
    .description("Semantic search for similar entities")
    .argument("<query>", "Search query")
    .option("-l, --limit <n>", "Max results", "10")
    .action(async (query: string, opts: { limit: string }) => {
      output(await getClient().search(query, parseInt(opts.limit)), getFormat());
    });

  program
    .command("disambiguate")
    .description("Get word senses for a polysemous term")
    .argument("<term>", "Term to disambiguate")
    .option("--language <lang>", "Language code", "en")
    .action(async (term: string, opts: { language: string }) => {
      output(await getClient().disambiguate(term, opts.language), getFormat());
    });

  program
    .command("semantic-triangle")
    .description("Get the Ogden-Richards semantic triangle for a term")
    .argument("<text>", "Text to analyze")
    .action(async (text: string) => {
      output(await getClient().semanticTriangle(text), getFormat());
    });

  program
    .command("map-properties")
    .description("Map natural language properties to candidate UHT trait bits")
    .argument("<properties...>", "Properties to map")
    .action(async (properties: string[]) => {
      output(await getClient().mapPropertiesToTraits(properties), getFormat());
    });
}
