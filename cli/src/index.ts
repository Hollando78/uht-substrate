#!/usr/bin/env node

import { Command } from "commander";
import { ApiError, UHTClient } from "./client.js";
import { resolveConfig, type CliGlobalOpts } from "./config.js";
import { registerClassifyCommands } from "./commands/classify.js";
import { registerCompareCommands } from "./commands/compare.js";
import { registerEntityCommands } from "./commands/entities.js";
import { registerSemanticCommands } from "./commands/semantic.js";
import { registerFactCommands } from "./commands/facts.js";
import { registerNamespaceCommands } from "./commands/namespaces.js";
import { registerInfoCommands } from "./commands/info.js";
import { registerConfigCommands } from "./commands/config.js";
import { registerImpactCommand } from "./commands/impact.js";

const program = new Command();

program
  .name("uht-substrate")
  .description("CLI for the Universal Hex Taxonomy Substrate API")
  .version("0.3.0")
  .option("--api-url <url>", "API base URL")
  .option("--token <token>", "Bearer token for authentication")
  .option("--format <fmt>", "Output format: json or pretty", "json")
  .option("--verbose", "Show request/response details", false);

let _client: UHTClient | undefined;

function getClient(): UHTClient {
  if (!_client) {
    const opts = program.opts<CliGlobalOpts & { verbose: boolean }>();
    const config = resolveConfig(opts);
    _client = new UHTClient(config);
    _client.verbose = opts.verbose ?? false;
  }
  return _client;
}

function getFormat(): "json" | "pretty" {
  const opts = program.opts<CliGlobalOpts>();
  const config = resolveConfig(opts);
  return config.format;
}

// Register all commands
registerInfoCommands(program, getClient, getFormat);
registerClassifyCommands(program, getClient, getFormat);
registerCompareCommands(program, getClient, getFormat);
registerEntityCommands(program, getClient, getFormat);
registerSemanticCommands(program, getClient, getFormat);
registerFactCommands(program, getClient, getFormat);
registerNamespaceCommands(program, getClient, getFormat);
registerConfigCommands(program);
registerImpactCommand(program, getClient, getFormat);

// Global error handler
function handleError(err: unknown): never {
  if (err instanceof ApiError) {
    if (err.status === 401) {
      console.error("Authentication failed. Set your token with:");
      console.error("  uht-substrate config set token <your-token>");
      console.error("  or: UHT_TOKEN=<token> uht-substrate ...");
    } else {
      console.error(`API error (${err.status}): ${err.body}`);
    }
  } else if (err instanceof Error) {
    console.error(`Error: ${err.message}`);
  } else {
    console.error(`Error: ${String(err)}`);
  }
  process.exit(1);
}

program.parseAsync(process.argv).catch(handleError);
