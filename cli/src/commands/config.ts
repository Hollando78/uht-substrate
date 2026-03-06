import { Command } from "commander";
import { saveConfigFile, showConfig } from "../config.js";

const VALID_KEYS: Record<string, string> = {
  "api-url": "apiUrl",
  "token": "token",
  "format": "format",
};

export function registerConfigCommands(program: Command): void {
  const config = program
    .command("config")
    .description("Manage CLI configuration");

  config
    .command("set")
    .description("Set a config value")
    .argument("<key>", `Config key: ${Object.keys(VALID_KEYS).join(", ")}`)
    .argument("<value>", "Config value")
    .action((key: string, value: string) => {
      const mapped = VALID_KEYS[key];
      if (!mapped) {
        console.error(`Unknown config key: ${key}`);
        console.error(`Valid keys: ${Object.keys(VALID_KEYS).join(", ")}`);
        process.exit(1);
      }
      saveConfigFile({ [mapped]: value });
      console.log(`Set ${key} = ${key === "token" ? "***" : value}`);
    });

  config
    .command("show")
    .description("Show current configuration")
    .action(() => {
      const cfg = showConfig();
      for (const [k, v] of Object.entries(cfg)) {
        console.log(`${k}: ${v}`);
      }
    });
}
