import { existsSync, readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export interface Config {
  apiUrl: string;
  token: string;
  format: "json" | "pretty";
}

const DEFAULT_API_URL = "https://substrate.universalhex.org/api";

function getConfigDir(): string {
  const xdg = process.env["XDG_CONFIG_HOME"];
  return xdg ? join(xdg, "uht-substrate") : join(homedir(), ".config", "uht-substrate");
}

function getConfigPath(): string {
  return join(getConfigDir(), "config.json");
}

function loadConfigFile(): Partial<Config> {
  const path = getConfigPath();
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf-8")) as Partial<Config>;
  } catch {
    return {};
  }
}

export function saveConfigFile(updates: Partial<Config>): void {
  const dir = getConfigDir();
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  const existing = loadConfigFile();
  const merged = { ...existing, ...updates };
  writeFileSync(getConfigPath(), JSON.stringify(merged, null, 2) + "\n");
}

export function showConfig(): Record<string, string> {
  const file = loadConfigFile();
  return {
    "Config file": getConfigPath(),
    "apiUrl": file.apiUrl ?? `(default) ${DEFAULT_API_URL}`,
    "token": file.token ? "***" + file.token.slice(-4) : "(not set)",
    "format": file.format ?? "(default) json",
    "env UHT_API_URL": process.env["UHT_API_URL"] ?? "(not set)",
    "env UHT_TOKEN": process.env["UHT_TOKEN"] ? "***" : "(not set)",
  };
}

export interface CliGlobalOpts {
  apiUrl?: string;
  token?: string;
  format?: string;
}

export function resolveConfig(opts: CliGlobalOpts): Config {
  const file = loadConfigFile();
  return {
    apiUrl:
      opts.apiUrl ??
      process.env["UHT_API_URL"] ??
      file.apiUrl ??
      DEFAULT_API_URL,
    token:
      opts.token ??
      process.env["UHT_TOKEN"] ??
      file.token ??
      "",
    format:
      (opts.format as Config["format"]) ??
      file.format ??
      "json",
  };
}
