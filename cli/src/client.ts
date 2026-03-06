import type { Config } from "./config.js";

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: string,
  ) {
    super(`API error ${status}: ${body}`);
    this.name = "ApiError";
  }
}

export class UHTClient {
  private baseUrl: string;
  private token: string | undefined;
  public verbose: boolean = false;

  constructor(config: Config) {
    this.baseUrl = config.apiUrl.replace(/\/+$/, "");
    this.token = config.token || undefined;
  }

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body?: Record<string, unknown>,
    query?: Record<string, string>,
  ): Promise<T> {
    // Ensure baseUrl ends with / and path doesn't start with / for correct URL joining
    const base = this.baseUrl.endsWith("/") ? this.baseUrl : this.baseUrl + "/";
    const relativePath = path.startsWith("/") ? path.slice(1) : path;
    const url = new URL(relativePath, base);
    if (query) {
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined && v !== "") url.searchParams.set(k, v);
      }
    }

    const headers: Record<string, string> = {};
    if (body) headers["Content-Type"] = "application/json";
    if (this.token) headers["Authorization"] = `Bearer ${this.token}`;

    if (this.verbose) {
      console.error(`${method} ${url.toString()}`);
      if (body) console.error(JSON.stringify(body, null, 2));
    }

    let response: Response;
    try {
      response = await fetch(url.toString(), {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (err) {
      throw new Error(
        `Network error: could not connect to ${this.baseUrl}. Check your API URL and network connection.`,
      );
    }

    if (!response.ok) {
      const text = await response.text();
      throw new ApiError(response.status, text);
    }

    return (await response.json()) as T;
  }

  // --- Info ---

  async root() {
    return this.request("GET", "/");
  }

  async info() {
    return this.request("GET", "/info");
  }

  async traits() {
    return this.request("GET", "/traits");
  }

  async patterns() {
    return this.request("GET", "/patterns");
  }

  // --- Classification ---

  async classify(entity: string, opts?: { context?: string; use_semantic_priors?: boolean }) {
    return this.request("POST", "/classify", {
      entity,
      context: opts?.context ?? "",
      use_semantic_priors: opts?.use_semantic_priors ?? false,
    });
  }

  // --- Comparison ---

  async compare(entityA: string, entityB: string) {
    return this.request("POST", "/compare", { entity_a: entityA, entity_b: entityB });
  }

  async batchCompare(entity: string, candidates: string[]) {
    return this.request("POST", "/batch-compare", { entity, candidates });
  }

  // --- Search ---

  async search(query: string, limit?: number) {
    return this.request("POST", "/search", { query, limit: limit ?? 10 });
  }

  async disambiguate(term: string, language?: string) {
    return this.request("POST", "/disambiguate", { term, language: language ?? "en" });
  }

  async semanticTriangle(text: string) {
    return this.request("POST", "/semantic-triangle", { text });
  }

  async mapPropertiesToTraits(properties: string[]) {
    return this.request("POST", "/map-properties-to-traits", { properties });
  }

  // --- Entities ---

  async listEntities(opts?: { name_contains?: string; hex_pattern?: string; limit?: number; offset?: number }) {
    const query: Record<string, string> = {};
    if (opts?.name_contains) query["name_contains"] = opts.name_contains;
    if (opts?.hex_pattern) query["hex_pattern"] = opts.hex_pattern;
    if (opts?.limit) query["limit"] = String(opts.limit);
    if (opts?.offset) query["offset"] = String(opts.offset);
    return this.request("GET", "/entities", undefined, query);
  }

  async deleteEntity(name: string, source?: string) {
    return this.request("POST", "/entities/delete", { name, source: source ?? "local" });
  }

  async inferProperties(entity: string) {
    return this.request("POST", "/entities/infer", { entity });
  }

  async exploreNeighborhood(entity: string, opts?: { metric?: string; limit?: number; min_similarity?: number }) {
    return this.request("POST", "/entities/explore", {
      entity,
      metric: opts?.metric ?? "embedding",
      limit: opts?.limit ?? 10,
      min_similarity: opts?.min_similarity ?? 0.3,
    });
  }

  async findSimilar(entity: string, opts?: { limit?: number; min_shared_traits?: number }) {
    return this.request("POST", "/entities/find-similar", {
      entity,
      limit: opts?.limit ?? 5,
      min_shared_traits: opts?.min_shared_traits ?? 20,
    });
  }

  async searchByTraits(traits: Record<string, string>) {
    return this.request("GET", "/search/traits", undefined, traits);
  }

  // --- Namespaces ---

  async createNamespace(code: string, name: string, description?: string) {
    return this.request("POST", "/namespaces/create", { code, name, description: description ?? "" });
  }

  async listNamespaces(opts?: { parent?: string; include_descendants?: boolean }) {
    return this.request("POST", "/namespaces/list", {
      parent: opts?.parent ?? "",
      include_descendants: opts?.include_descendants ?? false,
    });
  }

  async assignNamespace(entityName: string, namespace: string, primary?: boolean) {
    return this.request("POST", "/namespaces/assign", {
      entity_name: entityName,
      namespace,
      primary: primary ?? true,
    });
  }

  // --- Facts ---

  async storeFact(subject: string, predicate: string, objectValue: string, opts?: { user_id?: string; namespace?: string }) {
    return this.request("POST", "/facts/store", {
      subject,
      predicate,
      object_value: objectValue,
      user_id: opts?.user_id ?? "default",
      namespace: opts?.namespace ?? "",
    });
  }

  async storeFactsBulk(facts: Array<{ subject: string; predicate: string; object_value: string; user_id?: string; namespace?: string }>) {
    return this.request("POST", "/facts/store-bulk", { facts });
  }

  async upsertFact(subject: string, predicate: string, objectValue: string, opts?: { user_id?: string; namespace?: string }) {
    return this.request("POST", "/facts/upsert", {
      subject,
      predicate,
      object_value: objectValue,
      user_id: opts?.user_id ?? "default",
      namespace: opts?.namespace ?? "",
    });
  }

  async queryFacts(opts?: { subject?: string; object_value?: string; predicate?: string; category?: string; user_id?: string; namespace?: string; limit?: number }) {
    return this.request("POST", "/facts/query", {
      subject: opts?.subject ?? "",
      object_value: opts?.object_value ?? "",
      predicate: opts?.predicate ?? "",
      category: opts?.category ?? "",
      user_id: opts?.user_id ?? "",
      namespace: opts?.namespace ?? "",
      limit: opts?.limit ?? 20,
    });
  }

  async updateFact(factId: string, opts?: { subject?: string; predicate?: string; object_value?: string }) {
    return this.request("POST", "/facts/update", {
      fact_id: factId,
      subject: opts?.subject ?? "",
      predicate: opts?.predicate ?? "",
      object_value: opts?.object_value ?? "",
    });
  }

  async deleteFact(factId: string) {
    return this.request("POST", "/facts/delete", { fact_id: factId });
  }

  async getUserContext(userId?: string) {
    return this.request("POST", "/facts/user-context", { user_id: userId ?? "default" });
  }

  async getNamespaceContext(namespace: string, userId?: string) {
    return this.request("POST", "/facts/namespace-context", { namespace, user_id: userId ?? "" });
  }
}
