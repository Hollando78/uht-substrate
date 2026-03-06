/** Trait value from a classification. */
export interface TraitValue {
  bit_position: number;
  name: string;
  present: boolean;
  confidence: number;
  justification?: string;
}

/** Classification result. */
export interface ClassificationResult {
  uuid: string;
  name: string;
  hex_code: string;
  binary: string;
  traits: TraitValue[];
  created_at: string;
  inferred_properties?: Record<string, unknown>;
}

/** Comparison result. */
export interface CompareResult {
  entity_a: string;
  entity_b: string;
  hex_a: string;
  hex_b: string;
  jaccard_similarity: number;
  hamming_distance: number;
  shared_traits: string[];
  traits_a_only: string[];
  traits_b_only: string[];
  analysis: string;
  trait_level_diff?: Record<string, unknown>;
}

/** Semantic search result. */
export interface SearchResult {
  uuid: string;
  name: string;
  description?: string;
  hex_code: string;
  similarity_score: number;
}

/** Entity listing. */
export interface Entity {
  uuid: string;
  name: string;
  hex_code: string;
  binary?: string;
  description?: string;
  aliases?: string[];
  traits?: TraitValue[];
  created_at?: string;
}

/** Fact record. */
export interface Fact {
  id: string;
  subject: string;
  predicate: string;
  object_value: string;
  category: string;
  user_id: string;
  namespace?: string;
  created_at: string;
}

/** Namespace record. */
export interface Namespace {
  code: string;
  name: string;
  description?: string;
  parent?: string;
  entity_count?: number;
}

/** API error response. */
export interface ApiErrorResponse {
  error: string;
  detail?: string;
}
