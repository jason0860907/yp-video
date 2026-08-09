/** Schema-driven form state over the generated train request contracts.
 *
 *  contracts/*_request.schema.json (emitted from the backend pydantic models
 *  by `make contract`) carry every parameter's default, bounds, enum options
 *  and description. The train pages build their config forms from that one
 *  source of truth: `useSchemaForm` turns a schema into form state seeded
 *  with the backend defaults, and the components/form fields read FieldMeta
 *  to render labelled, bounded, documented inputs. The form values object is
 *  the request payload — the models reject unknown fields, so `body: values`
 *  round-trips by construction.
 */

import { useCallback, useState } from 'react';

export interface FieldMeta {
  type: 'integer' | 'number' | 'string' | 'boolean' | 'array';
  title: string;
  description?: string;
  /** Enum options, when the backend field is a Literal. */
  options?: string[];
  /** Fixed value (e.g. the discriminator `source`) — never rendered. */
  const?: unknown;
  default?: unknown;
  /** Inclusive bounds (pydantic ge/le) — safe to clamp to. */
  min?: number;
  max?: number;
  /** Exclusive bounds (pydantic gt/lt) — hints only, never clamp onto them. */
  exclusiveMin?: number;
  exclusiveMax?: number;
  nullable: boolean;
  required: boolean;
}

type RawProperty = Record<string, unknown>;

interface RawSchema {
  properties?: Record<string, RawProperty>;
  required?: string[];
}

function parseProperty(prop: RawProperty, required: boolean): FieldMeta {
  // pydantic emits `T | None` as anyOf [T-branch, {type: null}], keeping
  // default/title/description at the top level — fold the branch back in.
  let core = prop;
  let nullable = false;
  const anyOf = prop.anyOf as RawProperty[] | undefined;
  if (anyOf) {
    const nonNull = anyOf.filter((branch) => branch.type !== 'null');
    nullable = nonNull.length < anyOf.length;
    core = { ...nonNull[0], ...prop };
  }
  return {
    type: core.type as FieldMeta['type'],
    title: (core.title as string | undefined) ?? '',
    description: core.description as string | undefined,
    options: core.enum as string[] | undefined,
    const: core.const,
    default: core.default,
    min: core.minimum as number | undefined,
    max: core.maximum as number | undefined,
    exclusiveMin: core.exclusiveMinimum as number | undefined,
    exclusiveMax: core.exclusiveMaximum as number | undefined,
    nullable,
    required,
  };
}

export function parseRequestSchema(schema: unknown): Record<string, FieldMeta> {
  const { properties = {}, required = [] } = schema as RawSchema;
  const fields: Record<string, FieldMeta> = {};
  for (const [name, prop] of Object.entries(properties)) {
    fields[name] = parseProperty(prop, required.includes(name));
  }
  return fields;
}

/** The backend defaults as a complete, valid form value object. Fields with
 *  neither a default nor a const (required inputs like `dataset`) must come
 *  from `seed` — a missing one is a programming error, so it throws. */
export function buildDefaults<T extends object>(
  fields: Record<string, FieldMeta>,
  seed?: Partial<T>,
): T {
  const values: Record<string, unknown> = {};
  for (const [name, meta] of Object.entries(fields)) {
    const seeded = seed?.[name as keyof T];
    if (seeded !== undefined) values[name] = seeded;
    else if (meta.default !== undefined) values[name] = meta.default;
    else if (meta.const !== undefined) values[name] = meta.const;
    else if (meta.type === 'array') values[name] = [];
    else if (meta.nullable) values[name] = null;
    else throw new Error(`buildDefaults: field "${name}" needs a seed value`);
  }
  return values as T;
}

export interface SchemaFormState<T extends object> {
  values: T;
  set: <K extends keyof T>(key: K, value: T[K]) => void;
  reset: () => void;
  fields: Record<string, FieldMeta>;
}

export function useSchemaForm<T extends object>(
  schema: unknown,
  seed?: Partial<T>,
): SchemaFormState<T> {
  // Schema and seed are fixed for the page's lifetime; parse once.
  const [statics] = useState(() => {
    const fields = parseRequestSchema(schema);
    return { fields, initial: buildDefaults<T>(fields, seed) };
  });
  const [values, setValues] = useState<T>(statics.initial);
  const set = useCallback(
    <K extends keyof T>(key: K, value: T[K]) =>
      setValues((current) => ({ ...current, [key]: value })),
    [],
  );
  const reset = useCallback(() => setValues(statics.initial), [statics]);
  return { values, set, reset, fields: statics.fields };
}
