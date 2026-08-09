/** Context plumbing for schema-bound form fields.
 *
 *  A page wraps its config card in `<SchemaForm form={useSchemaForm(...)}>`;
 *  the Schema*Field components then reach any field by name — value, setter
 *  and FieldMeta (label, bounds, options, description) all come from the
 *  contract, so a field renders from one line of JSX.
 */

import { createContext, useContext, type ReactNode } from 'react';
import type { FieldMeta, SchemaFormState } from '@/lib/schemaForm';

interface SchemaFormContextValue {
  values: Record<string, unknown>;
  set: (key: string, value: unknown) => void;
  fields: Record<string, FieldMeta>;
}

const SchemaFormContext = createContext<SchemaFormContextValue | null>(null);

export function SchemaForm<T extends object>({
  form,
  children,
}: {
  form: SchemaFormState<T>;
  children: ReactNode;
}) {
  return (
    <SchemaFormContext.Provider value={form as unknown as SchemaFormContextValue}>
      {children}
    </SchemaFormContext.Provider>
  );
}

export interface SchemaFieldHandle {
  meta: FieldMeta;
  value: unknown;
  set: (value: unknown) => void;
}

/** A field's value/setter/metadata by schema property name. Throws on names
 *  the contract doesn't know — a typo should fail loudly, not render an
 *  empty input. */
export function useSchemaField(name: string): SchemaFieldHandle {
  const ctx = useContext(SchemaFormContext);
  if (!ctx) throw new Error('useSchemaField must be used inside <SchemaForm>');
  const meta = ctx.fields[name];
  if (!meta) throw new Error(`Unknown schema field: ${name}`);
  return { meta, value: ctx.values[name], set: (value) => ctx.set(name, value) };
}
