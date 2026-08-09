/** Schema-bound form fields. One line of JSX per parameter:
 *
 *      <SchemaNumberField name="num_epochs" />
 *
 *  Label, bounds, step, enum options, default and description tooltip all
 *  come from the request contract via useSchemaField. Layout stays with the
 *  page — these are grid cells, not a form renderer.
 */

import { cn } from '@/lib/cn';
import { NumberInput, fieldCls } from '@/components/form/Field';
import { FieldShell } from './FieldLabel';
import { NullableNumberInput } from './NullableNumberInput';
import { SearchSelect, type SearchSelectOption } from './SearchSelect';
import { useSchemaField } from './SchemaForm';

interface CommonProps {
  name: string;
  /** Display label override; defaults to the schema title. */
  label?: string;
  className?: string;
}

export function SchemaNumberField({
  name,
  label,
  step,
  className,
}: CommonProps & { step?: number | 'any' }) {
  const { meta, value, set } = useSchemaField(name);
  const resolvedStep = step ?? (meta.type === 'integer' ? 1 : 'any');
  return (
    <FieldShell label={label ?? meta.title} description={meta.description} className={className}>
      {meta.nullable ? (
        <NullableNumberInput
          value={value as number | null}
          min={meta.min}
          max={meta.max}
          step={resolvedStep}
          onChange={set}
        />
      ) : (
        <NumberInput
          value={value as number}
          min={meta.min}
          max={meta.max}
          step={resolvedStep}
          onChange={set}
        />
      )}
    </FieldShell>
  );
}

export function SchemaSelectField({
  name,
  label,
  options,
  optionLabels,
  className,
}: CommonProps & {
  /** Curated subset override; defaults to the schema's enum. */
  options?: readonly (string | number)[];
  /** Display text per option value; defaults to the raw value. */
  optionLabels?: Record<string, string>;
}) {
  const { meta, value, set } = useSchemaField(name);
  const numeric = meta.type === 'integer' || meta.type === 'number';
  const opts = options ?? meta.options ?? [];
  return (
    <FieldShell label={label ?? meta.title} description={meta.description} className={className}>
      <select
        value={String(value)}
        onChange={(e) => set(numeric ? Number(e.target.value) : e.target.value)}
        className={cn(fieldCls, 'cursor-pointer appearance-none')}
      >
        {opts.map((option) => (
          <option key={option} value={option}>
            {optionLabels?.[String(option)] ?? option}
          </option>
        ))}
      </select>
    </FieldShell>
  );
}

export function SchemaTextField({
  name,
  label,
  placeholder,
  className,
}: CommonProps & { placeholder?: string }) {
  const { meta, value, set } = useSchemaField(name);
  return (
    <FieldShell label={label ?? meta.title} description={meta.description} className={className}>
      <input
        type="text"
        value={(value as string | null) ?? ''}
        placeholder={placeholder}
        onChange={(e) => set(meta.nullable && e.target.value === '' ? null : e.target.value)}
        className={fieldCls}
      />
    </FieldShell>
  );
}

export function SchemaCheckboxField({ name, label, className }: CommonProps) {
  const { meta, value, set } = useSchemaField(name);
  return (
    <label
      title={meta.description}
      className={cn(
        'inline-flex cursor-pointer items-center gap-2 text-xs text-text-secondary',
        className,
      )}
    >
      <input
        type="checkbox"
        checked={Boolean(value)}
        onChange={(e) => set(e.target.checked)}
        className="h-3.5 w-3.5 accent-primary"
      />
      {label ?? meta.title}
    </label>
  );
}

/** Searchable select over runtime options (checkpoints, datasets) for a
 *  nullable string field — '' in the widget is null in the form. */
export function SchemaSearchSelectField({
  name,
  label,
  options,
  placeholder,
  className,
}: CommonProps & { options: SearchSelectOption[]; placeholder?: string }) {
  const { meta, value, set } = useSchemaField(name);
  return (
    <FieldShell label={label ?? meta.title} description={meta.description} className={className}>
      <SearchSelect
        value={(value as string | null) ?? ''}
        onChange={(next) => set(meta.nullable && next === '' ? null : next)}
        options={options}
        placeholder={placeholder}
      />
    </FieldShell>
  );
}
