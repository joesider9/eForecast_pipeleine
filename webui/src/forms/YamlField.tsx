import { FieldProps } from '@rjsf/utils'
import CodeMirror from '@uiw/react-codemirror'
import { yaml as yamlLanguage } from '@codemirror/lang-yaml'
import { Box, Button, FormHelperText, Typography } from '@mui/material'
import type { FormContext } from '../fields/context'
import { useYamlBuffer } from './useYamlBuffer'

// A YAML box for values the generated form cannot express well (dicts, unions, untyped lists).
// Raw text and parse errors live in the case draft, so tab unmounts cannot discard
// unfinished work. The parsed document keeps its last valid value until repair.
export default function YamlField(props: FieldProps) {
  const { formData, onChange, name, schema, required } = props
  const section = (props.formContext as FormContext | undefined)?.yamlSection ?? ''
  const label = schema.title ?? name
  const { text, error, change, discard } = useYamlBuffer(
    JSON.stringify(['field', section, props.idSchema.$id]), label, section, formData, onChange)

  return (
    <Box sx={{ mb: 1.5 }}>
      <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', mb: 0.5 }}>
        {schema.title ?? name}{required ? ' *' : ''} <span style={{ opacity: 0.6 }}>(YAML)</span>
      </Typography>
      <CodeMirror value={text} height="auto" minHeight="60px" extensions={[yamlLanguage()]} onChange={change}
        onCreateEditor={view => view.contentDOM.setAttribute('aria-label', `${schema.title ?? name} YAML`)}
        basicSetup={{ lineNumbers: false, foldGutter: false, highlightActiveLine: false }} />
      {schema.description && <FormHelperText>{schema.description}</FormHelperText>}
      {error && <FormHelperText error>{error}</FormHelperText>}
      {error && <Button size="small" onClick={discard}>Discard YAML edits</Button>}
    </Box>
  )
}
