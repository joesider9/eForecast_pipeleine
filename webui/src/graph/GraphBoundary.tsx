import { Component, type ReactNode } from 'react'
import { Alert, Button } from '@mui/material'

// The case owns the document and YAML buffers above this boundary. An unexpected
// graph failure must not take down those drafts or the rest of the case editor.
export default class GraphBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() { return { failed: true } }

  render() {
    if (this.state.failed) return <Alert severity="error"
      action={<Button onClick={() => this.setState({ failed: false })}>Try editor again</Button>}>
      The experiment editor could not render. Your case draft and YAML edits are retained.
    </Alert>
    return this.props.children
  }
}
