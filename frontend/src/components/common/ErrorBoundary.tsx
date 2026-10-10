import { Component, type ErrorInfo, type ReactNode } from 'react'
import { ErrorState } from './ErrorState'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught component error in ACS Control Center:', error, errorInfo)
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-[50vh] flex items-center justify-center p-6">
          <ErrorState
            title="Application Render Exception"
            message={this.state.error?.message || 'An unhandled React lifecycle exception occurred.'}
            onRetry={() => {
              this.setState({ hasError: false, error: null })
              window.location.reload()
            }}
          />
        </div>
      )
    }

    return this.props.children
  }
}
