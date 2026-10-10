import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { CommandPalette } from './CommandPalette'

describe('CommandPalette Component', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <MemoryRouter>
        <CommandPalette isOpen={false} onClose={vi.fn()} />
      </MemoryRouter>
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders search input and commands when open', () => {
    render(
      <MemoryRouter>
        <CommandPalette isOpen={true} onClose={vi.fn()} />
      </MemoryRouter>
    )

    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(
      screen.getByPlaceholderText(/Type a command or jump to page/i)
    ).toBeInTheDocument()
    expect(screen.getByText(/Live Analysis/i)).toBeInTheDocument()
    expect(screen.getByText(/Incident Logs/i)).toBeInTheDocument()
    expect(screen.getByText(/Analytics/i)).toBeInTheDocument()
  })

  it('filters commands when user types search query', () => {
    render(
      <MemoryRouter>
        <CommandPalette isOpen={true} onClose={vi.fn()} />
      </MemoryRouter>
    )

    const input = screen.getByPlaceholderText(/Type a command or jump to page/i)
    fireEvent.change(input, { target: { value: 'demo' } })

    expect(screen.getByText(/Presentation Demo Mode/i)).toBeInTheDocument()
    expect(screen.queryByText(/Incident Logs/i)).not.toBeInTheDocument()
  })

  it('calls onClose when Escape is pressed', () => {
    const onClose = vi.fn()
    render(
      <MemoryRouter>
        <CommandPalette isOpen={true} onClose={onClose} />
      </MemoryRouter>
    )

    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })
})
