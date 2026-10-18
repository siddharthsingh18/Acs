import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { BottomNav } from './BottomNav'

describe('BottomNav Component', () => {
  it('renders mobile navigation links with ARIA labels and destinations', () => {
    render(
      <MemoryRouter>
        <BottomNav />
      </MemoryRouter>
    )

    const nav = screen.getByRole('navigation', { name: /Mobile Bottom Navigation/i })
    expect(nav).toBeInTheDocument()

    // Expect navigation items
    expect(screen.getByText(/Live Analysis/i)).toBeInTheDocument()
    expect(screen.getByText(/Review Queue/i)).toBeInTheDocument()
    expect(screen.getByText(/Incident Logs/i)).toBeInTheDocument()
    expect(screen.getByText(/Analytics/i)).toBeInTheDocument()
    expect(screen.getByText(/System Settings/i)).toBeInTheDocument()
    expect(screen.getByText(/System Architecture/i)).toBeInTheDocument()
  })
})
