import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { CameraLocationPicker } from './CameraLocationPicker'

const { leafletMock } = vi.hoisted(() => {
  const mockMap = {
    setView: vi.fn().mockReturnThis(),
    invalidateSize: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    remove: vi.fn().mockReturnThis(),
  }

  const mockMarker = {
    addTo: vi.fn().mockReturnThis(),
    setLatLng: vi.fn().mockReturnThis(),
    getLatLng: vi.fn().mockReturnValue({ lat: 24.7136, lng: 46.6753 }),
    on: vi.fn().mockReturnThis(),
    remove: vi.fn().mockReturnThis(),
  }

  const leafletMock = {
    map: vi.fn(() => mockMap),
    tileLayer: vi.fn(() => ({ addTo: vi.fn() })),
    divIcon: vi.fn((opts: any) => opts),
    marker: vi.fn(() => mockMarker),
  }

  return { leafletMock }
})

vi.mock('leaflet', () => ({
  default: leafletMock,
  ...leafletMock,
}))

vi.mock('@/lib/leafletSetup', () => ({
  default: leafletMock,
  L: leafletMock,
}))

describe('CameraLocationPicker Component', () => {
  it('renders coordinate badge and strict LTR map container', () => {
    const handleChange = vi.fn()
    const { container } = render(
      <CameraLocationPicker lat={24.7136} lng={46.6753} onChange={handleChange} />
    )

    expect(screen.getByText(/Lat: 24.71360, Lng: 46.67530/i)).toBeInTheDocument()
    expect(screen.getByText('Pin Active')).toBeInTheDocument()

    // Map container must have strict LTR direction
    const mapDiv = container.querySelector('[dir="ltr"]')
    expect(mapDiv).not.toBeNull()
    expect(mapDiv?.getAttribute('dir')).toBe('ltr')
  })

  it('renders quick presets and allows fast coordinate assignment', () => {
    const handleChange = vi.fn()
    render(<CameraLocationPicker onChange={handleChange} />)

    const presetBtn = screen.getByText('Riyadh HQ')
    expect(presetBtn).toBeInTheDocument()

    fireEvent.click(presetBtn)
    expect(handleChange).toHaveBeenCalledWith({ lat: 24.7136, lng: 46.6753 })
  })

  it('allows clearing coordinates', () => {
    const handleChange = vi.fn()
    render(<CameraLocationPicker lat={24.7136} lng={46.6753} onChange={handleChange} />)

    const clearBtn = screen.getByTitle('Remove marker')
    expect(clearBtn).toBeInTheDocument()

    fireEvent.click(clearBtn)
    expect(handleChange).toHaveBeenCalledWith(null)
  })
})
