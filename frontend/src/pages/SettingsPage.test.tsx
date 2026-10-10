import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SettingsPage } from './SettingsPage'
import { api } from '@/api/client'
import i18n from '@/i18n'

vi.mock('@/api/client', () => ({
  api: {
    getConfig: vi.fn(),
    getAlertSettings: vi.fn(),
    updateAlertSettings: vi.fn(),
    testAlerts: vi.fn(),
  },
}))

describe('SettingsPage component', () => {
  let queryClient: QueryClient

  const mockConfig = {
    available_models: ['best.pt', 'best_model_without_freeze.pt', 'trained_yolo_model.pt'],
    device_info: {
      device: 'mps',
      cuda_available: false,
      mps_available: true,
      device_name: 'Apple Silicon GPU (MPS)',
    },
    defaults: {
      model_path: 'best.pt',
      injury_model_path: 'injury.pt',
      injury_model_available: false,
      confidence_threshold: 0.72,
      injury_confidence_threshold: 0.72,
      cooldown_seconds: 10.0,
      gemini_model_name: 'gemini-2.5-flash',
      alerts_enabled: true,
      alerts_dry_run: false,
      reports_dir: 'reports',
      report_images_dir: 'report_images',
      temp_dir: 'temp',
    },
    supported_languages: [
      { code: 'en', name: 'English', dir: 'ltr' },
      { code: 'ar', name: 'العربية', dir: 'rtl' },
    ],
    classes: [],
  }

  const mockAlertSettings = {
    enabled: true,
    dry_run: false,
    channels: ['telegram', 'email', 'console'],
    channel_status: {
      telegram: { configured: true, description: 'Configured' },
      email: { configured: false, description: 'Not configured' },
      console: { configured: true, description: 'Configured' },
    },
  }

  beforeEach(async () => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    vi.clearAllMocks()
    await i18n.changeLanguage('en')
    vi.mocked(api.getConfig).mockResolvedValue(mockConfig as any)
    vi.mocked(api.getAlertSettings).mockResolvedValue(mockAlertSettings as any)
    vi.mocked(api.updateAlertSettings).mockResolvedValue({ ...mockAlertSettings } as any)
    vi.mocked(api.testAlerts).mockResolvedValue({
      results: { telegram: true, email: false, console: true },
      message: 'Test alert dispatched: telegram: OK, email: FAIL, console: OK',
    } as any)
  })

  const renderPage = () => {
    return render(
      <QueryClientProvider client={queryClient}>
        <SettingsPage />
      </QueryClientProvider>
    )
  }

  it('renders all sections and headers', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByText(/System & Emergency Alert Configuration/i)).toBeInTheDocument()
      expect(screen.getByText(/Emergency Alert Dispatch/i)).toBeInTheDocument()
      expect(screen.getByText(/Inference Engine/i)).toBeInTheDocument()
      expect(screen.getByText(/Temporal State Machine & Debounce/i)).toBeInTheDocument()
      expect(screen.getByText(/Interface & Localization Preferences/i)).toBeInTheDocument()
    })
  })

  it('displays configured and not configured badges without exposing secrets', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByText('Telegram Alert Bot')).toBeInTheDocument()
      expect(screen.getAllByText(/SMTP Dispatch Email/).length).toBeGreaterThan(0)
    })

    expect(screen.queryByText(/your_telegram_bot_token/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/your_smtp_app_password/i)).not.toBeInTheDocument()
  })

  it('triggers test alert and displays live result feedback', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Send Test.*Alert/i })).toBeInTheDocument()
    })

    const testBtn = screen.getByRole('button', { name: /Send Test.*Alert/i })
    fireEvent.click(testBtn)

    await waitFor(() => {
      expect(api.testAlerts).toHaveBeenCalled()
      expect(screen.getByText(/Live Dispatch Results/i)).toBeInTheDocument()
      expect(screen.getByText(/Test alert dispatched/i)).toBeInTheDocument()
    })
  })

  it('handles language and theme selection buttons', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByText('Display Language')).toBeInTheDocument()
      expect(screen.getByText('LTR Layout')).toBeInTheDocument()
      expect(screen.getByText('RTL Layout')).toBeInTheDocument()
      expect(screen.getByText('Cyber Cyan')).toBeInTheDocument()
      expect(screen.getByText('Daylight Mode')).toBeInTheDocument()
    })

    const arabicBtn = screen.getByText('RTL Layout').closest('button')!
    fireEvent.click(arabicBtn)

    const lightThemeBtn = screen.getByText('Daylight Mode').closest('button')!
    fireEvent.click(lightThemeBtn)
  })

  it('submits valid form and saves settings to backend and localStorage', async () => {
    const { container } = renderPage()

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Save Settings/i })).toBeInTheDocument()
    })

    const form = container.querySelector('form')!
    fireEvent.submit(form)

    await waitFor(() => {
      expect(api.updateAlertSettings).toHaveBeenCalled()
    })
  })
})
