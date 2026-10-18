import { test, expect } from '@playwright/test'

test.describe('Live Surveillance Smoke Test', () => {
  test('opens /live, runs sample video stream, and asserts an incident appears', async ({ page }) => {
    // Enable presentation demo mode via localStorage for fast, deterministic e2e smoke testing
    await page.addInitScript(() => {
      localStorage.setItem('acs_demo_mode', 'true')
      localStorage.setItem('acs_theme', 'dark')
    })

    // 1. Open /live page
    await page.goto('/live')

    // Expect page title to be visible
    await expect(page.locator('h1')).toContainText(/Automated CCTV Video Analysis/i)

    // 2. Select Sample Footage source (selected by default)
    const sampleButton = page.getByRole('button', { name: /Sample Footage/i })
    await expect(sampleButton).toBeVisible()

    // 3. Click Start Inference Stream
    const startButton = page.getByRole('button', { name: /Start Inference Stream/i })
    await expect(startButton).toBeVisible()
    await startButton.click()

    // 4. Assert stream transitions to active state
    const stopButton = page.getByRole('button', { name: /Stop Stream/i })
    await expect(stopButton).toBeVisible()

    // 5. Assert an incident appears in the Live Incident Feed
    const incidentCard = page.locator('text=INC-DEMO-2026-001').first()
    await expect(incidentCard).toBeVisible({ timeout: 10000 })

    // Check incident details in sidebar
    await expect(page.locator('text=Car-Car Collision').first()).toBeVisible()

    // 6. Click the incident card to assert IncidentDrawer opens
    await incidentCard.click()
    const drawer = page.getByRole('dialog')
    await expect(drawer).toBeVisible()
    await expect(drawer).toContainText('INC-DEMO-2026-001')
  })
})
