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

  test('switches to Live Camera Wall and interacts with 1x1, 2x2, 3x3 layout', async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem('acs_demo_mode', 'true')
      localStorage.setItem('acs_theme', 'dark')
    })

    await page.goto('/live')
    const wallButton = page.getByRole('button', { name: /Live Camera Wall/i })
    await expect(wallButton).toBeVisible()
    await wallButton.click()

    // Assert grid buttons are visible
    await expect(page.getByRole('button', { name: /1x1/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /2x2/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /3x3/i })).toBeVisible()

    // Assert Add Camera button is visible
    const addCamBtn = page.getByRole('button', { name: /Add Camera/i })
    await expect(addCamBtn).toBeVisible()
  })

  test('navigates to /map, verifies Leaflet map viewport, layer toggles, replay slider, and RTL assurance', async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem('acs_demo_mode', 'true')
      localStorage.setItem('acs_theme', 'dark')
    })

    // 1. Navigate to /map
    await page.goto('/map')

    // 2. Expect Map Header and OSM attribution badge
    await expect(page.locator('h1')).toContainText(/Interactive CCTV Map/i)
    await expect(page.getByText('OSM Tiles')).toBeVisible()

    // 3. Verify Leaflet map container has strict LTR direction
    const mapViewport = page.locator('#tactical-map-viewport')
    await expect(mapViewport).toBeVisible()
    await expect(mapViewport).toHaveAttribute('dir', 'ltr')

    // 4. Assert layer controls are present
    await expect(page.getByTitle(/Toggle Collision Hotspot Heatmap/i)).toBeVisible()
    await expect(page.getByTitle(/Toggle Marker Clustering/i)).toBeVisible()
    await expect(page.getByTitle(/Toggle Incidents/i)).toBeVisible()
    await expect(page.getByTitle(/Toggle Tactical Dark Mode Tiles/i)).toBeVisible()

    // 5. Assert bottom replay slider dock
    await expect(page.getByText(/Incident Chronological Replay/i)).toBeVisible()
    const replayModeToggle = page.getByRole('button', { name: /Show All Incidents/i })
    await expect(replayModeToggle).toBeVisible()
    await replayModeToggle.click()
    await expect(page.getByRole('button', { name: /Replay Mode/i })).toBeVisible()

    // 6. Test RTL assurance: Switch to Arabic and assert map container remains LTR
    await page.evaluate(() => {
      window.localStorage.setItem('i18nextLng', 'ar')
      window.location.reload()
    })
    await page.waitForLoadState('networkidle')

    const rtlMapViewport = page.locator('#tactical-map-viewport')
    await expect(rtlMapViewport).toBeVisible()
    await expect(rtlMapViewport).toHaveAttribute('dir', 'ltr')
  })
})
