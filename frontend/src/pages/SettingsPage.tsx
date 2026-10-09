import React, { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Sliders,
  Bell,
  Cpu,
  Send,
  CheckCircle2,
  Save,
  Radio,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { api, type AlertSettingsUpdateRequest } from '@/api/client'
import { useAppStore } from '@/store/appStore'

export const SettingsPage: React.FC = () => {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const { activeModel, setActiveModel } = useAppStore()

  // Form states for alerts
  const [enabled, setEnabled] = useState(true)
  const [dryRun, setDryRun] = useState(false)
  const [channels, setChannels] = useState<string[]>(['console'])
  const [testSuccessMsg, setTestSuccessMsg] = useState<string | null>(null)
  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null)

  // System config query
  const { data: config } = useQuery({
    queryKey: ['config'],
    queryFn: () => api.getConfig(),
  })

  // Alert settings query
  const { data: alertSettings } = useQuery({
    queryKey: ['alertSettings'],
    queryFn: () => api.getAlertSettings(),
  })

  useEffect(() => {
    if (alertSettings) {
      setEnabled(alertSettings.enabled)
      setDryRun(alertSettings.dry_run)
      setChannels(alertSettings.channels || ['console'])
    }
  }, [alertSettings])

  // Update Settings Mutation
  const updateMutation = useMutation({
    mutationFn: (newSettings: AlertSettingsUpdateRequest) =>
      api.updateAlertSettings(newSettings),
    onSuccess: () => {
      setSaveSuccessMsg('Settings saved successfully.')
      setTimeout(() => setSaveSuccessMsg(null), 3000)
      queryClient.invalidateQueries({ queryKey: ['alertSettings'] })
    },
  })

  // Test Alert Mutation
  const testMutation = useMutation({
    mutationFn: () => api.testAlerts(dryRun, 'en', channels),
    onSuccess: (data) => {
      const summary = Object.entries(data.results || {})
        .map(([k, v]) => `${k}: ${v ? 'OK' : 'Skipped/Failed'}`)
        .join(', ')
      setTestSuccessMsg(`Dispatched! (${summary})`)
      setTimeout(() => setTestSuccessMsg(null), 4000)
    },
  })

  const toggleChannel = (channel: string) => {
    setChannels((prev) =>
      prev.includes(channel) ? prev.filter((c) => c !== channel) : [...prev, channel]
    )
  }

  const handleSave = () => {
    updateMutation.mutate({
      enabled,
      dry_run: dryRun,
      channels,
    })
  }

  const ALL_CHANNELS = [
    { id: 'console', labelKey: 'settings.channel_console' },
    { id: 'webhook', labelKey: 'settings.channel_webhook' },
    { id: 'telegram', labelKey: 'settings.channel_telegram' },
    { id: 'sms', labelKey: 'settings.channel_sms' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2.5">
          <Sliders className="h-7 w-7 text-primary" />
          {t('settings.header')}
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {t('settings.subtitle')}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model & Vision Engine Settings */}
        <Card className="border-border">
          <CardHeader className="py-4 px-6 border-b border-border">
            <CardTitle className="text-base font-bold flex items-center gap-2">
              <Cpu className="h-4 w-4 text-cyan-400" />
              {t('settings.model_settings')}
            </CardTitle>
            <CardDescription className="text-xs">
              Configured YOLO detection model weights and hardware accelerator.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6 space-y-4">
            <div>
              <label className="text-xs font-semibold text-muted-foreground block mb-2">
                {t('settings.model_select')}
              </label>
              <select
                value={activeModel}
                onChange={(e) => setActiveModel(e.target.value)}
                className="w-full px-3 py-2 rounded-md border border-border bg-background text-xs font-mono text-foreground focus:ring-1 focus:ring-primary outline-none"
              >
                {config?.available_models && config.available_models.length > 0 ? (
                  config.available_models.map((m: string) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))
                ) : (
                  <option value="best.pt">best.pt (YOLOv11)</option>
                )}
              </select>
            </div>

            <div className="p-4 rounded-lg border border-border bg-muted/30 space-y-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Active Device:</span>
                <span className="font-bold text-foreground uppercase">
                  {config?.device_info?.device || 'MPS (Apple Silicon)'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Default Confidence:</span>
                <span className="font-bold text-cyan-400">
                  {config?.defaults?.confidence_threshold
                    ? `${(config.defaults.confidence_threshold * 100).toFixed(0)}%`
                    : '25%'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Injury Model:</span>
                <span className="text-foreground">
                  {config?.defaults?.injury_model_available
                    ? 'Available'
                    : 'Disabled (injury.pt not found)'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Emergency Alert Settings */}
        <Card className="border-border">
          <CardHeader className="py-4 px-6 border-b border-border">
            <CardTitle className="text-base font-bold flex items-center gap-2">
              <Bell className="h-4 w-4 text-[var(--status-alert)]" />
              {t('settings.alerts_settings')}
            </CardTitle>
            <CardDescription className="text-xs">
              Multi-channel automated incident dispatch and siren testing.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6 space-y-5">
            {/* Toggles */}
            <div className="space-y-3">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(e) => setEnabled(e.target.checked)}
                  className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                />
                <span className="text-xs font-semibold text-foreground">
                  {t('settings.enable_alerts')}
                </span>
              </label>

              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={dryRun}
                  onChange={(e) => setDryRun(e.target.checked)}
                  className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                />
                <div>
                  <span className="text-xs font-semibold text-foreground block">
                    {t('settings.dry_run')}
                  </span>
                  <span className="text-[11px] text-muted-foreground">
                    Simulates dispatch in logs without sending live messages.
                  </span>
                </div>
              </label>
            </div>

            {/* Channels Checklist */}
            <div className="space-y-2 pt-2 border-t border-border">
              <label className="text-xs font-semibold text-muted-foreground block">
                {t('settings.channels')}:
              </label>
              <div className="grid grid-cols-2 gap-2">
                {ALL_CHANNELS.map((ch) => (
                  <label
                    key={ch.id}
                    className="flex items-center gap-2 p-2.5 rounded-lg border border-border bg-background/50 hover:bg-muted/40 cursor-pointer transition-colors"
                  >
                    <input
                      type="checkbox"
                      checked={channels.includes(ch.id)}
                      onChange={() => toggleChannel(ch.id)}
                      className="rounded border-border text-primary focus:ring-primary h-3.5 w-3.5"
                    />
                    <span className="text-xs font-medium text-foreground">
                      {t(ch.labelKey)}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            {/* Actions & Feedback */}
            <div className="flex flex-col sm:flex-row items-center gap-3 pt-3 border-t border-border">
              <Button
                onClick={handleSave}
                disabled={updateMutation.isPending}
                className="w-full sm:w-auto gap-2 text-xs font-bold"
              >
                <Save className="h-3.5 w-3.5" />
                {t('settings.save_settings')}
              </Button>

              <Button
                variant="outline"
                onClick={() => testMutation.mutate()}
                disabled={testMutation.isPending}
                className="w-full sm:w-auto gap-2 text-xs font-semibold border-border hover:border-cyan-500 hover:text-cyan-400"
              >
                <Send className="h-3.5 w-3.5" />
                {testMutation.isPending ? t('settings.testing_alert') : t('settings.test_alert')}
              </Button>
            </div>

            {saveSuccessMsg && (
              <div className="p-3 rounded-md bg-[var(--status-ok)]/10 text-[var(--status-ok)] text-xs flex items-center gap-2 font-medium">
                <CheckCircle2 className="h-4 w-4" />
                {saveSuccessMsg}
              </div>
            )}

            {testSuccessMsg && (
              <div className="p-3 rounded-md bg-cyan-500/10 text-cyan-400 text-xs flex items-center gap-2 font-mono">
                <Radio className="h-4 w-4 animate-pulse" />
                {testSuccessMsg}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
