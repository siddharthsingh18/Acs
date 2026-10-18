/**
 * Web Push (PWA) client utilities for operator notifications.
 */
import { api } from '@/api/client'

export function isPushSupported(): boolean {
  return typeof window !== 'undefined' && 'serviceWorker' in navigator && 'PushManager' in window
}

export async function getPushSubscription(): Promise<PushSubscription | null> {
  if (!isPushSupported()) return null
  const registration = await navigator.serviceWorker.getRegistration()
  if (!registration) return null
  return await registration.pushManager.getSubscription()
}

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4)
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/')
  const rawData = window.atob(base64)
  const buffer = new ArrayBuffer(rawData.length)
  const outputArray = new Uint8Array(buffer)
  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i)
  }
  return outputArray
}

export async function registerServiceWorker(): Promise<ServiceWorkerRegistration | null> {
  if (!isPushSupported()) return null
  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    })
    return registration
  } catch (error) {
    console.error('Failed to register service worker:', error)
    return null
  }
}

export async function subscribeToPushNotifications(): Promise<{ success: boolean; error?: string }> {
  if (!isPushSupported()) {
    return { success: false, error: 'Web Push is not supported in this browser.' }
  }

  try {
    // Request notification permission from user
    const permission = await Notification.requestPermission()
    if (permission !== 'granted') {
      return { success: false, error: 'Push notification permission was denied.' }
    }

    // Register service worker if not already registered
    let registration = await navigator.serviceWorker.getRegistration()
    if (!registration) {
      registration = await registerServiceWorker()
    }
    if (!registration) {
      return { success: false, error: 'Could not initialize service worker.' }
    }

    // Fetch VAPID public key from backend
    const { public_key } = await api.getVapidPublicKey()
    if (!public_key) {
      return { success: false, error: 'Server did not return a valid VAPID public key.' }
    }

    const applicationServerKey = urlBase64ToUint8Array(public_key)

    // Subscribe via PushManager
    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: applicationServerKey as BufferSource,
    })

    const rawSub = subscription.toJSON()
    if (!rawSub.endpoint || !rawSub.keys?.p256dh || !rawSub.keys?.auth) {
      return { success: false, error: 'Push subscription keys are incomplete.' }
    }

    // Register subscription on backend
    await api.subscribeWebPush({
      endpoint: rawSub.endpoint,
      keys: {
        p256dh: rawSub.keys.p256dh,
        auth: rawSub.keys.auth,
      },
      user_agent: navigator.userAgent,
    })

    return { success: true }
  } catch (err: any) {
    console.error('Failed to subscribe to Web Push:', err)
    return { success: false, error: err.message || 'Error subscribing to notifications.' }
  }
}

export async function unsubscribeFromPushNotifications(): Promise<{ success: boolean; error?: string }> {
  if (!isPushSupported()) return { success: true }

  try {
    const registration = await navigator.serviceWorker.getRegistration()
    if (!registration) return { success: true }

    const subscription = await registration.pushManager.getSubscription()
    if (subscription) {
      const endpoint = subscription.endpoint
      await subscription.unsubscribe()
      await api.unsubscribeWebPush(endpoint).catch(() => {})
    }
    return { success: true }
  } catch (err: any) {
    console.error('Failed to unsubscribe from Web Push:', err)
    return { success: false, error: err.message }
  }
}
