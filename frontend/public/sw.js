// Service Worker for ACS Web Push Notifications (PWA)
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('push', (event) => {
  let data = {};
  if (event.data) {
    try {
      data = event.data.json();
    } catch (e) {
      data = { title: 'ACS Incident Alert', body: event.data.text() };
    }
  }

  const title = data.title || '🚨 ACS Emergency Dispatch Alert';
  const options = {
    body: data.body || 'A high-confidence traffic accident requires review.',
    icon: data.icon || '/favicon.svg',
    badge: '/favicon.svg',
    image: data.image || undefined,
    tag: data.tag || 'acs-incident-alert',
    renotify: true,
    requireInteraction: true,
    data: data.data || {},
    actions: [
      { action: 'ack', title: '✅ Acknowledge' },
      { action: 'open', title: '🔍 View Incident' }
    ]
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const data = event.notification.data || {};

  if (event.action === 'ack' && data.ack_url) {
    // One-click acknowledgment directly from the mobile/desktop lockscreen
    event.waitUntil(
      fetch(data.ack_url, { headers: { 'Accept': 'application/json' } })
        .catch((err) => console.error('One-click ack failed:', err))
        .then(() => {
          return self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
            if (clientList.length > 0) {
              return clientList[0].focus();
            }
            return self.clients.openWindow('/');
          });
        })
    );
    return;
  }

  // Default action or 'open' clicked: open or focus dashboard
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if (client.url && 'focus' in client) {
          return client.focus();
        }
      }
      if (self.clients.openWindow) {
        return self.clients.openWindow('/');
      }
    })
  );
});
