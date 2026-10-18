import { create } from 'zustand'

export type UserRole = 'viewer' | 'operator' | 'admin'

export interface AuthUser {
  id: string
  username: string
  full_name: string
  role: UserRole
  is_active: boolean
  created_at: number
  last_login_at?: number | null
}

interface AuthState {
  user: AuthUser | null
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean

  login: (username: string, password: string) => Promise<{ success: boolean; error?: string }>
  logout: () => Promise<void>
  refresh: () => Promise<boolean>
  fetchCurrentUser: () => Promise<void>
  setUser: (user: AuthUser | null) => void
}

const getSavedToken = (): string | null => {
  return localStorage.getItem('acs_access_token')
}

const getSavedRefreshToken = (): string | null => {
  return localStorage.getItem('acs_refresh_token')
}

const getSavedUser = (): AuthUser | null => {
  const raw = localStorage.getItem('acs_user')
  if (!raw) return null
  try {
    return JSON.parse(raw) as AuthUser
  } catch {
    return null
  }
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: getSavedUser(),
  token: getSavedToken(),
  refreshToken: getSavedRefreshToken(),
  isAuthenticated: Boolean(getSavedToken()),
  isLoading: false,

  setUser: (user) => {
    if (user) {
      localStorage.setItem('acs_user', JSON.stringify(user))
    } else {
      localStorage.removeItem('acs_user')
    }
    set({ user, isAuthenticated: Boolean(user) })
  },

  login: async (username: string, password: string) => {
    set({ isLoading: true })
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Authentication failed' }))
        return { success: false, error: err.detail || 'Login failed' }
      }

      const data = await response.json()
      localStorage.setItem('acs_access_token', data.access_token)
      localStorage.setItem('acs_refresh_token', data.refresh_token)
      localStorage.setItem('acs_user', JSON.stringify(data.user))

      set({
        token: data.access_token,
        refreshToken: data.refresh_token,
        user: data.user,
        isAuthenticated: true,
        isLoading: false,
      })

      return { success: true }
    } catch (e: any) {
      set({ isLoading: false })
      return { success: false, error: e.message || 'Network error during login' }
    }
  },

  refresh: async () => {
    const { refreshToken } = get()
    if (!refreshToken) return false

    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })

      if (!response.ok) {
        await get().logout()
        return false
      }

      const data = await response.json()
      localStorage.setItem('acs_access_token', data.access_token)
      if (data.refresh_token) {
        localStorage.setItem('acs_refresh_token', data.refresh_token)
      }
      set({
        token: data.access_token,
        refreshToken: data.refresh_token || refreshToken,
        user: data.user,
        isAuthenticated: true,
      })
      return true
    } catch {
      return false
    }
  },

  fetchCurrentUser: async () => {
    const { token } = get()
    if (!token) return

    try {
      const response = await fetch('/api/auth/me', {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (response.ok) {
        const user = await response.json()
        localStorage.setItem('acs_user', JSON.stringify(user))
        set({ user, isAuthenticated: true })
      } else if (response.status === 401) {
        const refreshed = await get().refresh()
        if (!refreshed) {
          await get().logout()
        }
      }
    } catch {
      // Offline or network error
    }
  },

  logout: async () => {
    const { token } = get()
    try {
      if (token) {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
        })
      }
    } catch {
      // Ignore network errors on logout
    }

    localStorage.removeItem('acs_access_token')
    localStorage.removeItem('acs_refresh_token')
    localStorage.removeItem('acs_user')

    set({
      token: null,
      refreshToken: null,
      user: null,
      isAuthenticated: false,
      isLoading: false,
    })
  },
}))

/**
 * Returns authorization headers with bearer token if authenticated.
 */
export const getAuthHeaders = (): Record<string, string> => {
  const token = localStorage.getItem('acs_access_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}
