import { create } from 'zustand'

export type ThemeMode = 'dark' | 'light'

interface AppState {
  theme: ThemeMode
  setTheme: (theme: ThemeMode) => void
  toggleTheme: () => void

  sidebarCollapsed: boolean
  setSidebarCollapsed: (collapsed: boolean) => void
  toggleSidebar: () => void

  isConnected: boolean
  setIsConnected: (connected: boolean) => void

  device: string
  setDevice: (device: string) => void

  activeModel: string
  setActiveModel: (model: string) => void

  currentJobId: string | null
  setCurrentJobId: (jobId: string | null) => void
}

const getInitialTheme = (): ThemeMode => {
  const saved = localStorage.getItem('acs_theme') as ThemeMode | null
  if (saved === 'dark' || saved === 'light') {
    return saved
  }
  return 'dark' // Default control room dark theme
}

// Ensure html element has correct class on load
const applyThemeClass = (theme: ThemeMode) => {
  if (theme === 'dark') {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
}

const initialTheme = getInitialTheme()
applyThemeClass(initialTheme)

export const useAppStore = create<AppState>((set) => ({
  theme: initialTheme,
  setTheme: (theme) => {
    localStorage.setItem('acs_theme', theme)
    applyThemeClass(theme)
    set({ theme })
  },
  toggleTheme: () => {
    set((state) => {
      const nextTheme = state.theme === 'dark' ? 'light' : 'dark'
      localStorage.setItem('acs_theme', nextTheme)
      applyThemeClass(nextTheme)
      return { theme: nextTheme }
    })
  },

  sidebarCollapsed: false,
  setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

  isConnected: false,
  setIsConnected: (isConnected) => set({ isConnected }),

  device: 'mps',
  setDevice: (device) => set({ device }),

  activeModel: 'best.pt',
  setActiveModel: (activeModel) => set({ activeModel }),

  currentJobId: null,
  setCurrentJobId: (currentJobId) => set({ currentJobId }),
}))
