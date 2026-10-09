import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import ar from './ar.json'
import en from './en.json'

const savedLanguage = localStorage.getItem('acs_language') || 'en'

// Apply language and dir attribute to <html> immediately on bootstrap
export const applyLanguageDirection = (lang: string) => {
  document.documentElement.lang = lang
  document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr'
}

applyLanguageDirection(savedLanguage)

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    ar: { translation: ar },
  },
  lng: savedLanguage,
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false,
  },
})

i18n.on('languageChanged', (lang) => {
  localStorage.setItem('acs_language', lang)
  applyLanguageDirection(lang)
})

export default i18n
