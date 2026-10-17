import { describe, it, expect } from 'vitest'
import en from './en.json'
import ar from './ar.json'

function getDeepKeys(obj: Record<string, any>, prefix = ''): string[] {
  return Object.keys(obj).reduce((res: string[], el: string) => {
    const key = prefix ? `${prefix}.${el}` : el
    if (typeof obj[el] === 'object' && obj[el] !== null && !Array.isArray(obj[el])) {
      res.push(...getDeepKeys(obj[el], key))
    } else {
      res.push(key)
    }
    return res
  }, [])
}

function getValueByPath(obj: Record<string, any>, path: string): string | undefined {
  return path.split('.').reduce((acc, part) => acc && acc[part], obj) as string | undefined
}

describe('i18n Locale Key Parity & Integrity', () => {
  const enKeys = getDeepKeys(en).sort()
  const arKeys = getDeepKeys(ar).sort()

  it('has identical top-level and nested translation keys in both en and ar', () => {
    const missingInAr = enKeys.filter((k) => !arKeys.includes(k))
    const missingInEn = arKeys.filter((k) => !enKeys.includes(k))

    expect(missingInAr, `Keys present in en.json but missing in ar.json: ${missingInAr.join(', ')}`).toEqual([])
    expect(missingInEn, `Keys present in ar.json but missing in en.json: ${missingInEn.join(', ')}`).toEqual([])
    expect(enKeys).toEqual(arKeys)
  })

  it('contains non-empty translation strings for all keys in both locales', () => {
    for (const key of enKeys) {
      const enVal = getValueByPath(en, key)
      const arVal = getValueByPath(ar, key)

      expect(enVal, `Empty or missing English translation for key: ${key}`).toBeTruthy()
      expect(typeof enVal).toBe('string')
      expect(enVal!.trim().length).toBeGreaterThan(0)

      expect(arVal, `Empty or missing Arabic translation for key: ${key}`).toBeTruthy()
      expect(typeof arVal).toBe('string')
      expect(arVal!.trim().length).toBeGreaterThan(0)
    }
  })

  it('preserves all interpolation placeholders between en and ar', () => {
    const placeholderRegex = /\{\{\s*(\w+)\s*\}\}/g

    for (const key of enKeys) {
      const enVal = getValueByPath(en, key) || ''
      const arVal = getValueByPath(ar, key) || ''

      const enPlaceholders = Array.from(enVal.matchAll(placeholderRegex), (m) => m[1]).sort()
      const arPlaceholders = Array.from(arVal.matchAll(placeholderRegex), (m) => m[1]).sort()

      expect(
        arPlaceholders,
        `Placeholder mismatch in key "${key}": EN has [${enPlaceholders}] but AR has [${arPlaceholders}]`
      ).toEqual(enPlaceholders)
    }
  })
})
