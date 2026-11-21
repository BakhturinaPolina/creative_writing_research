# Lexicon pilot smoke (v0.2-draft)

Built: 2026-11-21 02:32
Lexicon: `config/pov_lexicon_psyper_en_de_v0.2.yml` (0.2-draft)

Rule: a passage is a *literal candidate* if any academic/high term or regex hits,
or ≥2 supporting craft terms hit. Context-only never counts. Paraphrases are not
used as literal frequency features here.

## [PASS] en: Choose a first-person narrator who is not the main character.
- reason/expected: {'choice_tradeoffs', 'grammatical_first_person'}
- high hits: ['grammatical_first_person:first-person narrator', 'regex:grammatical_person']
- supporting hits: —
- literal_candidate: True

## [PASS] en: From an economic point of view, the contract is advantageous.
- reason/expected: Everyday sense of point of view.
- high hits: ['regex:general_pov']
- supporting hits: ['perspective_general:point of view']
- literal_candidate: False

## [CHECK] en: Stay inside one character's head for the entire scene.
- reason/expected: {'focalization_internal', 'focalization_fixed'}
- high hits: —
- supporting hits: —
- literal_candidate: False

## [PASS] de: Bei variabler Fokalisierung wechseln die begrenzten Wissenshorizonte der Figuren.
- reason/expected: {'focalization_variable', 'focalization_general'}
- high hits: ['focalization_general:Fokalisierung', 'regex:focalization', 'regex:focalization_variable']
- supporting hits: —
- literal_candidate: True

## [CHECK] de: Die Erzählerin ist eine Nebenfigur und wirkt am Geschehen mit.
- reason/expected: {'voice_participant', 'voice_homodiegetic'}
- high hits: —
- supporting hits: —
- literal_candidate: False

## [PASS] de: Aus finanzieller Sicht hat die Entscheidung Vorteile.
- reason/expected: Everyday meaning; no literary context.
- high hits: —
- supporting hits: —
- literal_candidate: False

## [CHECK] de: Szenisches Erzählen kann auch mit auktorialem Erzählen kombiniert werden.
- reason/expected: {'mode_dramatic', 'stance_narratorial'}
- high hits: —
- supporting hits: ['mode_dramatic:szenisches Erzählen']
- literal_candidate: False

## [PASS] en: From an economic point of view, the contract is advantageous.
- reason/expected: everyday_pov
- high hits: ['regex:general_pov']
- supporting hits: ['perspective_general:point of view']
- literal_candidate: False

## [PASS] en: The marketing voice of the brand should stay consistent.
- reason/expected: everyday_voice
- high hits: —
- supporting hits: —
- literal_candidate: False

## [PASS] de: Aus finanzieller Sicht hat die Entscheidung Vorteile.
- reason/expected: everyday_sicht
- high hits: —
- supporting hits: —
- literal_candidate: False

## [PASS] de: Die Perspektive der Anleger ist entscheidend.
- reason/expected: everyday_perspektive
- high hits: —
- supporting hits: —
- literal_candidate: False

**Summary:** 8 PASS, 0 FAIL, 3 CHECK
