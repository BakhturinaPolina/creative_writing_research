# PsyPer-related tags versus project retrieval concepts (v0.5)

This project uses a **36-concept retrieval lexicon** informed by PsyPer-related
annotation interests. **Not all 36 retrieval concepts are official PsyPer categories.**

PsyPer mappings in the YAML (`psyper:` lists) indicate *potential relevance* to
annotation codes — not logical equivalence, and not an official PsyPer scheme dump.

Original PsyPer coding manuals are **not** present as standalone documents in this
repository. Available in-repo materials:

- `config/pov_lexicon_psyper_en_de_v0.3.yml` — retrieval lexicon with `psyper` hints
- `docs/scientific_readme.md` §9.1 — project POV taxonomy dimensions
- `docs/scientific_readme.md` §10 — project annotation relevance/context/function labels

## Project retrieval concepts (36)

| concept_id | psyper_related_tags | note |
|---|---|---|
| `perspective_general` | A, B, C, D, E, F | Umbrella terminology; no specific category inferred. |
| `voice_general` | A, B, C | Narrative voice and identity of the narrator; voice can also refer to style. |
| `grammatical_first_person` | A, B | Grammatical person is evidence, not sufficient for homodiegetic status. |
| `grammatical_second_person` | A, B | Additional grammatical-person dimension; not a PsyPer voice class. |
| `grammatical_third_person` | A, B, D, E | Third person is not equivalent to heterodiegetic or unrestricted narration. |
| `voice_heterodiegetic` | 1, 1.1 | Detached means not a participant; not necessarily unemotional, objective or omniscient. |
| `voice_homodiegetic` | 2, 2.2 |  |
| `voice_eyewitness` | 2.2.1 |  |
| `voice_participant` | 2.2.2 |  |
| `voice_protagonist` | 2.2.3 |  |
| `stance_narratorial` | 3 | Authorial narration is a related historical/craft cue, not an exact equivalent. |
| `stance_figural` | 4 | Thought representation and character-centred language are related cues; distinguish from information access. |
| `focalization_general` | D, E |  |
| `focalization_zero` | 5, 5.1 | Omniscience and auktorial are related cues, not invariably zero focalization. |
| `focalization_internal` | 6 | Limited third person is a craft cue; first-person narration is not necessarily internally focalized. |
| `focalization_external` | 7, 7.1 | No access to inner states; 'showing' or an objective tone alone is insufficient. |
| `focalization_fixed` | 6.1 |  |
| `focalization_variable` | 6.2 | Several limited perspectives do not by themselves establish omniscience; multiple telling of same event is separate. |
| `pov_subject` | 8 |  |
| `pov_object` | 9 | A character can be an object of depiction within another character's internal perspective. |
| `mode_narrative` | 10, 11 | Search discussion of telling; lexical hits cannot distinguish ordinal levels 10 and 11. |
| `mode_dramatic` | 13, 14 | Search discussion of showing/scenic presentation; direct speech alone does not establish dramatic mode. |
| `mode_balance` | 10, 11, 12, 13, 14 | Discussion of the continuum, not automatic assignment of Mixed. |
| `consciousness_access` | 6, 7, 7.1, 8 |  |
| `thought_representation` | 3, 4, 6, 8 |  |
| `distance_depth` | 3, 4, 6, 8 | Related extra dimension: distance is not itself a PsyPer tag. |
| `switching_consistency` | 6.1, 6.2, 8, 9 |  |
| `multiple_focalization_same_event` | 6.2 | Related Genette category, not a distinct PsyPer tag; distinguish from alternating viewpoints. |
| `information_control` | 5, 5.1, 6, 6.1, 6.2, 7 |  |
| `reliability` | *(none / project-specific)* | Legacy/project extension independent of voice participation and focalization. |
| `narrator_level_time` | A, B, C | Additional dimensions; diegetic level and narrating time do not determine participation. |
| `filter_words` | 4, 6, 8 | Craft advice especially useful for deep/close POV; plain perception verbs are not retrieval hits alone. |
| `reader_effect` | *(none / project-specific)* | Related outcome dimension; require explicit connection to perspective, do not infer effects from narrator type. |
| `speech_dialogue` | 7, 10, 11, 12, 13, 14 | Related signal: discussion of dialogue is not necessarily discussion of POV or mode. |
| `narrative_style_voice` | C | Legacy extension; writer's voice and narrator's voice must be distinguished. |
| `choice_tradeoffs` | A, B, C, D, E, F, G | Meta-discussion about selecting a perspective; multi-label discourse cue. |

## Distinctions preserved for human audit

Do not automatically equate:

- Grammatical person vs narrator participation
- Narratorial stance vs focalization
- Information access vs reader knowledge
- Stanzel *auktoriale / personale Erzählsituation* vs Genette zero / internal focalization
- *Wissensvorsprung* / dramatic irony vs focalization (may relate to `information_control` / `reader_effect`)

No conceptual remaps were applied in v0.5 pending human audit decisions.
