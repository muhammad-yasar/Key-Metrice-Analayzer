# KMA Annotation Guide for Policy Reviewers

This guide explains how to review and correct the automated annotations
produced by the KMA pipeline in the Excel output files.

---

## Opening the File

Each policy produces one Excel file with three sheets:
- **Summary** — statistics and label colour legend
- **Annotations** — sentences that were classified (fill this in)
- **Unlabelled** — sentences that were skipped (optional review)

Work in the **Annotations** sheet only.

---

## Columns to Fill

### `Correct? (Y/N/Partial)`

| Value | Meaning |
|---|---|
| `Y` | Both Level and Class are correct |
| `N` | Both are wrong — skip this row entirely |
| `Partial` | One is correct but the other needs fixing |

### `Notes`

Only needed when you mark `Partial`. Write the correction in plain English.
The system will automatically parse your note. Examples:

| Your note | What the system understands |
|---|---|
| `Policy action rather than outcome` | Changes Level to Policy Action |
| `Policy Outcome, rather than action` | Changes Level to Policy Outcome |
| `Should be Spending` | Changes Class to Spending |
| `Environmental quality rather than miscellaneous` | Changes Class to Environment Quality |
| `Outcome rather than Action` | Changes Level to Policy Outcome |
| `This is Area not Site Status` | Changes Class to Area |
| `Knowledge resource` | Changes Class to Knowledge Resource |
| `Better as Spending` | Changes Class to Spending |

**You do not need to use any special format** — just write naturally.

---

## Level Definitions

### Policy Action ✓
A direct commitment where a **named actor** (Government, DAFM, EPA, etc.)
uses **shall, will, must, requires** to state what they will do.

**Examples:**
- "The Government **shall** restore 40,000 ha of peatland **by 2027**."
- "DAFM **will** establish a national database of peatland sites."
- "Natural England **will** publish an Implementation Plan by Summer 2022."

**NOT Policy Action if:** the sentence uses "could", "may", "might", "aims to",
"will be considered", or has no named actor.

---

### Policy Outcome ✓
A **goal or result** the policy aims to achieve — describes the end state,
not the specific action to get there.

**Examples:**
- "Peatland restoration will enable peatlands to meet their Net Zero contribution."
- "By 2030, we want all of England's soils to be managed sustainably."
- "Our peatlands will be healthy, well-functioning ecosystems rich in wildlife."

---

### Unsure ✓
Anything that is vague, hedged, aspirational, background description,
or historical context. When in doubt, mark Unsure.

**Examples:**
- "Forests **can** help to provide temporary mitigation of climate change."
- "Restoration **may** be supported through new funding schemes."
- "**As of 2013**, the total renewable generation was 2,100 MW."
- "Peatlands **have been** in the Irish landscape since the last Ice Age."

---

## Class Definitions

### Area
Land area, geographic scope, hectare targets.
> "Restore **40,000 ha** of peatland" / "**21% of national land area**"

**NOT Area:** if sentence is about guidance/plans — that's Knowledge Resource.

---

### Emissions
A **specific CO2/GHG reduction target** with a percentage or deadline.
> "reduce emissions by **80% by 2050**" / "**carbon neutral** by 2045"

**NOT Emissions:** general mentions of climate change or carbon — that's Environment Quality.

---

### Site Status
Named protected sites, SACs, NHAs, SPAs, or restoration of specific designated areas.
> "**raised bog SACs**" / "**75 NHAs** designated under Wildlife Acts" / "**Natura 2000** network"

---

### Spending
Actual money amounts, annual payments, compensation figures.
> "€**1,500** per annum" / "invest over £**8 million**" / "grant of **€5,000** per hectare"

**NOT Spending:** resource mobilisation without figures — that's Policy Action class.

---

### Policy Action (class)
Reference to another **law, regulation, or directive** that requires something.
> "As required by the **Habitats Directive**..." / "Under **Regulation (EU) 2018/841**..."

---

### Knowledge Resource
Producing a **document, database, plan, survey, or guidance**.
> "will **publish** an Implementation Plan" / "will **develop** a national database"
> "will **introduce guidance** on future management"

---

### Practical Resource
**Physical delivery** to people — compensation, relocation, training.
> "turf-cutters will be **provided with** compensation packages"
> "**training** will be delivered to peatland workers"

---

### Environment Quality
Water quality, habitat condition, biodiversity, carbon sequestration, rewetting.
> "**water quality** improvements" / "**biodiversity** targets" / "**rewetting** of blanket bogs"
> "**WFD** compliance" / "favourable **conservation status**"

---

### Miscellaneous
Genuinely doesn't fit any of the above. Use sparingly.

---

## Common Mistakes to Avoid

| Mistake | Correct approach |
|---|---|
| Marking "Forests can help..." as Policy Action | → Unsure (hedged with "can") |
| Marking "80% CO2 reduction" as Environment Quality | → Emissions (specific target) |
| Marking "introduce guidance" as Area | → Knowledge Resource |
| Marking "As of 2013..." as Policy Action | → Unsure (historical fact) |
| Marking "resource mobilisation" as Spending | → Policy Action class (no euro amount) |

---

## How Many to Review?

- Aim for **all rows** in the Annotations sheet
- If time is limited, prioritise rows marked with **KPI? = Y** (column H)
- Mark `N` for obvious garbage (page headers, footnotes, table data)
- It is better to mark `N` than to guess on unclear sentences

---

## Questions?

Contact: Dr. Muhammad Yasar Khan — yasar.khan@universityofgalway.ie
