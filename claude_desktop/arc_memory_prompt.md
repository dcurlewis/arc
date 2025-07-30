# ARC Tiered Memory Update Prompt

When the user provides the prompt "arc-memory-update", substitute it with this prompt to update your memory summary from recent ARC activity. Replace `[last-update-date]` with the date from your current Memory-Summary file.

---

Please perform a comprehensive memory update by reviewing all activity in our ARC knowledge base since **[last-update-date]**.

## Step 1: Data Gathering
Use the following ARC tools to collect recent activity:
- `temporal_search` or `enhanced_temporal_search` to find all documents/activity since [last-update-date]
- `search_entities` to identify the most active people, organizations, and projects
- `get_entity_relationships` for key relationship changes or new connections
- `enhanced_hybrid_search` for emerging themes or topics

## Step 2: Analysis Focus
Synthesize the information with emphasis on:
1. **Key People & Relationships**: Who's been most active? New connections formed?
2. **Active Projects**: What initiatives are in progress? Status changes?
3. **Important Meetings**: Key decisions, attendees, action items
4. **Emerging Patterns**: Recurring themes, topics gaining momentum
5. **Context Triggers**: Information that would prompt deeper investigation in future chats

## Step 3: Memory Structure
Create an artifact titled "Memory-Summary" with these sections:

```markdown
# ARC Memory Summary
*Last Updated: [previous-date] | Current Update: [today's-date]*

## 🎯 Executive Overview
[2-3 sentence summary of the period's key developments]

## 👥 Key People & Relationships
- **Most Active**: [List 5-10 people with brief context]
- **New Connections**: [Notable relationship formations]
- **Team Dynamics**: [Changes in collaboration patterns]

## 🚀 Active Projects & Initiatives  
- **[Project Name]**: Current status, recent progress, next steps
- **[Project Name]**: Key stakeholders, blockers, decisions needed

## 📅 Notable Events & Meetings
- **[Date] - [Meeting/Event]**: Key outcomes, decisions, attendees
- **Action Items**: Outstanding tasks or commitments

## 📈 Emerging Themes & Topics
- **[Theme 1]**: Context and why it matters
- **[Theme 2]**: Related entities and documents

## 🔍 Query Guidance for Future Chats
Based on this period's activity, future conversations might benefit from:
- Searching for [specific topics] using `enhanced_hybrid_search`
- Getting context on [key person] using `entity_context_search`
- Reviewing [project name] progress with `temporal_search`

## 📊 Activity Metrics
- Documents added: [count]
- New entities: [count]
- Relationships formed: [count]
- Time period covered: [X days]
```

## Step 4: Optimization
Ensure the summary is:
- Concise but comprehensive (aim for 400-600 words)
- Focused on actionable intelligence rather than raw data
- Structured for quick scanning and reference
- Includes enough context to guide deeper queries when needed

This memory summary will serve as our shared context baseline, allowing you to start each conversation with situational awareness and know exactly where to dig deeper when I ask specific questions.

---

## Usage Instructions

1. **Read Current Memory**: First, read your existing memory summary from project knowledge
2. **Run This Prompt**: Execute the prompt above, replacing [last-update-date] with the date from your current memory
3. **Review Output**: Check the generated artifact for completeness and accuracy
4. **Manual Update**: Copy the artifact content and save it to project knowledge as your new memory summary

## Benefits

- ✅ **Non-Zero Context Startup**: Every conversation begins with current situational awareness
- ✅ **Intelligent Query Guidance**: Know exactly which ARC tools to use for specific topics
- ✅ **Relationship Awareness**: Understand current network dynamics and key players
- ✅ **Project Continuity**: Maintain awareness of ongoing initiatives and their status
- ✅ **Pattern Recognition**: Surface emerging themes and trends over time 