---
title: Corpus Dashboard
description: Fast access patterns for the active nuthatch corpus
tags: [dashboard]
---

# Corpus Dashboard

> Use Cmd+G to open the graph view, or run the Dataview queries below
> for filtered lookups. Each query block is a starting point; edit
> per your corpus's tag vocabulary.

## Recently ingested papers

```dataview
TABLE year, authors, ingested
FROM "cards"
WHERE type = "paper"
SORT ingested DESC
LIMIT 25
```

## Highest-relevance papers

```dataview
TABLE relevance, year, authors
FROM "cards"
WHERE type = "paper"
SORT relevance DESC
LIMIT 25
```

## Papers by community

```dataview
TABLE community_id, year, authors
FROM "cards"
WHERE type = "paper" AND community_id != null
SORT community_id ASC, year ASC
```

## Communities

```dataview
TABLE n_members
FROM "communities"
SORT n_members DESC
```

## Pending review (status = exploratory)

```dataview
TABLE year, authors, ingested
FROM "cards"
WHERE status = "exploratory"
SORT ingested ASC
```
