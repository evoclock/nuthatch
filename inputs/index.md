---
title: Corpus Index
description: Content catalog for the nuthatch corpus
tags: [index]
---

# Corpus Index

## Papers

```dataview
TABLE title, year, authors
FROM "cards"
WHERE type = "paper"
SORT year ASC, title ASC
```

## Communities

```dataview
TABLE community_id, n_members
FROM "communities"
SORT community_id ASC
```

## Tags

```dataview
LIST
FROM "cards"
FLATTEN tags as tag
GROUP BY tag
SORT tag ASC
```
